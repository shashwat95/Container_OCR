import cv2
import multiprocessing as mp
import torch
import numpy as np
import os
import logging
import time
from datetime import datetime
from rapidocr_onnxruntime import RapidOCR
from ultralytics import YOLO
from typing import Dict, Any, Optional

class VideoPipeline:
    # Add at class level
    QUEUE_TIMEOUT = 5  # Seconds to wait for queue input
    MAX_CONSECUTIVE_EMPTY = 6  # About 30 seconds of empty checks (5*6)
    
    def __init__(self, camera_configs: list, test_mode: bool = False):
        self.camera_configs = camera_configs
        self.test_mode = test_mode
        self.global_shutdown = mp.Event()
        
        # Initialize output directory early
        if self.test_mode:
            self.output_dir = os.path.join('debug_output', datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create a directory for visualization frames
            self.visualization_dir = os.path.join(self.output_dir, 'visualization_frames')
            os.makedirs(self.visualization_dir, exist_ok=True)  # Create the directory for visualization frames

        self._setup_shared_state()
        self._validate_configs()
        self._setup_logging() if self.test_mode else None

    def _start_logger_process(self):
        """Create and start dedicated logging process"""
        if not self.test_mode:
            return None
            
        logger_proc = mp.Process(
            target=self._logger_worker,
            name="LoggerProcess",
            daemon=True
        )
        logger_proc.start()
        return logger_proc

    def _logger_worker(self):
        """Dedicated logging process worker"""
        while not self.global_shutdown.is_set():
            try:
                level, message = self.log_queue.get(timeout=1)
                if level == 'debug':
                    self.logger.debug(message)
                elif level == 'info':
                    self.logger.info(message)
                elif level == 'warning':
                    self.logger.warning(message)
                elif level == 'error':
                    self.logger.error(message)
            except mp.queues.Empty:
                continue
            except Exception as e:
                print(f"Logger error: {str(e)}")

    def _setup_shared_state(self):
        """Initialize shared resources and synchronization primitives"""
        self.manager = mp.Manager()
        self.tracked_objects = self.manager.dict()
        self.frame_queues = {cfg['camera_id']: mp.Queue(maxsize=30) for cfg in self.camera_configs}
        self.tracking_lock = mp.Lock()
        self.resolution_dict = self.manager.dict()
        self.log_queue = mp.Queue() if self.test_mode else None

    def _validate_configs(self):
        """Validate camera configuration structure"""
        required_keys = ['rtsp_url', 'camera_id', 'location']
        for cfg in self.camera_configs:
            if not all(k in cfg for k in required_keys):
                raise ValueError(f"Invalid camera config: {cfg}")
            if 'roi_points' not in cfg:
                cfg['roi_points'] = np.array([[273,1072],[533,841],[1409,807],[1856,1066]], np.int32)
    
    def _monitor_processes(self, processes: list):
        """Monitor process health and trigger shutdown if any process dies"""
        for p in processes:
            if not p.is_alive():
                self.log("error", f"Process {p.name} ({p.pid}) died unexpectedly")
                self.global_shutdown.set()
                break
                
    def start(self):
        """Start the entire pipeline with process monitoring"""
        logger_proc = None
        try:
            capture_procs = self._start_capture_processes()
            process_procs = self._start_processing_processes()
            
            if self.test_mode:
                logger_proc = self._start_logger_process()
                self.log("info", "Pipeline started in test mode")

            while not self.global_shutdown.is_set():
                self._monitor_processes(capture_procs + process_procs)
                time.sleep(1)

        except KeyboardInterrupt:
            self.log("info", "Shutdown signal received")
        finally:
            self._cleanup_resources()
            if logger_proc:
                logger_proc.join(timeout=1)

    def _start_capture_processes(self):
        """Launch frame capture processes with error handling"""
        processes = []
        for cfg in self.camera_configs:
            p = mp.Process(
                target=self.capture_worker,
                args=(cfg['rtsp_url'], cfg['camera_id'], cfg['roi_points']),
                name=f"Capture_{cfg['camera_id']}"
            )
            p.start()
            processes.append(p)
            self.log("debug", f"Started capture process for {cfg['camera_id']}")
        return processes

    def _start_processing_processes(self):
        """Launch processing processes with isolated resources"""
        processes = []
        for cfg in self.camera_configs:
            p = mp.Process(
                target=self.processing_worker,
                args=(cfg['camera_id'], cfg['roi_points'], self.test_mode),
                name=f"Process_{cfg['camera_id']}"
            )
            p.start()
            processes.append(p)
            self.log("debug", f"Started processing process for {cfg['camera_id']}")
        return processes

    def capture_worker(self, rtsp_url: str, camera_id: str, roi_points: np.ndarray):
        """Frame capture worker with proper stream end detection and reconnection logic"""
        reconnect_attempts = 0
        cap = None
        last_frame_time = time.time()
        
        try:
            while not self.global_shutdown.is_set() and reconnect_attempts < 5:
                try:
                    # Initialize video capture with hardware acceleration
                    cap = cv2.VideoCapture(rtsp_url)
                    if not cap.isOpened():
                        raise IOError(f"Failed to open stream: {rtsp_url}")
                    
                    # Set decoder preferences if available
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                    
                    # Update shared resolution information
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.resolution_dict[camera_id] = (width, height)
                    self.log("debug", f"Stream opened: {width}x{height}")

                    while not self.global_shutdown.is_set():
                        ret, frame = cap.read()
                        if not ret:
                            self.log("warning", f"Empty frame received from {camera_id}")
                            break

                        try:
                            # Put frame in queue with timestamp and ROI
                            self.frame_queues[camera_id].put(
                                (frame.copy(), datetime.now(), roi_points), 
                                timeout=1
                            )
                            reconnect_attempts = 0  # Reset on successful frame
                            last_frame_time = time.time()
                        except mp.queues.Full:
                            self.log("warning", f"Queue full for {camera_id}, dropping frame")
                            continue

                except Exception as e:
                    self.log("error", f"Capture error {camera_id}: {str(e)}")
                    reconnect_attempts += 1
                    sleep_time = min(2 ** reconnect_attempts, 30)  # Exponential backoff
                    self.log("debug", f"Reconnecting in {sleep_time}s (attempt {reconnect_attempts}/5)")
                    time.sleep(sleep_time)

                finally:
                    if cap and cap.isOpened():
                        cap.release()
                    time.sleep(0.1)  # Prevent tight loop on failure

        except Exception as e:
            self.log("error", f"Fatal capture error {camera_id}: {str(e)}")
        
        finally:
            # Signal stream end to processing workers
            try:
                self.frame_queues[camera_id].put((None, None, None), timeout=1)
                self.log("debug", f"Sent stream end signal for {camera_id}")
            except:
                pass
            
            if cap and cap.isOpened():
                cap.release()
            self.log("info", f"Capture worker for {camera_id} terminated")

    def _is_completely_inside(self, outer_box: list, inner_box: list) -> bool:
        """Check if inner bounding box is completely contained within outer box"""
        outer_x1, outer_y1, outer_x2, outer_y2 = outer_box
        inner_x1, inner_y1, inner_x2, inner_y2 = inner_box
        
        return (inner_x1 >= outer_x1 and 
                inner_y1 >= outer_y1 and 
                inner_x2 <= outer_x2 and 
                inner_y2 <= outer_y2)

    def processing_worker(self, camera_id: str, roi_points: np.ndarray, test_mode: bool):
        """Main processing worker with stream end detection"""
        # Initialize local resources
        model = YOLO('/Workspace/sg/Mycodes/Yolo/runs/detect/train138/weights/best.pt').to(self._get_device())
        ocr_engine = RapidOCR()
        consecutive_empty = 0
        
        try:
            if test_mode:
                self.log("debug", f"Processing worker started for {camera_id}")

            while not self.global_shutdown.is_set():
                try:
                    frame, timestamp, roi = self.frame_queues[camera_id].get(
                        timeout=self.QUEUE_TIMEOUT
                    )
                    consecutive_empty = 0  # Reset counter on successful frame get

                    # Process frame
                    processed = self._process_frame(frame, model, roi, camera_id, timestamp)
                    
                    # Update tracking state
                    with self.tracking_lock:
                        self._update_tracking_state(camera_id, processed)
                    
                    # OCR processing
                    if processed.get('best_detection'):
                        self._process_ocr(frame, processed['best_detection'], ocr_engine)

                        # Save visualization frame if a new best detection is found
                        debug_frame = self._create_debug_frame(frame, processed, camera_id, roi)
                        image_filename = os.path.join(self.visualization_dir, f"{camera_id}_best_detection_{timestamp.strftime('%Y%m%d_%H%M%S')}.png")
                        cv2.imwrite(image_filename, debug_frame)  # Save the frame as an image
                        self.log("debug", f"Saved visualization frame to {image_filename}")

                except mp.queues.Empty:
                    consecutive_empty += 1
                    if test_mode and consecutive_empty >= self.MAX_CONSECUTIVE_EMPTY:
                        self.log("debug", f"No frames for {self.MAX_CONSECUTIVE_EMPTY} checks. Closing worker.")
                        break
                    continue
                    
                except Exception as e:
                    self.log("error", f"Processing error {camera_id}: {str(e)}")
                    break

        finally:
            if model:
                del model
            torch.cuda.empty_cache()

    def _process_frame(self, frame: np.ndarray, model: YOLO, roi: np.ndarray, 
                     camera_id: str, timestamp: datetime) -> dict:
        """Process a single frame through the pipeline"""
        processed = {'detections': [], 'best_detection': None, 'timestamp': timestamp}
        
        if frame is None:  # Handle stream end signals
            return processed

        # Run object detection
        with torch.no_grad():
            results = model.track(frame, persist=True, device=model.device)

        if not results or not results[0].boxes:
            return processed

        # Convert tensors to numpy safely
        try:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
        except Exception as e:
            self.log("error", f"Tensor conversion failed: {str(e)}")
            return processed

        # Process detections
        class0_detections = []
        class1_detections = []
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
            detection = {
                'box': box.tolist(),
                'confidence': float(conf),
                'class_id': int(cls),
                'track_id': track_ids[i] if i < len(track_ids) else -1,
                'in_roi': self._is_in_roi(box, roi)
            }
            
            if cls == 0 and detection['in_roi']:
                class0_detections.append(detection)
            elif cls == 1:
                class1_detections.append(detection)

        # Process associations between class0 and class1
        for class0 in class0_detections:
            class0_box = class0['box']
            potential_associations = []
            
            # Find class1 detections completely inside class0's box
            for class1 in class1_detections:
                if self._is_completely_inside(class0_box, class1['box']):
                    potential_associations.append(class1)
            
            # Select best association
            if potential_associations:
                best_class1 = max(potential_associations, key=lambda x: x['confidence'])
                class0['associated_class_1'] = best_class1
            else:
                class0['associated_class_1'] = None

        processed['detections'] = class0_detections + class1_detections
        
        # Find best detection from tracked objects
        best_detection = None
        max_confidence = 0.0
        
        # Check both current frame and historical associations
        for class0 in class0_detections:
            if class0['associated_class_1'] and class0['associated_class_1']['confidence'] > max_confidence:
                best_detection = class0['associated_class_1']
                max_confidence = best_detection['confidence']
        
        # Check historical tracked objects
        current_time = datetime.now()
        for key in list(self.tracked_objects.keys()):
            if key[0] != camera_id:
                continue
                
            obj = self.tracked_objects[key]
            if (current_time - obj['last_seen']).total_seconds() < 5:  # 5-second activity window
                if obj['associated_class_1'] and obj['associated_class_1']['confidence'] > max_confidence:
                    best_detection = obj['associated_class_1']
                    max_confidence = best_detection['confidence']

        processed['best_detection'] = best_detection
        return processed

    def _update_tracking_state(self, camera_id: str, processed: dict):
        """Update shared tracking state with thread-safe locking"""
        current_time = processed['timestamp']
        for detection in processed['detections']:
            key = (camera_id, detection['track_id'])
            
            if detection['class_id'] == 0:
                # Handle class 0 objects with associations
                if key not in self.tracked_objects:
                    self.tracked_objects[key] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'confidence_history': [detection['confidence']],
                        'positions': [detection['box']],
                        'associated_class_1': detection.get('associated_class_1', None)
                    }
                else:
                    # Update existing tracking info
                    self.tracked_objects[key]['last_seen'] = current_time
                    self.tracked_objects[key]['confidence_history'].append(detection['confidence'])
                    self.tracked_objects[key]['positions'].append(detection['box'])
                    
                    # Update association if better than existing
                    current_assoc = self.tracked_objects[key]['associated_class_1']
                    new_assoc = detection.get('associated_class_1')
                    if new_assoc and (not current_assoc or new_assoc['confidence'] > current_assoc['confidence']):
                        self.tracked_objects[key]['associated_class_1'] = new_assoc

    def _is_in_roi(self, box: list, roi: np.ndarray) -> bool:
        """Check if detection is within region of interest"""
        x_center = (box[0] + box[2]) / 2
        y_bottom = box[3]
        return cv2.pointPolygonTest(roi, (x_center, y_bottom), False) >= 0

    def _create_debug_frame(self, frame: np.ndarray, processed: dict, 
                        camera_id: str, roi: np.ndarray) -> np.ndarray:
        """Create annotated debug frame with tracking info"""
        debug_frame = frame.copy()
        
        # Draw ROI and detections
        cv2.polylines(debug_frame, [roi], True, (0, 255, 255), 2)
        
        # Draw all detections
        for det in processed['detections']:
            color = (0, 255, 0) if det['in_roi'] else (0, 0, 255)
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{det['track_id']} Cls:{det['class_id']} Conf:{det['confidence']:.2f}"
            cv2.putText(debug_frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Highlight best detection
        if processed['best_detection']:
            best = processed['best_detection']
            x1, y1, x2, y2 = map(int, best['box'])
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(debug_frame, "BEST", (x1, y1-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Show OCR result if available
            if 'ocr_text' in best:
                text_x = debug_frame.shape[1] - 10 - cv2.getTextSize(best['ocr_text'], 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
                cv2.putText(debug_frame, best['ocr_text'], (text_x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return debug_frame

    def _process_ocr(self, frame: np.ndarray, detection: dict, ocr_engine: RapidOCR):
        """Perform OCR on detected region"""
        x1, y1, x2, y2 = map(int, detection['box'])
        cropped = frame[y1:y2, x1:x2]
        
        if cropped.size == 0:
            return
            
        try:
            # Convert to JPEG buffer for RapidOCR
            _, buffer = cv2.imencode('.jpg', cropped)
            result, _ = ocr_engine(buffer.tobytes(), use_det=False, use_cls=False, use_rec=True, rec_use_cuda=True)
            if result:
                detection['ocr_text'] = result[0][0]
        except Exception as e:
            self.log("error", f"OCR failed: {str(e)}")

    def _setup_logging(self):
        """Configure centralized logging with rotation"""
        self.logger = logging.getLogger('VideoPipeline')
        self.logger.setLevel(logging.DEBUG)
        
        # Only add handlers once
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            # Rotating file handler (10MB per file, max 5 files)
            file_handler = logging.handlers.RotatingFileHandler(
                os.path.join(self.output_dir, 'pipeline.log'),
                maxBytes=10*1024*1024,
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Console handler for warnings and errors
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log(self, level: str, message: str):
        """Thread-safe logging through queue"""
        if self.test_mode and self.log_queue:
            self.log_queue.put((level, message))

    def _cleanup_resources(self):
        """Clean up all resources and terminate processes"""
        self.global_shutdown.set()
        time.sleep(1)  # Allow processes to finish current operations
        
        # Clear all queues
        for q in self.frame_queues.values():
            while not q.empty():
                try:
                    q.get_nowait()
                except mp.queues.Empty:
                    break
        
        # Release CUDA memory
        torch.cuda.empty_cache()

    def _get_device(self):
        """Get available compute device"""
        return 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    pipeline = None
    try:
        camera_configs = [
            {
                'rtsp_url': 'rtsp://192.168.10.139:8554/stream101',
                'camera_id': 'cam1',
                'location': 'Entrance',
                'roi_points': np.array([[273,1072],[533,841],[1409,807],[1856,1066]], np.int32)
            },
            {
                'rtsp_url': 'rtsp://192.168.10.139:8554/stream102',
                'camera_id': 'cam2',
                'location': 'Entrance',
                'roi_points': np.array([[273,1072],[533,841],[1409,807],[1856,1066]], np.int32)
            },
            {
                'rtsp_url': 'rtsp://192.168.10.139:8554/stream103',
                'camera_id': 'cam2',
                'location': 'Entrance',
                'roi_points': np.array([[273,1072],[533,841],[1409,807],[1856,1066]], np.int32)
            }            
        ]
        
        pipeline = VideoPipeline(camera_configs, test_mode=True)
        pipeline.start()
        
    except KeyboardInterrupt:
        print("\nUser initiated shutdown")
    except Exception as e:
        print(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline:
            pipeline._cleanup_resources()
        print("System shutdown completed")