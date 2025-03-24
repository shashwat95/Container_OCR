import cv2
import threading
import torch
from ultralytics import YOLO
import numpy as np
import os
import logging
from datetime import datetime
from rapidocr_onnxruntime import RapidOCR  # Import RapidOCR
import time

class BatchProcessor:
    def __init__(self, model, device, batch_size=8, max_queue_size=32):
        """
        Initialize the batch processor for efficient inference.
        
        Args:
            model: YOLO model instance
            device: Computing device ('cuda' or 'cpu')
            batch_size: Maximum number of frames to process at once
            max_queue_size: Maximum queue size before processing
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.frame_queue = []
        self.metadata_queue = []
        self.processing_lock = threading.Lock()
        self.results_cache = {}
        self.ocr_engine = RapidOCR()
        
    def add_frame(self, frame, metadata):
        """
        Add a frame to the processing queue with its associated metadata.
        
        Args:
            frame: Video frame to process
            metadata: Dictionary containing frame metadata (camera_id, frame_id, etc.)
            
        Returns:
            Whether batch processing was triggered
        """
        with self.processing_lock:
            self.frame_queue.append(frame)
            self.metadata_queue.append(metadata)
            
            if len(self.frame_queue) >= self.max_queue_size:
                self.process_batch()
                return True
        return False
    
    def process_batch(self):
        """
        Process accumulated frames in batches for efficient inference.
        """
        if not self.frame_queue:
            return
        with self.processing_lock:
            # Process frames in batches
            for i in range(0, len(self.frame_queue), self.batch_size):
                batch_frames = self.frame_queue[i:i+self.batch_size]
                batch_metadata = self.metadata_queue[i:i+self.batch_size]
                
                # Run model inference on batch
                with torch.no_grad():  # Prevent gradient tracking
                    results = self.model.track(batch_frames, persist=True, device=self.device)
                
                # Store results with metadata
                for j, (result, metadata) in enumerate(zip(results, batch_metadata)):
                    camera_id = metadata['camera_id']
                    frame_id = metadata['frame_id']
                    self.results_cache[(camera_id, frame_id)] = {
                        'result': result,
                        'metadata': metadata,
                        'processed': False
                    }
                
                # Clear batch frames and metadata to free memory
                del batch_frames, batch_metadata
                
            # Clear queues after processing
            self.frame_queue = []
            self.metadata_queue = []
            self.clean_cache()  # Clean cache after processing
    
    def get_result(self, camera_id, frame_id):
        """
        Get processing result for a specific frame.
        
        Args:
            camera_id: Camera identifier
            frame_id: Frame identifier
            
        Returns:
            Processing result or None if not available
        """
        key = (camera_id, frame_id)
        if key in self.results_cache:
            result_data = self.results_cache[key]
            if not result_data['processed']:
                result_data['processed'] = True
                # Remove the processed frame from the frame queue
                with self.processing_lock:
                    self.frame_queue.pop(0)
                    self.metadata_queue.pop(0)
            return result_data['result']
        return None
    
    def clean_cache(self, max_cache_size=100):
        """
        Clean up processed results to prevent memory issues.
        
        Args:
            max_cache_size: Maximum number of results to keep in cache
        """
        if len(self.results_cache) > max_cache_size:
            # Find processed entries to remove
            processed_keys = [k for k, v in self.results_cache.items() if v['processed']]
            # Sort by frame_id to remove oldest first
            processed_keys.sort(key=lambda x: x[1])
            # Remove oldest processed entries
            for key in processed_keys[:len(processed_keys) - max_cache_size//2]:
                del self.results_cache[key]
                
    def run_ocr(self, frame, box):
        """
        Run OCR on a cropped region of a frame.
        
        Args:
            frame: Source frame
            box: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            OCR result text and elapsed time
        """
        x1, y1, x2, y2 = map(int, box)
        cropped_image = frame[y1:y2, x1:x2]
        
        if cropped_image.size > 0:
            _, buffer = cv2.imencode('.jpg', cropped_image)
            img_data = buffer.tobytes()
            result, elapse_list = self.ocr_engine(img_data, use_det=False, use_cls=False, use_rec=True)
            
            if result:
                return result[0][0], elapse_list
        
        return None, None

class VideoPipeline:
    def __init__(self, camera_configs, test_mode=False, batch_size=8):
        """
        Initialize the pipeline with camera configurations.
        
        Args:
            camera_configs: List of camera configuration dictionaries
            test_mode: Boolean flag to enable test mode with visualizations
            batch_size: Number of frames to process in a batch
        """
        self.camera_configs = camera_configs
        self.streams = {}
        self.tracked_objects = {}
        
        # Set up CUDA device if available
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Load model on the available device
        self.model = YOLO('/Workspace/sg/Mycodes/Yolo/runs/detect/train138/weights/best.pt').to(self.device)
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(self.model, self.device, batch_size=batch_size)
        
        self.predefined_point = (977,955)
        self.camera_resolutions = {}
        self.test_mode = test_mode
        self.video_writers = {}
        
        # Store resolution for each camera
        for config in camera_configs:
            camera_id = config['camera_id']
            self.camera_resolutions[camera_id] = (
                config.get('width', 1280),  # Default to 1280 if not specified
                config.get('height', 720)   # Default to 720 if not specified
            )
            
        # Create output directory for test mode
        if self.test_mode:
            self.output_dir = os.path.join('debug_output', datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Set up logging
            self.setup_logging()
            
            self.logger.info(f"Test mode enabled. Debug videos will be saved to {self.output_dir}")
            self.logger.info(f"Using device: {self.device}")
            if 'cuda' in self.device:
                self.logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                self.logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            self.logger.info(f"Batch processing enabled with batch size: {batch_size}")
        
        # Create a thread to continuously process batches
        self.batch_processing_thread = threading.Thread(target=self.process_batches_continuously)
        self.running = True  # Control flag for the processing thread
    
    def setup_logging(self):
        log_file = os.path.join(self.output_dir, 'pipeline.log')
        self.logger = logging.getLogger('VideoPipeline')
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.info("Logging initialized")
    
    def start_streams(self):
        """
        Start capturing video from all RTSP streams in parallel and start batch processing.
        """
        if self.test_mode:
            self.logger.info(f"Starting {len(self.camera_configs)} camera streams and batch processing")
        
        # Start batch processing thread
        self.batch_processing_thread.start()
        
        threads = []
        for config in self.camera_configs:
            camera_id = config['camera_id']
            rtsp_url = config['rtsp_url']
            thread = threading.Thread(target=self.capture_stream, args=(rtsp_url, camera_id))
            threads.append(thread)
            thread.start()
            
            if self.test_mode:
                self.logger.info(f"Started thread for camera {camera_id} with URL {rtsp_url}")

        for thread in threads:
            thread.join()
            
        # Signal batch processing thread to stop and wait for it to finish
        self.running = False
        self.batch_processing_thread.join()
    
    def process_batches_continuously(self):
        """
        Continuously process frame batches in a separate thread.
        """
        while self.running:
            self.batch_processor.process_batch()
            self.batch_processor.clean_cache()
            time.sleep(0.01)  # Small sleep to prevent CPU hogging
    
    def capture_stream(self, rtsp_url, camera_id):
        """
        Capture video stream from the given RTSP URL and add frames to batch processor.
        """
        if self.test_mode:
            self.logger.info(f"Attempting to connect to stream {rtsp_url} for camera {camera_id}")
            
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            error_msg = f'Error: Unable to open stream {rtsp_url} for camera {camera_id}'
            if self.test_mode:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            return None
        
        # Initialize video writer for test mode
        if self.test_mode:
            self.setup_video_writer(camera_id, cap)
            
        frame_id = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                error_msg = f'Error: Unable to read frame from stream for camera {camera_id}'
                if self.test_mode:
                    self.logger.error(error_msg)
                    self.logger.info(f"Stream ended for camera {camera_id} after {frame_id} frames")
                else:
                    print(error_msg)
                break
            
            if self.test_mode and frame_id % 100 == 0:  # Log every 100 frames
                self.logger.debug(f"Adding frame {frame_id} from camera {camera_id} to batch")
            
            # Create a copy of the frame for visualization if in test mode
            if self.test_mode:
                vis_frame = frame.copy()
            else:
                vis_frame = None
            
            # Create metadata for the frame
            metadata = {
                'camera_id': camera_id,
                'frame_id': frame_id,
                'timestamp': datetime.now(),
                'vis_frame': vis_frame
            }
            
            # Add frame to batch processor
            batch_processed = self.batch_processor.add_frame(frame, metadata)
            
            # If batch wasn't immediately processed, check if this frame's result is ready
            if not batch_processed:
                result = self.batch_processor.get_result(camera_id, frame_id)
                if result is not None:
                    self.process_detection_result(result, frame, camera_id, frame_id, vis_frame)
            
            # In test mode, save to video but do not display
            if self.test_mode and vis_frame is not None:
                self.video_writers[camera_id].write(vis_frame)
            
            frame_id += 1
            
        cap.release()
        
        # Clean up resources in test mode
        if self.test_mode:
            if camera_id in self.video_writers:
                self.video_writers[camera_id].release()
                self.logger.info(f"Released video writer for camera {camera_id}")
            self.logger.info(f"Stream processing completed for camera {camera_id} after {frame_id} frames")
    
    def setup_video_writer(self, camera_id, cap):
        """Set up video writer for a camera in test mode"""
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width > 0 and actual_height > 0:
            self.camera_resolutions[camera_id] = (actual_width, actual_height)
            self.logger.info(f"Camera {camera_id} resolution: {actual_width}x{actual_height}")
        
        width, height = self.camera_resolutions[camera_id]
        output_path = os.path.join(self.output_dir, f"{camera_id}_debug.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Adjust as needed
        self.video_writers[camera_id] = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        self.logger.info(f"Created video writer for camera {camera_id}: {output_path}")
    
    def process_detection_result(self, result, frame, camera_id, frame_id, vis_frame=None):
        """
        Process detection results from the batch processor.
        
        Args:
            result: YOLO detection result
            frame: Original frame
            camera_id: Camera identifier
            frame_id: Frame identifier
            vis_frame: Optional visualization frame
        """
        if self.test_mode and frame_id % 100 == 0:
            self.logger.debug(f"Processing result for frame {frame_id} from camera {camera_id}")
        
        roi = self.define_roi()
        
        if result.boxes is None or len(result.boxes) == 0:
            # Draw ROI in test mode even if no detections
            if self.test_mode and vis_frame is not None:
                self.draw_roi(vis_frame, roi)
                self.draw_predefined_point(vis_frame)
            return
        
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id
        
        if track_ids is None:
            # Draw ROI in test mode even if no tracking
            if self.test_mode and vis_frame is not None:
                self.draw_roi(vis_frame, roi)
                self.draw_predefined_point(vis_frame)
            return
            
        track_ids = track_ids.int().cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy()
        class_names = result.names  # Get class names from model
        
        # Draw ROI in test mode
        if self.test_mode and vis_frame is not None:
            self.draw_roi(vis_frame, roi)
            self.draw_predefined_point(vis_frame)
        
        # Continue with existing detection processing logic
        filtered_detections = []
        class0_in_roi_count = 0
        class1_in_roi_count = 0
        
        # First stage: Identify objects inside ROI
        if self.test_mode:
            self.logger.debug(f"STAGE 1: Filtering detections inside ROI (frame {frame_id}, camera {camera_id})")
        
        for i, (box, track_id, conf, cls) in enumerate(zip(boxes, track_ids, confs, clss)):
            x1, y1, x2, y2 = box
            
            # Check if detection is within ROI
            in_roi = self.is_inside_roi(box, roi)
            
            # Always process class ID 1 detections, regardless of ROI
            detection = {
                'box': (x1, y1, x2, y2),
                'conf': conf,
                'class_id': cls,
                'track_id': track_id
            }
            
            if cls == 0 and in_roi:
                filtered_detections.append(detection)
                class0_in_roi_count += 1
                if self.test_mode:
                    class_name = class_names[int(cls)] if int(cls) in class_names else f"Class {int(cls)}"
                    self.logger.debug(
                        f"ROI: Class 0 object detected - ID {int(track_id)}, {class_name}, "
                        f"conf {conf:.2f}, box [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                    )
                # Update tracking for class 0 objects
                self.update_tracking(camera_id, track_id, detection, frame_id)
            elif cls == 1:
                filtered_detections.append(detection)  # Always add class ID 1 detections
                class1_in_roi_count += 1
                if self.test_mode:
                    class_name = class_names[int(cls)] if int(cls) in class_names else f"Class {int(cls)}"
                    self.logger.debug(
                        f"Class 1 object detected - ID {int(track_id)}, {class_name}, "
                        f"conf {conf:.2f}, box [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                    )
            
            # In test mode, visualize all detections but with different colors
            if self.test_mode and vis_frame is not None:
                color = (0, 255, 0) if in_roi else (0, 0, 255)  # Green if in ROI, red if outside
                self.draw_detection(vis_frame, box, track_id, conf, cls, class_names, color)
        
        # If a new best detection is found, send it to the OCR model
        if best_detection is not None and vis_frame is not None:
            box = best_detection['box']
            # Use the batch processor's OCR engine to process this detection
            ocr_result, elapsed = self.batch_processor.run_ocr(frame, box)
            
            if ocr_result:
                # Display OCR result on visualization frame
                ocr_output_text = f"OCR: {ocr_result}"
                text_x = vis_frame.shape[1] - 10 - cv2.getTextSize(ocr_output_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
                text_y = 130
                cv2.putText(vis_frame, ocr_output_text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def define_roi(self):
        roi = np.array([[273,1072],[533,841],[1409,807],[1856,1066]], np.int32)
        if self.test_mode:
            self.logger.debug(f"Defined ROI with points: {roi.tolist()}")
        return roi

    def run_detection(self, frame, roi, frame_id, camera_id, vis_frame=None):
        if self.test_mode and frame_id % 100 == 0:
            self.logger.debug(f"--- Starting detection on frame {frame_id} from camera {camera_id} ---")
        results = self.model.track(frame, persist=True, device=self.device, stream=True)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            if self.test_mode:
                self.logger.debug(f"No detections in frame {frame_id} from camera {camera_id}")
                if vis_frame is not None:
                    self.draw_roi(vis_frame, roi)
                    self.draw_predefined_point(vis_frame)
            return
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id
        if track_ids is None:
            if self.test_mode:
                self.logger.debug(f"No tracking IDs in frame {frame_id} from camera {camera_id}")
                if vis_frame is not None:
                    self.draw_roi(vis_frame, roi)
                    self.draw_predefined_point(vis_frame)
            return
        track_ids = track_ids.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names
        if self.test_mode and vis_frame is not None:
            self.draw_roi(vis_frame, roi)
            self.draw_predefined_point(vis_frame)
        filtered_detections = []
        class0_in_roi_count = 0
        class1_in_roi_count = 0
        if self.test_mode:
            self.logger.debug(f"STAGE 1: Filtering detections inside ROI (frame {frame_id}, camera {camera_id})")
        for i, (box, track_id, conf, cls) in enumerate(zip(boxes, track_ids, confs, clss)):
            x1, y1, x2, y2 = box
            in_roi = self.is_inside_roi(box, roi)
            detection = {
                'box': (x1, y1, x2, y2),
                'conf': conf,
                'class_id': cls,
                'track_id': track_id
            }
            if cls == 0 and in_roi:
                filtered_detections.append(detection)
                class0_in_roi_count += 1
                if self.test_mode:
                    class_name = class_names[int(cls)] if int(cls) in class_names else f"Class {int(cls)}"
                    self.logger.debug(
                        f"ROI: Class 0 object detected - ID {int(track_id)}, {class_name}, "
                        f"conf {conf:.2f}, box [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                    )
                self.update_tracking(camera_id, track_id, detection, frame_id)
            elif cls == 1:
                filtered_detections.append(detection)
                class1_in_roi_count += 1
                if self.test_mode:
                    class_name = class_names[int(cls)] if int(cls) in class_names else f"Class {int(cls)}"
                    self.logger.debug(
                        f"Class 1 object detected - ID {int(track_id)}, {class_name}, "
                        f"conf {conf:.2f}, box [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                    )
            if self.test_mode and vis_frame is not None:
                color = (0, 255, 0) if in_roi else (0, 0, 255)
                self.draw_detection(vis_frame, box, track_id, conf, cls, class_names, color)
        if self.test_mode:
            self.logger.debug(f"ROI Summary: Found {class0_in_roi_count} class 0 objects and {class1_in_roi_count} class 1 objects inside ROI")
        best_detection = self.voter_logic(camera_id, filtered_detections, frame_id, vis_frame if self.test_mode else None)
        if best_detection is not None:
            box = best_detection['box']
            x1, y1, x2, y2 = box
            cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
            if cropped_image.size > 0:
                _, buffer = cv2.imencode('.jpg', cropped_image)
                img_data = buffer.tobytes()
                result, elapse_list = self.batch_processor.run_ocr(frame, box)
                result = result[0][0]
                if self.test_mode:
                    self.logger.debug(f"OCR Result: {result}, Elapsed Time: {elapse_list}")
                ocr_output_text = f"OCR: {result}"
                text_x = vis_frame.shape[1] - 10 - cv2.getTextSize(ocr_output_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
                text_y = 130
                cv2.putText(vis_frame, ocr_output_text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if self.test_mode:
                if best_detection is not None:
                    self.logger.debug(
                        f"RESULT: Best detection selected - "
                        f"Class {int(best_detection['class_id'])} ID {int(best_detection['track_id'])}, "
                        f"conf {best_detection['conf']:.2f}, box {[int(coord) for coord in best_detection['box']]}"
                    )
                else:
                    self.logger.debug("RESULT: No best detection selected")
                self.logger.debug(f"--- Completed detection processing for frame {frame_id} ---")

    def update_tracking(self, camera_id, track_id, detection, frame_id):
        key = (camera_id, int(track_id))
        if key not in self.tracked_objects:
            if self.test_mode:
                class_id = detection['class_id']
                self.logger.debug(
                    f"Tracking: New object tracked - Camera {camera_id}, "
                    f"Class {int(class_id)} ID {int(track_id)}, first seen at frame {frame_id}"
                )
            self.tracked_objects[key] = {
                'detections': [],
                'last_seen': frame_id,
                'confidence': [],
                'class_id': detection['class_id']
            }
        self.tracked_objects[key]['detections'].append((frame_id, detection))
        self.tracked_objects[key]['last_seen'] = frame_id
        self.tracked_objects[key]['confidence'].append(detection['conf'])
        if self.test_mode:
            track_length = len(self.tracked_objects[key]['detections'])
            if track_length % 10 == 0:
                avg_conf = sum(self.tracked_objects[key]['confidence']) / len(self.tracked_objects[key]['confidence'])
                class_id = detection['class_id']
                self.logger.debug(
                    f"Tracking milestone: Camera {camera_id}, Class {int(class_id)} ID {int(track_id)} - "
                    f"Tracked for {track_length} frames, avg confidence: {avg_conf:.3f}"
                )
        if len(self.tracked_objects[key]['detections']) > 30:
            self.tracked_objects[key]['detections'].pop(0)
            self.tracked_objects[key]['confidence'].pop(0)

    def voter_logic(self, camera_id, detections, frame_id, vis_frame=None):
        best_detections = {}
        class0_with_associations = [d for d in detections if d['class_id'] == 0 and 'associated_class_1' in d]
        if class0_with_associations:
            for detection in class0_with_associations:
                track_id = detection['track_id']
                if track_id not in best_detections:
                    best_detections[track_id] = {'best_score': 0, 'best_detection': None}
                associated_class_1 = detection['associated_class_1']
                score = associated_class_1['conf']
                if score > best_detections[track_id]['best_score']:
                    best_detections[track_id]['best_score'] = score
                    best_detections[track_id]['best_detection'] = associated_class_1
                    best_detections[track_id]['best_box'] = associated_class_1['box']
        best_detection_result = None
        highest_score = 0
        best_track_id = None
        for track_id, best_detection_info in best_detections.items():
            if best_detection_info['best_detection'] is not None:
                current_score = best_detection_info['best_score']
                if current_score > highest_score:
                    highest_score = current_score
                    best_track_id = track_id
                    best_detection_result = {
                        'box': best_detection_info['best_box'],
                        'conf': best_detection_info['best_detection']['conf'],
                        'class_id': best_detection_info['best_detection']['class_id'],
                        'track_id': best_detection_info['best_detection']['track_id'],
                        'parent_class0_id': track_id
                    }
        return best_detection_result

    def is_inside_roi(self, box, roi):
        x1, y1, x2, y2 = box
        bottom_left = (x1, y2)
        bottom_right = (x2, y2)
        bottom_left_inside = cv2.pointPolygonTest(roi, bottom_left, False) >= 0
        bottom_right_inside = cv2.pointPolygonTest(roi, bottom_right, False) >= 0
        return bottom_left_inside and bottom_right_inside

    def draw_roi(self, frame, roi):
        cv2.polylines(frame, [roi], True, (0, 255, 0), 2)

    def draw_predefined_point(self, frame):
        cv2.circle(frame, self.predefined_point, 5, (0, 0, 255), -1)

    def draw_detection(self, frame, box, track_id, conf, cls, class_names, color):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{class_names[int(cls)]} {track_id} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Example usage:
if __name__ == "__main__":
    # Configure environment to use GPU(s)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use all available GPUs
    
    camera_configs = [
        {
            'rtsp_url': 'rtsp://192.168.10.139:8554/stream101', 
            'camera_id': 'cam1', 
            'location': 'Entrance',
            'width': 1920,
            'height': 1080
        },        
        {
            'rtsp_url': 'rtsp://192.168.10.139:8554/stream102', 
            'camera_id': 'cam2', 
            'location': 'Lobby',
            'width': 1920,
            'height': 1080
        },
        {
            'rtsp_url': 'rtsp://192.168.10.139:8554/stream103', 
            'camera_id': 'cam3', 
            'location': 'Lobby',
            'width': 1920,
            'height': 1080
        }
    ]
    
    # Set up basic logging before pipeline initialization
    log_dir = os.path.join('debug_output', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'startup.log'),
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    root_logger = logging.getLogger()
    root_logger.info("Starting Video Pipeline application")
    
    # Log GPU information
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        root_logger.info(f"Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            root_logger.info(f"GPU {i}: {gpu_name} with {gpu_memory:.2f} GB memory")
    else:
        root_logger.warning("No CUDA devices available. Using CPU mode.")
    
    try:
        # Set test_mode=True to enable visual debugging output (saved to file only)
        pipeline = VideoPipeline(camera_configs, test_mode=True)
        pipeline.start_streams()
    except Exception as e:
        root_logger.error(f"Fatal error in application: {str(e)}", exc_info=True)
        raise