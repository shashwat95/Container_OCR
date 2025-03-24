print("Starting script execution...")
import cv2
import torch
import numpy as np
import os
import time
import logging
import random
import string
from datetime import datetime
from rapidocr_onnxruntime import RapidOCR
from ultralytics import YOLO
from database.db_operations import DatabaseManager
from config.config import BaseConfig as Config
import sys
import psutil
import queue
import threading
import builtins
import multiprocessing as mp
from typing import Dict, List
import re
from urllib.parse import urlparse

if not hasattr(np, 'bool'):
    np.bool = builtins.bool

run_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
print(f"Starting script with run ID: {run_id}")


print("Imports completed successfully")

# Configure logging
print("Setting up logging configuration...")
log_dir = Config.LOG_PATH
print(f"Log directory path: {log_dir}")

os.makedirs(log_dir, exist_ok=True)
print(f"Log directory created/verified at: {log_dir}")

log_file = log_dir / 'ml_engine.log'
print(f"Log file path: {log_file}")

print("Initializing logging configuration...")
# Create logger
logger = logging.getLogger('VideoPipeline')
logger.setLevel(Config.LOG_LEVEL)

# Create handlers
file_handler = logging.FileHandler(log_file, mode='a')
console_handler = logging.StreamHandler()

# Create formatter
formatter = logging.Formatter(Config.LOG_FORMAT)

# Set formatter for handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

print("Logging configuration completed")

print("Creating logger instance...")
print(f"Logger name: {logger.name}")
print(f"Logger level: {logger.level}")
print(f"Logger handlers: {logger.handlers}")
print(f"Logger effective level: {logger.getEffectiveLevel()}")

logger.info("Logging initialized. Log file: %s", log_file)

class MultiProcessingFrameQueue:
    """Wrapper around mp.Queue with logging capabilities"""
    def __init__(self, max_size=100):
        self.queue = mp.Queue(maxsize=max_size)
        self.max_size = max_size
        self.last_size_log_time = time.time()
        self.last_enqueued_time = time.time()

    def put(self, frame_data, timeout=1):
        try:
            self.queue.put(frame_data, timeout=timeout)
            self.last_enqueued_time = time.time()
            return True
        except queue.Full:
            return False

    def get(self, timeout=1):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self):
        # Note: qsize() is not reliable on all platforms
        try:
            return self.queue.qsize()
        except NotImplementedError:
            return -1

    def log_status(self, force=False):
        """Log queue status info periodically or when forced"""
        current_time = time.time()
        if force or current_time - self.last_size_log_time >= Config.QUEUE_LOG_INTERVAL:
            try:
                queue_size = self.qsize()
                if queue_size >= 0:  # Only log if we can get the queue size
                    queue_fullness = (queue_size / self.max_size) * 100
                    time_since_last_frame = current_time - self.last_enqueued_time
                    
                    logger.info(f"Frame queue status: {queue_size}/{self.max_size} frames ({queue_fullness:.1f}%), "
                              f"Last frame: {time_since_last_frame:.2f}s ago")
                    
                    # Evaluate real-time processing capability
                    if queue_fullness > 80:
                        logger.warning("Frame queue is nearing capacity (>80%) - processing may not be real-time")
                    elif queue_fullness < 10 and queue_size > 0 and time_since_last_frame < 5.0:
                        logger.info("Processing is keeping up with incoming frames (queue <10% full)")
            except Exception as e:
                logger.warning(f"Could not get queue status: {str(e)}")
            
            self.last_size_log_time = current_time

class StreamReader:
    """Handles reading frames from a single RTSP stream in a separate process"""
    def __init__(self, camera_config: dict, frame_queue: MultiProcessingFrameQueue):
        self.camera_config = camera_config
        self.frame_queue = frame_queue.queue  # Use the underlying mp.Queue
        self.process = None
        self.shutdown = mp.Event()
        
    def start(self):
        """Start the stream reader process"""
        self.process = mp.Process(
            target=self._read_stream,
            args=(self.camera_config, self.frame_queue, self.shutdown)
        )
        self.process.daemon = True
        self.process.start()
        
    def stop(self):
        """Stop the stream reader process"""
        if self.process:
            self.shutdown.set()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()

    @staticmethod
    def _check_gstreamer_plugins():
        """Check if GStreamer support is available in OpenCV"""
        # Check if OpenCV is built with GStreamer support
        if not cv2.getBuildInformation().find('GStreamer') != -1:
            logger.error("OpenCV is not built with GStreamer support")
            return False

        # Log OpenCV build information for debugging
        build_info = cv2.getBuildInformation()
        gst_section = [line for line in build_info.split('\n') if 'GStreamer' in line]
        logger.debug("OpenCV GStreamer build information:")
        for line in gst_section:
            logger.debug(line)

        return True

    @staticmethod
    def _read_stream(camera_config: dict, frame_queue: mp.Queue, shutdown: mp.Event):
        """Process function that reads frames from the stream"""
        logger.info(f"Starting stream reader for camera {camera_config['camera_id']}")
        cap = None
        reconnect_delay = Config.RECONNECT_DELAY_BASE
        frame_count = 0
        use_gstreamer = True  # Flag to track if we should use GStreamer
        connection_method = None  # Track which method is currently being used
        
        # Check GStreamer support at startup
        if not StreamReader._check_gstreamer_plugins():
            logger.warning("GStreamer support not available, falling back to direct RTSP")
            use_gstreamer = False
        
        while not shutdown.is_set():
            try:
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    
                    logger.info(f"Connecting to stream for camera {camera_config['camera_id']}...")
                    rtsp_url = camera_config['rtsp_url']

                    if use_gstreamer:
                        try:
                            # Parse the RTSP URL
                            parsed_url = urlparse(rtsp_url)
                            
                            # Extract authentication info if present
                            username = None
                            password = None
                            if '@' in parsed_url.netloc:
                                auth_part = parsed_url.netloc.split('@')[0]
                                if ':' in auth_part:
                                    username, password = auth_part.split(':')
                            
                            # Get host and port
                            if '@' in parsed_url.netloc:
                                host_part = parsed_url.netloc.split('@')[1]
                            else:
                                host_part = parsed_url.netloc
                                
                            if ':' in host_part:
                                host, port = host_part.split(':')
                                port = int(port)
                            else:
                                host = host_part
                                port = 554  # Default RTSP port

                            # Check if running on Jetson
                            is_jetson = os.path.exists('/proc/device-tree/model') and 'NVIDIA Jetson' in open('/proc/device-tree/model', 'r').read()

                            # Construct GStreamer pipeline based on platform and URL type
                            if is_jetson:
                                # Jetson-specific pipeline with hardware acceleration
                                if username and password:
                                    gst_pipeline = (
                                        f'rtspsrc location="{rtsp_url}" protocols=tcp latency=0 '
                                        f'tcp-timeout=0 drop-on-latency=true ntp-sync=true '
                                        f'user-id="{username}" user-pw="{password}" ! '
                                        'rtph264depay ! h264parse ! '
                                        'nvv4l2decoder mjpeg=1 ! '
                                        'nvvidconv ! video/x-raw,format=BGRx ! '
                                        'videoconvert ! video/x-raw,format=BGR ! '
                                        'appsink sync=false drop=true emit-signals=true'
                                    )
                                else:
                                    gst_pipeline = (
                                        f'rtspsrc location="{rtsp_url}" protocols=tcp latency=0 '
                                        f'tcp-timeout=0 drop-on-latency=true ntp-sync=true ! '
                                        'rtph264depay ! h264parse ! '
                                        'nvv4l2decoder mjpeg=1 ! '
                                        'nvvidconv ! video/x-raw,format=BGRx ! '
                                        'videoconvert ! video/x-raw,format=BGR ! '
                                        'appsink sync=false drop=true emit-signals=true'
                                    )
                            else:
                                # Non-Jetson pipeline with software decoding
                                if username and password:
                                    gst_pipeline = (
                                        f'rtspsrc location="{rtsp_url}" protocols=tcp latency=0 '
                                        f'tcp-timeout=0 drop-on-latency=true ntp-sync=true '
                                        f'user-id="{username}" user-pw="{password}" ! '
                                        'rtph264depay ! h264parse ! '
                                        'avdec_h264 ! videoconvert ! '
                                        'video/x-raw,format=BGR ! '
                                        'appsink sync=false drop=true emit-signals=true'
                                    )
                                else:
                                    gst_pipeline = (
                                        f'rtspsrc location="{rtsp_url}" protocols=tcp latency=0 '
                                        f'tcp-timeout=0 drop-on-latency=true ntp-sync=true ! '
                                        'rtph264depay ! h264parse ! '
                                        'avdec_h264 ! videoconvert ! '
                                        'video/x-raw,format=BGR ! '
                                        'appsink sync=false drop=true emit-signals=true'
                                    )

                            logger.info(f"Attempting GStreamer pipeline for camera {camera_config['camera_id']}")
                            # Log sanitized pipeline (remove credentials)
                            sanitized_pipeline = gst_pipeline.replace(f'user-id="{username}"', 'user-id="***"').replace(f'user-pw="{password}"', 'user-pw="***"') if username and password else gst_pipeline
                            logger.debug(f"Pipeline: {sanitized_pipeline}")
                            
                            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                            
                            if not cap.isOpened():
                                logger.error(f"GStreamer pipeline failed to open. Pipeline used: {sanitized_pipeline}")
                                # Log OpenCV build information for debugging
                                logger.debug(f"OpenCV build info: {cv2.getBuildInformation()}")
                                use_gstreamer = False
                                continue
                            else:
                                connection_method = f"GStreamer TCP ({'Jetson' if is_jetson else 'Software'} decoder)"
                                logger.info(f"Successfully connected using {connection_method} for camera {camera_config['camera_id']}")
                        except Exception as e:
                            logger.warning(f"GStreamer error for camera {camera_config['camera_id']}: {str(e)}")
                            use_gstreamer = False
                            connection_method = None
                            continue
                    else:
                        # Direct RTSP connection
                        logger.info(f"Attempting direct RTSP connection for camera {camera_config['camera_id']}")
                        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)  # Explicitly use FFMPEG backend
                        
                        if not cap.isOpened():
                            logger.warning(f"Direct RTSP connection failed for camera {camera_config['camera_id']}")
                            connection_method = None
                            time.sleep(reconnect_delay)
                            reconnect_delay = min(reconnect_delay * 1.5, Config.RECONNECT_DELAY_MAX)
                            continue
                        else:
                            connection_method = "Direct RTSP"
                            logger.info(f"Successfully connected using direct RTSP for camera {camera_config['camera_id']}")
                            # Set buffer size
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Log the final connection method being used
                    if connection_method:
                        logger.info(f"Camera {camera_config['camera_id']} is using {connection_method} connection method")
                    
                    reconnect_delay = Config.RECONNECT_DELAY_BASE
                
                ret, frame = cap.read()
                timestamp = datetime.now()
                frame_count += 1
                
                if not ret:
                    logger.warning(f"Frame capture failed for camera {camera_config['camera_id']} using {connection_method}")
                    cap.release()
                    cap = None
                    continue
                
                # Only process frames according to DROP_FRAME configuration
                if frame_count % Config.DROP_FRAME == 0:
                    # Package frame with metadata
                    frame_data = {
                        'frame': frame,
                        'timestamp': timestamp,
                        'camera_id': camera_config['camera_id'],
                        'camera_config': camera_config,
                        'connection_method': connection_method  # Include connection method in metadata
                    }
                    
                    try:
                        frame_queue.put(frame_data, timeout=1)
                    except queue.Full:
                        logger.warning(f"Frame queue full for camera {camera_config['camera_id']}, dropping frame")
                    
            except Exception as e:
                logger.error(f"Error in stream reader for camera {camera_config['camera_id']} using {connection_method}: {str(e)}")
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(1)
        
        if cap is not None:
            cap.release()
        logger.info(f"Stream reader stopped for camera {camera_config['camera_id']} (was using {connection_method})")

class VideoPipeline:
    """Modified VideoPipeline to handle multiple streams"""
    def __init__(self, camera_configs: List[dict]):
        # Initialize basic attributes first
        self.shutdown = False
        self.stream_readers = {}
        self.tracked_objects = {}
        self.models_loaded = False
        self.last_log_time = time.time()
        
        # Ensure camera_configs is a list
        if isinstance(camera_configs, dict):
            camera_configs = [camera_configs]
        elif not isinstance(camera_configs, list):
            raise ValueError("camera_configs must be a dict or list of dicts")
            
        self.camera_configs = camera_configs
        
        # Log GPU availability with Jetson support
        try:
            # Check if running on Jetson
            jetson_info = None
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    jetson_info = f.read()

            if jetson_info and 'NVIDIA Jetson' in jetson_info:
                logger.info(f"Detected Jetson device: {jetson_info.strip()}")
                gpu_name = "NVIDIA Jetson Xavier NX"
                logger.info(f"Using Jetson GPU: {gpu_name}")
                # Set CUDA device for PyTorch
                torch.cuda.set_device(0)
                logger.info("Successfully set CUDA device for Jetson")
            else:
                # Standard GPU detection for non-Jetson devices
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        logger.info(f"GPU {i}: {gpu_name}")
                    logger.info(f"Using GPU for processing")
                else:
                    logger.warning("No standard GPU found, checking for Jetson GPU...")
                    # Double check Jetson GPU availability through CUDA
                    if torch.cuda.is_available():
                        logger.info("Jetson GPU is available through CUDA")
                    else:
                        logger.warning("No GPU found, using CPU for processing")

            # Additional CUDA info
            if torch.cuda.is_available():
                logger.info(f"CUDA Version: {torch.version.cuda}")
                logger.info(f"PyTorch CUDA available: Yes")
                logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
                logger.info(f"CUDA device capability: {torch.cuda.get_device_capability()}")
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"Current device properties: {torch.cuda.get_device_properties(torch.cuda.current_device())}")
        except Exception as e:
            logger.error(f"Error during GPU detection: {str(e)}")
            logger.warning("Falling back to CPU processing")

        # Initialize database manager
        self.db_manager = DatabaseManager()
        
        # Create a single shared frame queue for all streams
        self.frame_queue = MultiProcessingFrameQueue(max_size=Config.FRAME_QUEUE_SIZE)
        
        # Create stream readers for each camera
        logger.info(f"Creating stream readers for {len(self.camera_configs)} cameras")
        for config in self.camera_configs:
            logger.info(f"Creating stream reader for camera {config['camera_id']}")
            reader = StreamReader(config, self.frame_queue)
            self.stream_readers[config['camera_id']] = reader
        
        logger.info(f"Initialized {len(self.stream_readers)} stream readers")
        
        # Generate a unique session identifier for track IDs
        self.session_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        logger.info(f"Initialized pipeline with session ID: {self.session_id}")
        
        # Create evidence images directory
        os.makedirs(Config.EVIDENCE_IMAGES_DIR, exist_ok=True)

        self._setup_state()
        self._validate_configs()

    def _setup_state(self):
        """Initialize resources"""
        self.tracked_objects = {}
        self.last_log_time = time.time()

    def _validate_configs(self):
        """Validate all camera configurations"""
        for config in self.camera_configs:
            required_keys = ['rtsp_url', 'camera_id', 'location']
            if not all(k in config for k in required_keys):
                raise ValueError(f"Invalid camera config: {config}")
            if 'roi_points' not in config:
                config['roi_points'] = np.array(Config.DEFAULT_CAMERA_CONFIG['roi_points'], np.int32)

    def start(self):
        """Start the pipeline with multiple stream readers"""
        try:
            logger.info("Starting pipeline with multiple streams")
            
            # Initialize models
            device = self._get_device()
            logger.info(f"Loading YOLO model on {device}")
            model = YOLO(Config.MODEL_PATH)
            logger.info("Successfully loaded YOLO model")
            
            logger.info("Initializing OCR engine")
            ocr_engine = RapidOCR()
            logger.info("Successfully initialized OCR engine")
            
            # Signal that models are loaded
            self.models_loaded = True
            
            # Start all stream readers
            for camera_id, reader in self.stream_readers.items():
                logger.info(f"Starting stream reader for camera {camera_id}")
                reader.start()
            
            frame_count = 0
            processed_track_ids = set()

            # Main processing loop
            while not self.shutdown:
                try:
                    # Get frame data from queue
                    try:
                        frame_data = self.frame_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    
                    if frame_data is None:
                        continue

                    frame = frame_data['frame']
                    timestamp = frame_data['timestamp']
                    camera_id = frame_data['camera_id']
                    camera_config = frame_data['camera_config']
                    connection_method = frame_data['connection_method']
                    
                    frame_count += 1
                    
                    # Process frame at specified interval
                    if frame_count % Config.DROP_FRAME == 0:
                        # Monitor system resources periodically
                        self._log_status()
                        
                        # Process the frame
                        processed = self._process_frame(frame, model, camera_config['roi_points'], 
                                                     camera_id, timestamp)
                        
                        # Update tracking state
                        self._update_tracking_state(camera_id, processed)
                        
                        # Process best detection
                        if processed.get('best_detection'):
                            best_detection = processed['best_detection']
                            
                            # Apply OCR if needed
                            if 'ocr_text' not in best_detection:
                                self._process_ocr(frame, best_detection, ocr_engine)
                            
                            # Check if we've already processed this track_id
                            track_id = best_detection['track_id']
                            unique_track_id = self._get_unique_track_id(track_id)
                            
                            if unique_track_id not in processed_track_ids:
                                evidence_filename = os.path.join(
                                    Config.EVIDENCE_IMAGES_DIR, 
                                    f"{camera_id}_evidence_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
                                )
                                
                                # Create database record
                                record = {
                                    'datetime': timestamp,
                                    'ocr_output': best_detection.get('ocr_text', ''),
                                    'camera_id': camera_id,
                                    'image_path': evidence_filename,
                                    'track_id': unique_track_id,
                                    'confidence': best_detection.get('confidence', 0.0),
                                    'connection_method': connection_method
                                }
                                
                                if Config.PUSH_DATA:
                                    # Check if a record for this track_id already exists
                                    existing_record = self.db_manager.get_record_by_track_id(unique_track_id)
                                    if existing_record is None:
                                        # Insert new record
                                        cv2.imwrite(evidence_filename, frame)
                                        self.db_manager.insert_record(record)
                                    else:
                                        # Check if the current detection has better confidence
                                        current_confidence = best_detection.get('confidence', 0.0)
                                        existing_confidence = existing_record.get('confidence', 0.0)
                                        
                                        if current_confidence > existing_confidence:
                                            os.remove(existing_record.get('image_path'))
                                            # Update existing record with better confidence detection
                                            self.db_manager.update_record(existing_record['id'], record)
                                            cv2.imwrite(evidence_filename, frame)
                                
                                # Mark this track_id as processed
                                processed_track_ids.add(unique_track_id)
                            
                            # If the OCR text has changed for an existing track_id, update the record
                            elif Config.PUSH_DATA:
                                existing_record = self.db_manager.get_record_by_track_id(unique_track_id)
                                if existing_record:
                                    current_confidence = best_detection.get('confidence', 0.0)
                                    existing_confidence = existing_record.get('confidence', 0.0)
                                    
                                    if current_confidence > existing_confidence:
                                        evidence_filename = os.path.join(
                                            Config.EVIDENCE_IMAGES_DIR,
                                            f"{camera_id}_evidence_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
                                        )
                                        record = {
                                            'datetime': timestamp,
                                            'ocr_output': best_detection.get('ocr_text', ''),
                                            'camera_id': camera_id,
                                            'image_path': evidence_filename,
                                            'track_id': unique_track_id,
                                            'confidence': current_confidence,
                                            'connection_method': connection_method
                                        }
                                        self.db_manager.update_record(existing_record['id'], record)
                                        cv2.imwrite(evidence_filename, frame)

                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            self.shutdown = True
            # Stop all stream readers
            if hasattr(self, 'stream_readers'):
                for reader in self.stream_readers.values():
                    try:
                        reader.stop()
                    except Exception as e:
                        logger.error(f"Error stopping stream reader: {str(e)}")
            
            # Clean up resources
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
            logger.info("Pipeline shutdown completed")

    def _process_frame(self, frame: np.ndarray, model: YOLO, roi: np.ndarray, 
                    camera_id: str, timestamp: datetime) -> dict:
        """Process a single frame through the pipeline"""
        processed = {'detections': [], 'best_detection': None, 'timestamp': timestamp}
        
        if frame is None:
            return processed

        # Run object detection
        with torch.no_grad():
            results = model.track(frame, persist=True, device=model.device, verbose=False)

        if not results or not results[0].boxes:
            return processed

        # Convert tensors to numpy safely
        try:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
        except Exception as e:
            return processed

        # Process detections
        class0_detections = []
        class1_detections = []
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
            track_id_val = track_ids[i] if i < len(track_ids) else -1
            
            detection = {
                'box': box.tolist(),
                'confidence': float(conf),
                'class_id': int(cls),
                'track_id': track_id_val,
                'in_roi': self._is_in_roi(box, roi)
            }
            
            if cls == 0 and detection['in_roi']:
                class0_detections.append(detection)
            elif cls == 1:
                class1_detections.append(detection)
        
        # Process associations between class0 and class1
        for i, class0 in enumerate(class0_detections):
            class0_box = class0['box']
            potential_associations = []
            
            # Find class1 detections completely inside class0's box
            for j, class1 in enumerate(class1_detections):
                if self._is_completely_inside(class0_box, class1['box']):
                    potential_associations.append(class1)
            
            # Select best association
            if potential_associations:
                best_class1 = max(potential_associations, key=lambda x: x['confidence'])
                class0['associated_class1'] = best_class1
            else:
                class0['associated_class1'] = None

        processed['detections'] = class0_detections + class1_detections
        
        # Find best detection using improved voting mechanism
        best_detection = self._get_best_detection(camera_id, class0_detections, timestamp)
        processed['best_detection'] = best_detection
        
        return processed

    def _get_best_detection(self, camera_id: str, class0_detections: list, timestamp: datetime) -> dict:
        """Get the best detection based on confidence scores across time"""
        best_detection = None
        max_confidence = 0.0
        current_time = timestamp
        
        # Check current frame's detections
        for i, class0 in enumerate(class0_detections):
            # Only consider detections that have an associated class1 object
            if 'associated_class1' in class0 and class0['associated_class1']:
                # Get the track_id for this class0 detection
                track_id = class0['track_id']
                key = (camera_id, track_id)
                
                # Check if this object is already being tracked
                if key in self.tracked_objects:
                    # Get the history of associated class1 objects for this class0 track
                    track_obj = self.tracked_objects[key]
                    
                    # Include the current detection in our consideration
                    current_class1 = class0['associated_class1']
                    current_confidence = current_class1['confidence']
                    
                    # Compare with the best historical detection for this track
                    if 'best_class1' in track_obj:
                        best_historical = track_obj['best_class1']
                        hist_conf = best_historical['confidence']
                        
                        if current_confidence > hist_conf:
                            # Current detection is better, consider this our candidate
                            if current_confidence > max_confidence:
                                max_confidence = current_confidence
                                best_detection = current_class1.copy()
                                best_detection['track_id'] = track_id
                                best_detection['confidence'] = current_confidence  # Ensure confidence is included
                        else:
                            # Historical detection is better, consider this our candidate
                            if hist_conf > max_confidence:
                                max_confidence = hist_conf
                                best_detection = best_historical.copy()
                                best_detection['track_id'] = track_id
                                best_detection['confidence'] = hist_conf  # Ensure confidence is included
                    else:
                        # No historical best yet, use current as candidate
                        if current_confidence > max_confidence:
                            max_confidence = current_confidence
                            best_detection = current_class1.copy()
                            best_detection['track_id'] = track_id
                            best_detection['confidence'] = current_confidence  # Ensure confidence is included
                else:
                    # New track, use current detection as candidate
                    current_class1 = class0['associated_class1']
                    current_confidence = current_class1['confidence']
                    if current_confidence > max_confidence:
                        max_confidence = current_confidence
                        best_detection = current_class1.copy()
                        best_detection['track_id'] = track_id
                        best_detection['confidence'] = current_confidence  # Ensure confidence is included

        # Check all tracked objects to find any potential better detections
        # that might not be in the current frame
        for key, track_obj in self.tracked_objects.items():
            # Only consider objects from this camera
            if key[0] != camera_id:
                continue
                
            track_id = key[1]
            
            # Check if the object is still being actively tracked (within time window)
            time_since_last_seen = (current_time - track_obj['last_seen']).total_seconds()
            
            if time_since_last_seen < Config.TRACKING_WINDOW:
                # Check if there's a good class1 detection associated with this track
                if 'best_class1' in track_obj:
                    best_historical = track_obj['best_class1']
                    hist_conf = best_historical['confidence']
                    
                    if hist_conf > max_confidence:
                        max_confidence = hist_conf
                        best_detection = best_historical.copy()
                        best_detection['track_id'] = key[1]  # Use the track_id from the key
                        best_detection['confidence'] = hist_conf  # Ensure confidence is included
        
        # Store this best_detection for next comparison
        if not hasattr(self, '_prev_best_detection'):
            self._prev_best_detection = {}
        self._prev_best_detection = best_detection.copy() if best_detection else None
        
        return best_detection

    def _update_tracking_state(self, camera_id: str, processed: dict):
        """Update tracking state"""
        current_time = processed['timestamp']
        
        # Process class0 detections and their associations
        for detection in processed['detections']:
            if detection['class_id'] == 0 and detection['in_roi']:
                track_id = detection['track_id']
                key = (camera_id, track_id)
                
                # Create new entry for this tracked object if it doesn't exist
                if key not in self.tracked_objects:
                    self.tracked_objects[key] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'confidence_history': [detection['confidence']],
                        'positions': [detection['box']],
                        'class1_history': []
                    }
                else:
                    # Update existing tracking info
                    self.tracked_objects[key]['last_seen'] = current_time
                    self.tracked_objects[key]['confidence_history'].append(detection['confidence'])
                    self.tracked_objects[key]['positions'].append(detection['box'])
                
                # Update class1 association if present
                if 'associated_class1' in detection and detection['associated_class1']:
                    class1_obj = detection['associated_class1']
                    current_conf = class1_obj['confidence']
                    
                    # Add to history with timestamp
                    class1_entry = {
                        'confidence': current_conf,
                        'box': class1_obj['box'],
                        'timestamp': current_time
                    }
                    self.tracked_objects[key]['class1_history'].append(class1_entry)
                    
                    # Update the best class1 detection for this track
                    if 'best_class1' not in self.tracked_objects[key]:
                        self.tracked_objects[key]['best_class1'] = class1_obj.copy()
                        self.tracked_objects[key]['best_class1_time'] = current_time
                    elif current_conf > self.tracked_objects[key]['best_class1']['confidence']:
                        self.tracked_objects[key]['best_class1'] = class1_obj.copy()
                        self.tracked_objects[key]['best_class1_time'] = current_time
        
        # Clean up old tracked objects
        old_count = len(self.tracked_objects)
        self._cleanup_tracking_state(current_time)
        new_count = len(self.tracked_objects)

    def _cleanup_tracking_state(self, current_time: datetime):
        """Remove tracked objects that haven't been seen recently"""
        keys_to_remove = []
        for key, obj in self.tracked_objects.items():
            time_since_last_seen = (current_time - obj['last_seen']).total_seconds()
            # If not seen for more than twice the tracking window, remove
            if time_since_last_seen > Config.TRACKING_WINDOW * 2:
                keys_to_remove.append(key)
        
        # Remove the keys
        for key in keys_to_remove:
            del self.tracked_objects[key]

    def _is_in_roi(self, box: list, roi: np.ndarray) -> bool:
        """Check if detection is within region of interest"""
        x_center = (box[0] + box[2]) / 2
        y_bottom = box[3]
        return cv2.pointPolygonTest(roi, (x_center, y_bottom), False) >= 0

    def _is_completely_inside(self, outer_box: list, inner_box: list) -> bool:
        """Check if inner bounding box is completely contained within outer box"""
        outer_x1, outer_y1, outer_x2, outer_y2 = outer_box
        inner_x1, inner_y1, inner_x2, inner_y2 = inner_box
        
        return (inner_x1 >= outer_x1 and 
                inner_y1 >= outer_y1 and 
                inner_x2 <= outer_x2 and 
                inner_y2 <= outer_y2)

    def _get_unique_track_id(self, track_id: int) -> str:
        """Generate a unique track ID by combining the numerical ID with the session identifier"""
        return f"{track_id}_{self.session_id}"

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
                # Filter to keep only numerical characters
                ocr_text = result[0][0]
                filtered_text = ''.join(char for char in ocr_text if char.isdigit())
                detection['ocr_text'] = filtered_text
            else:
                detection['ocr_text'] = ''
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            detection['ocr_text'] = ''

    def stop(self):
        """Stop the pipeline and all stream readers"""
        self.shutdown = True
        if hasattr(self, 'stream_readers'):
            for reader in self.stream_readers.values():
                try:
                    reader.stop()
                except Exception as e:
                    logger.error(f"Error stopping stream reader: {str(e)}")
        torch.cuda.empty_cache()

    def _get_device(self):
        """Get available compute device with Jetson support"""
        try:
            # Check if running on Jetson
            if os.path.exists('/proc/device-tree/model'):
                with open('/proc/device-tree/model', 'r') as f:
                    if 'NVIDIA Jetson' in f.read():
                        return 'cuda:0'  # Jetson should always use CUDA

            # Standard device selection for non-Jetson
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception as e:
            logger.error(f"Error determining device: {str(e)}")
            return 'cpu'

    def _log_status(self):
        """Log status information periodically"""
        current_time = time.time()
        if current_time - self.last_log_time >= Config.LOGGING_INTERVAL:
            process = psutil.Process()
            logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f}MB")
            self.frame_queue.log_status(force=True)
            self.last_log_time = current_time

# Add to Config class if needed
if not hasattr(Config, 'FRAME_QUEUE_SIZE'):
    Config.FRAME_QUEUE_SIZE = 500  # Default max queue size
if not hasattr(Config, 'QUEUE_LOG_INTERVAL'):
    Config.QUEUE_LOG_INTERVAL = 5  # Log queue status every 60 seconds
if not hasattr(Config, 'MAX_RECONNECT_DELAY'):
    Config.MAX_RECONNECT_DELAY = 60  # Maximum reconnect delay in seconds

if __name__ == "__main__":
    print("\n=== Starting main execution ===")
    pipeline = None
    try:
        print("Reading camera configurations from Config...")
        # Get camera configurations from Config
        camera_configs = []
        
        if not hasattr(Config, 'DEFAULT_CAMERA_CONFIG'):
            raise ValueError("DEFAULT_CAMERA_CONFIG not found in Config")
            
        # Convert to list if it's a single config dict, or use as is if it's already a list
        if isinstance(Config.DEFAULT_CAMERA_CONFIG, dict):
            camera_configs = [Config.DEFAULT_CAMERA_CONFIG]
        else:
            camera_configs = Config.DEFAULT_CAMERA_CONFIG
            
        print(f"Found {len(camera_configs)} camera configuration(s)")
        for config in camera_configs:
            print(f"- Camera ID: {config.get('camera_id', 'Unknown')}")
        
        print("Initializing VideoPipeline...")
        pipeline = VideoPipeline(camera_configs)
        
        print("Starting pipeline...")
        pipeline.start()
        
    except KeyboardInterrupt:
        print("Manual interruption received. Shutting down...")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up resources...")
        if pipeline:
            pipeline.stop()
        print("Pipeline shutdown complete.")
