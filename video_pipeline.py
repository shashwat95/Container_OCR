print("Starting script execution...")
import cv2
import torch
import numpy as np
import os
import time
import logging
import random
import string
import json
from datetime import datetime
from rapidocr_onnxruntime import RapidOCR
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
import supervision as sv
from supervision.detection.core import Detections
from ultralytics import YOLO  # Keep this for detection

# Filter OpenCV frame error messages
class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        
    def write(self, message):
        # Filter out specific error messages
        if not any(pattern in message for pattern in [
            "left block unavailable", 
            "error while decoding MB",
            "requested intra",
            "h264"
        ]):
            self.original_stderr.write(message)
            
    def flush(self):
        self.original_stderr.flush()

# Redirect stderr through our filter
sys.stderr = StderrFilter(sys.stderr)

# Add test mode flag to configuration
if not hasattr(Config, 'TEST_MODE'):
    Config.TEST_MODE = False  # Set to True to enable debugging visualizations

# Add ROI timeout parameter (keeping for compatibility but not using it)
if not hasattr(Config, 'ROI_TIMEOUT'):
    Config.ROI_TIMEOUT = 10  # This will only be used for logging purposes

# Add camera-specific tracking windows
if not hasattr(Config, 'CAMERA_TRACKING_WINDOWS'):
    Config.CAMERA_TRACKING_WINDOWS = {
        'CAM_1': 60,     # 1 minute for camera 1
        'CAM_2': 60,     # 1 minute for camera 2
        'CAM_3': 7200,   # 2 hours for camera 3
        'default': 300   # 5 minutes default
    }

# Add debug directory
if not hasattr(Config, 'DEBUG_DIR'):
    Config.DEBUG_DIR = Config.LOG_PATH / 'debug_frames'
    os.makedirs(Config.DEBUG_DIR, exist_ok=True)

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
    """Handles reading frames from a single RTSP stream in a separate process using FFMPEG with TCP"""
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
    def _read_stream(camera_config: dict, frame_queue: mp.Queue, shutdown: mp.Event):
        """Process function that reads frames from the stream using FFMPEG with TCP protocol"""
        logger.info(f"Starting stream reader for camera {camera_config['camera_id']}")
        cap = None
        reconnect_delay = Config.RECONNECT_DELAY_BASE
        frame_count = 0
        connection_method = "FFMPEG TCP"  # Track which method is being used
        error_count = 0
        last_error_log_time = time.time()
        
        while not shutdown.is_set():
            try:
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    
                    logger.info(f"Connecting to stream for camera {camera_config['camera_id']}...")
                    rtsp_url = camera_config['rtsp_url']
                    
                    # Add FFMPEG TCP options to RTSP URL
                    rtsp_options = "rtsp_transport=tcp&timeout=5000000&buffer_size=1024000"
                    rtsp_with_options = f"{rtsp_url}?{rtsp_options}"
                    
                    # Log sanitized URL (hide password)
                    sanitized_url = re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', rtsp_with_options)
                    logger.info(f"Attempting to open stream with FFMPEG TCP: {sanitized_url}")
                    
                    # Start time for connection
                    start_time = time.time()
                    cap = cv2.VideoCapture(rtsp_with_options, cv2.CAP_FFMPEG)
                    connection_time = time.time() - start_time
                    
                    if not cap.isOpened():
                        logger.warning(f"Failed to open stream for camera {camera_config['camera_id']}! " 
                                     f"Connection attempt took {connection_time:.2f} seconds")
                        cap = None
                        time.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 1.5, Config.RECONNECT_DELAY_MAX)
                        continue
                    else:
                        logger.info(f"Successfully connected to camera {camera_config['camera_id']} using FFMPEG TCP! "
                                  f"Connection took {connection_time:.2f} seconds")
                        
                        # Get and log capture properties
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # Set buffer size to 1 to minimize latency
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        buffer_size = cap.get(cv2.CAP_PROP_BUFFERSIZE)
                        
                        logger.info(f"Stream properties for camera {camera_config['camera_id']}: "
                                  f"Resolution: {width}x{height}, FPS: {fps}, Buffer size: {buffer_size}")
                        
                        reconnect_delay = Config.RECONNECT_DELAY_BASE
                        error_count = 0
                
                ret, frame = cap.read()
                timestamp = datetime.now()
                frame_count += 1
                
                if not ret:
                    error_count += 1
                    current_time = time.time()
                    
                    # Only log errors periodically to reduce spam
                    if error_count <= 5 or current_time - last_error_log_time > 60:
                        logger.warning(f"Frame capture failed for camera {camera_config['camera_id']} (error count: {error_count})")
                        last_error_log_time = current_time
                        
                    if error_count > 30:  # After 30 consecutive errors, reconnect
                        logger.warning(f"Too many consecutive errors ({error_count}), reconnecting to camera {camera_config['camera_id']}")
                        cap.release()
                        cap = None
                    
                    time.sleep(0.1)  # Short delay to avoid tight loop
                    continue
                
                # Reset error count on successful frame
                error_count = 0
                
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
                error_count += 1
                current_time = time.time()
                
                # Only log errors periodically to reduce spam
                if error_count <= 5 or current_time - last_error_log_time > 60:
                    logger.error(f"Error in stream reader for camera {camera_config['camera_id']}: {str(e)}")
                    last_error_log_time = current_time
                
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(1)
        
        if cap is not None:
            cap.release()
        logger.info(f"Stream reader stopped for camera {camera_config['camera_id']}")

class VideoPipeline:
    """Modified VideoPipeline to handle multiple streams with improved ROI tracking"""
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

        # Initialize trackers for each camera
        self.trackers = {}
        for config in camera_configs:
            camera_id = config['camera_id']
            self.trackers[camera_id] = sv.ByteTrack()

        # Create Debug directory if in test mode
        if Config.TEST_MODE:
            os.makedirs(Config.DEBUG_DIR, exist_ok=True)
            logger.info(f"Test mode enabled - debug images will be saved to {Config.DEBUG_DIR}")

        self._setup_state()
        self._validate_configs()

    def _get_camera_tracking_window(self, camera_id: str) -> int:
        """Get the camera-specific tracking window in seconds"""
        if camera_id in Config.CAMERA_TRACKING_WINDOWS:
            return Config.CAMERA_TRACKING_WINDOWS[camera_id]
        return Config.CAMERA_TRACKING_WINDOWS.get('default', 300)  # Default 5 minutes

    def _draw_roi(self, frame, roi_points, color=(0, 255, 0), thickness=2):
        """Draw ROI polygon on the frame for debugging"""
        if frame is None or roi_points is None:
            return frame
        
        # Convert to appropriate format and draw
        frame_with_roi = frame.copy()
        roi_points_array = np.array(roi_points, dtype=np.int32)
        cv2.polylines(frame_with_roi, [roi_points_array], True, color, thickness)
        
        return frame_with_roi

    def _create_debug_visualization(self, frame, roi_points, detections, camera_id, timestamp, 
                           tracked_objects=None, best_detection=None, action=None, 
                           unique_track_id=None, existing_record=None):
        """Create a visualization frame with relevant debugging information"""
        if not Config.TEST_MODE or frame is None:
            return None
        
        debug_frame = frame.copy()
        
        # Draw ROI on the frame
        debug_frame = self._draw_roi(debug_frame, roi_points)
        
        # Draw all detections
        for det in detections:
            # Container detections (class_id 0)
            if det['class_id'] == 0:
                track_id = det['track_id']
                key = (camera_id, track_id)
                
                # Determine ROI status and color
                if det['in_roi']:
                    color = (0, 255, 0)  # Green for in ROI
                    roi_status = "IN_ROI"
                elif key in tracked_objects:
                    # Check if this container was seen outside ROI after being in ROI
                    track_obj = tracked_objects[key]
                    if track_obj.get('exited_roi', False):
                        color = (0, 0, 255)  # Red for containers that left ROI
                        roi_status = "EXITED ROI"
                    else:
                        color = (255, 0, 0)  # Blue for never in ROI
                        roi_status = "NEVER IN ROI"
                else:
                    color = (255, 0, 0)  # Blue for not tracked
                    roi_status = "OUT"
                
                box = [int(x) for x in det['box']]
                cv2.rectangle(debug_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                # Add track ID and ROI status
                label = f"Container #{track_id} {roi_status}"
                cv2.putText(debug_frame, label, (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw ROI test points (bottom corners)
                x_left = box[0]
                x_right = box[2]
                y_bottom = box[3]
                
                # Draw the bottom corners that are checked for ROI
                cv2.circle(debug_frame, (int(x_left), int(y_bottom)), 4, (255, 0, 255), -1)  # Magenta for left corner
                cv2.circle(debug_frame, (int(x_right), int(y_bottom)), 4, (255, 0, 255), -1)  # Magenta for right corner
            
            # Text detections (class_id 1)
            elif det['class_id'] == 1:
                # Use different color for text regions
                box = [int(x) for x in det['box']]
                cv2.rectangle(debug_frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                
                # Add confidence score
                conf = det.get('confidence', 0)
                label = f"Text {conf:.2f}"
                cv2.putText(debug_frame, label, (box[0], box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Highlight associated detections
        for det in detections:
            if det['class_id'] == 0 and 'associated_class1' in det and det['associated_class1']:
                # Draw a line between container and its text
                box_container = [int(x) for x in det['box']]
                box_text = [int(x) for x in det['associated_class1']['box']]
                
                # Calculate center points
                center_container = (
                    (box_container[0] + box_container[2]) // 2,
                    (box_container[1] + box_container[3]) // 2
                )
                center_text = (
                    (box_text[0] + box_text[2]) // 2,
                    (box_text[1] + box_text[3]) // 2
                )
                
                # Draw connecting line
                cv2.line(debug_frame, center_container, center_text, (255, 255, 0), 2)
                
                # Display text confidence score near the line
                text_conf = det['associated_class1'].get('confidence', 0)
                mid_point = (
                    (center_container[0] + center_text[0]) // 2,
                    (center_container[1] + center_text[1]) // 2
                )
                cv2.putText(debug_frame, f"{text_conf:.2f}", mid_point, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Highlight best detection if available
        if best_detection:
            box = [int(x) for x in best_detection['box']]
            cv2.rectangle(debug_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 3)
            
            # Add a 'Best' label with confidence
            det_conf = best_detection.get('confidence', 0)
            ocr_conf = best_detection.get('ocr_confidence', 0)
            
            # Include both detection and OCR confidence in label
            cv2.putText(debug_frame, f"BEST Det:{det_conf:.2f} OCR:{ocr_conf:.2f}", 
                        (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add OCR text if available
            if 'ocr_text' in best_detection and best_detection['ocr_text']:
                ocr_text = best_detection['ocr_text']
                cv2.putText(debug_frame, f"OCR: {ocr_text}", (box[0], box[3]+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add timestamp and camera info
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        tracking_window = self._get_camera_tracking_window(camera_id)
        cv2.putText(debug_frame, f"Cam: {camera_id} | Time: {time_str} | Tracking window: {tracking_window}s", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add database operation information
        if action:
            y_pos = 60
            action_color = (0, 255, 255)  # Yellow for DB actions
            
            # Add the action type (INSERT or UPDATE)
            cv2.putText(debug_frame, f"DB ACTION: {action}", 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
            y_pos += 30
            
            # Add track ID information
            if unique_track_id:
                cv2.putText(debug_frame, f"Track ID: {unique_track_id}", 
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
                y_pos += 30
            
            # Add existing record info for updates
            if action == "UPDATE" and existing_record:
                old_conf = existing_record.get('confidence', 0.0)
                new_conf = best_detection.get('ocr_confidence', 0.0)
                cv2.putText(debug_frame, f"Confidence: {old_conf:.2f} → {new_conf:.2f}", 
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
                y_pos += 30
                
                old_ocr = existing_record.get('ocr_output', '')
                new_ocr = best_detection.get('ocr_text', '')
                cv2.putText(debug_frame, f"OCR: {old_ocr} → {new_ocr}", 
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
        
        return debug_frame

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
            
            # Log ROI points for debugging
            logger.info(f"Camera {config['camera_id']} ROI: {config['roi_points']}")
            
            # Log camera-specific tracking window
            tracking_window = self._get_camera_tracking_window(config['camera_id'])
            logger.info(f"Camera {config['camera_id']} tracking window: {tracking_window} seconds")

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
                        
                        # Process the frame - detect containers and text regions
                        processed = self._process_frame(frame, model, camera_config['roi_points'], 
                                                     camera_id, timestamp)
                        
                        # Update tracking state
                        self._update_tracking_state(camera_id, processed)
                        
                        # Process each container with associated text
                        for container_detection in processed['detections']:
                            if container_detection['class_id'] == 0 and 'associated_class1' in container_detection:
                                track_id = container_detection['track_id']
                                key = (camera_id, track_id)
                                
                                # Check if container is currently in ROI
                                is_currently_in_roi = container_detection['in_roi']
                                
                                # Only process containers currently in ROI
                                # And containers that haven't been seen outside ROI after being in ROI
                                if key in self.tracked_objects:
                                    track_obj = self.tracked_objects[key]
                                    
                                    # Skip if the container has exited ROI after being in it
                                    if track_obj.get('exited_roi', False):
                                        if Config.TEST_MODE and random.random() < 0.01:  # Reduce log spam
                                            logger.debug(f"Skipping container #{track_id} - previously exited ROI")
                                        continue
                                
                                # Skip if not currently in ROI
                                if not is_currently_in_roi:
                                    continue
                                
                                # Get the associated text region
                                associated_text = container_detection['associated_class1']
                                if associated_text is None:
                                    continue
                                    
                                # Process OCR on this text region
                                self._process_ocr(frame, associated_text, ocr_engine)
                                
                                # Only continue with valid OCR results
                                if not associated_text.get('ocr_text') or associated_text.get('ocr_confidence', 0) <= 0:
                                    continue
                                
                                # Track the best OCR result for this container
                                unique_track_id = self._get_unique_track_id(track_id, camera_id)
                                current_ocr_text = associated_text.get('ocr_text', '')
                                current_ocr_confidence = associated_text.get('ocr_confidence', 0.0)
                                
                                # Get previous best OCR confidence for this container
                                previous_best_confidence = 0.0
                                previous_best_text = ''
                                
                                if key in self.tracked_objects and 'best_ocr' in self.tracked_objects[key]:
                                    previous_best_confidence = self.tracked_objects[key]['best_ocr'].get('confidence', 0.0)
                                    previous_best_text = self.tracked_objects[key]['best_ocr'].get('text', '')
                                
                                # Create copy of associated_text for best_detection
                                best_detection = associated_text.copy()
                                
                                # Update best OCR if current is better than previous
                                is_better_ocr = current_ocr_confidence > previous_best_confidence
                                
                                if is_better_ocr:
                                    # Update memory tracking
                                    if key not in self.tracked_objects:
                                        self.tracked_objects[key] = {
                                            'first_seen': timestamp,
                                            'last_seen': timestamp,
                                            'in_roi': is_currently_in_roi,
                                            'exited_roi': False,  # New flag to track if container has left ROI
                                            'first_in_roi': timestamp,  # When container first entered ROI
                                            'confidence_history': [container_detection['confidence']],
                                            'positions': [container_detection['box']],
                                            'best_ocr': {
                                                'text': current_ocr_text,
                                                'confidence': current_ocr_confidence,
                                                'timestamp': timestamp
                                            }
                                        }
                                        logger.info(f"New container #{track_id} with first OCR result - " +
                                                   f"Text: '{current_ocr_text}', Confidence: {current_ocr_confidence:.3f}")
                                    else:
                                        self.tracked_objects[key]['best_ocr'] = {
                                            'text': current_ocr_text,
                                            'confidence': current_ocr_confidence,
                                            'timestamp': timestamp
                                        }
                                        logger.info(f"Improved OCR for container #{track_id} - " +
                                                   f"Previous: '{previous_best_text}' ({previous_best_confidence:.3f}), " +
                                                   f"New: '{current_ocr_text}' ({current_ocr_confidence:.3f}), " +
                                                   f"Improvement: +{(current_ocr_confidence - previous_best_confidence):.3f}")
                                    
                                    # Now we have a valid best detection to process for database
                                    evidence_filename = os.path.join(
                                        Config.EVIDENCE_IMAGES_DIR, 
                                        f"{camera_id}_evidence_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
                                    )
                                    
                                    # Create database record
                                    record = {
                                        'datetime': timestamp,
                                        'ocr_output': current_ocr_text,
                                        'confidence': current_ocr_confidence,  # Use 'confidence' for the DB field
                                        'camera_id': camera_id,
                                        'image_path': evidence_filename,
                                        'track_id': unique_track_id,
                                        'connection_method': connection_method,
                                        'in_roi': is_currently_in_roi
                                    }
                                    
                                    if Config.PUSH_DATA:
                                        # Check if a record for this track_id already exists
                                        existing_record = self.db_manager.get_record_by_track_id(unique_track_id)
                                        if existing_record is None:
                                            # Insert new record
                                            logger.info(f"Creating new DB record for container {unique_track_id} - " +
                                                      f"OCR: '{current_ocr_text}', Confidence: {current_ocr_confidence:.3f}")
                                            cv2.imwrite(evidence_filename, frame)
                                            self.db_manager.insert_record(record)
                                            
                                            # Save debug visualization for INSERT operation
                                            if Config.TEST_MODE:
                                                debug_frame = self._create_debug_visualization(
                                                    frame=frame,
                                                    roi_points=camera_config['roi_points'],
                                                    detections=processed['detections'],
                                                    camera_id=camera_id,
                                                    timestamp=timestamp,
                                                    tracked_objects=self.tracked_objects,
                                                    best_detection=best_detection,
                                                    action="INSERT",
                                                    unique_track_id=unique_track_id
                                                )
                                                
                                                if debug_frame is not None:
                                                    debug_path = os.path.join(
                                                        Config.DEBUG_DIR, 
                                                        f"DB_INSERT_{camera_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                                                    )
                                                    cv2.imwrite(debug_path, debug_frame)
                                                    logger.debug(f"Saved INSERT debug frame to {debug_path}")
                                        else:
                                            # Check if current detection has better OCR confidence
                                            existing_ocr_confidence = existing_record.get('confidence', 0.0)
                                            
                                            if current_ocr_confidence > existing_ocr_confidence:
                                                existing_record_image_path = existing_record.get('image_path')
                                                # Ensure path exists before trying to remove
                                                full_path = f'{Config.BASE_DIR}/{existing_record_image_path}'
                                                if os.path.exists(full_path):
                                                    os.remove(full_path)
                                                else:
                                                    logger.warning(f"Could not find existing image at {full_path}")
                                                
                                                # Update existing record with better OCR confidence detection
                                                logger.info(f"Updating DB record for container {unique_track_id} - " +
                                                          f"Previous OCR: '{existing_record.get('ocr_output', '')}' ({existing_ocr_confidence:.3f}), " +
                                                          f"New OCR: '{current_ocr_text}' ({current_ocr_confidence:.3f}), " +
                                                          f"Improvement: +{(current_ocr_confidence - existing_ocr_confidence):.3f}")
                                                self.db_manager.update_record(existing_record['id'], record)
                                                cv2.imwrite(evidence_filename, frame)
                                                
                                                # Save debug visualization for UPDATE operation
                                                if Config.TEST_MODE:
                                                    debug_frame = self._create_debug_visualization(
                                                        frame=frame,
                                                        roi_points=camera_config['roi_points'],
                                                        detections=processed['detections'],
                                                        camera_id=camera_id,
                                                        timestamp=timestamp,
                                                        tracked_objects=self.tracked_objects,
                                                        best_detection=best_detection,
                                                        action="UPDATE",
                                                        unique_track_id=unique_track_id,
                                                        existing_record=existing_record
                                                    )
                                                    
                                                    if debug_frame is not None:
                                                        debug_path = os.path.join(
                                                            Config.DEBUG_DIR, 
                                                            f"DB_UPDATE_{camera_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                                                        )
                                                        cv2.imwrite(debug_path, debug_frame)
                                                        logger.debug(f"Saved UPDATE debug frame to {debug_path}")

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
        processed = {'detections': [], 'timestamp': timestamp}
        
        if frame is None:
            return processed

        # Get camera-specific tracker
        tracker = self.trackers[camera_id]
        
        # Run YOLO detection (without tracking)
        with torch.no_grad():
            results = model(frame, device=model.device, verbose=False)[0]

        if not results or not results.boxes:
            return processed

        # Convert YOLO results to supervision Detections format
        boxes = results.boxes.xyxy.cpu().numpy()
        confidence = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        # Create supervision Detections object with original class IDs
        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidence,
            class_id=class_ids  # Keep original class IDs for later processing
        )
        
        # Update tracks using camera-specific tracker
        tracked_detections = tracker.update_with_detections(detections)

        # Process tracked detections
        class0_detections = []
        class1_detections = []
        
        for i in range(len(tracked_detections)):
            box = tracked_detections.xyxy[i]
            conf = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 0.0
            cls = class_ids[i]  # Use original class ID
            track_id = tracked_detections.tracker_id[i] if tracked_detections.tracker_id is not None else -1
            
            detection = {
                'box': box.tolist(),
                'confidence': float(conf),
                'class_id': int(cls),
                'track_id': int(track_id),
                'in_roi': self._is_in_roi(box, roi)
            }
            
            if cls == 0:  # All containers, regardless of ROI status
                class0_detections.append(detection)
            elif cls == 1:  # All text regions
                class1_detections.append(detection)

        # Process associations between class0 and class1
        for i, class0 in enumerate(class0_detections):
            # Find associations for all containers, not just those in ROI
            class0_box = class0['box']
            potential_associations = []
            
            # Find class1 detections completely inside class0's box
            for j, class1 in enumerate(class1_detections):
                if self._is_completely_inside(class0_box, class1['box']):
                    potential_associations.append(class1)
            
            # Select best association based on detection confidence
            if potential_associations:
                best_class1 = max(potential_associations, key=lambda x: x['confidence'])
                class0['associated_class1'] = best_class1
                
                # Log association
                if class0['in_roi'] and random.random() < 0.1:  # Limit log spam
                    logger.debug(f"Associated text region (conf: {best_class1['confidence']:.3f}) with " +
                               f"container #{class0['track_id']} (in ROI: {class0['in_roi']})")
            else:
                class0['associated_class1'] = None

        processed['detections'] = class0_detections + class1_detections
        return processed

    def _update_tracking_state(self, camera_id: str, processed: dict):
        """Update tracking state with improved ROI tracking and exit detection"""
        current_time = processed['timestamp']
        
        # Track in_roi histories for debugging
        roi_changes = []
        
        # Process class0 detections and their associations
        for detection in processed['detections']:
            if detection['class_id'] == 0:  # All containers
                track_id = detection['track_id']
                key = (camera_id, track_id)  # Key includes camera_id for separation
                is_in_roi = detection['in_roi']
                
                # Create new entry for this tracked object if it doesn't exist
                if key not in self.tracked_objects:
                    self.tracked_objects[key] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'confidence_history': [detection['confidence']],
                        'positions': [detection['box']],
                        'in_roi': is_in_roi,  # Current ROI status
                        'exited_roi': False,   # Flag to track if container has left ROI after being in it
                        'first_in_roi': current_time if is_in_roi else None,  # When container first entered ROI
                        'roi_history': [(current_time, is_in_roi)]  # Track ROI changes over time
                    }
                else:
                    # Update existing tracking info
                    track_obj = self.tracked_objects[key]
                    track_obj['last_seen'] = current_time
                    track_obj['confidence_history'].append(detection['confidence'])
                    track_obj['positions'].append(detection['box'])
                    
                    # Update ROI status and history
                    prev_roi_status = track_obj['in_roi']
                    track_obj['in_roi'] = is_in_roi
                    
                    # If container was in ROI but now is not, mark as exited ROI
                    if prev_roi_status and not is_in_roi:
                        track_obj['exited_roi'] = True
                        in_roi_duration = (current_time - track_obj.get('first_in_roi', current_time)).total_seconds()
                        logger.info(f"Container #{track_id} has EXITED ROI and will no longer be processed " +
                                   f"(was in ROI for {in_roi_duration:.1f} seconds)")
                    
                    # If container entered ROI for the first time
                    if not prev_roi_status and is_in_roi and not track_obj.get('first_in_roi'):
                        track_obj['first_in_roi'] = current_time
                        logger.info(f"Container #{track_id} has ENTERED ROI for the first time")
                    
                    # Track ROI transitions for debugging
                    if prev_roi_status != is_in_roi:
                        track_obj['roi_history'].append((current_time, is_in_roi))
                        roi_changes.append((track_id, prev_roi_status, is_in_roi))
        
        # Log tracking stats occasionally
        if Config.TEST_MODE and random.random() < 0.01:  # ~1% of frames
            in_roi_count = sum(1 for k, v in self.tracked_objects.items() 
                             if k[0] == camera_id and v.get('in_roi', False))
            
            exited_roi_count = sum(1 for k, v in self.tracked_objects.items() 
                                 if k[0] == camera_id and v.get('exited_roi', False))
            
            logger.info(f"Tracking stats - Total: {len(self.tracked_objects)}, " +
                       f"In ROI: {in_roi_count}, Exited ROI: {exited_roi_count}")
        
        # Clean up old tracked objects
        self._cleanup_tracking_state(current_time)

    def _cleanup_tracking_state(self, current_time: datetime):
        """Remove tracked objects that haven't been seen recently using camera-specific tracking windows"""
        keys_to_remove = []
        for key, obj in self.tracked_objects.items():
            camera_id = key[0]
            tracking_window = self._get_camera_tracking_window(camera_id)
            time_since_last_seen = (current_time - obj['last_seen']).total_seconds()
            
            # If not seen for more than twice the tracking window, remove
            if time_since_last_seen > tracking_window * 2:
                keys_to_remove.append(key)
        
        # Remove the keys
        for key in keys_to_remove:
            del self.tracked_objects[key]

    def _is_in_roi(self, box: list, roi: np.ndarray) -> bool:
        """Check if detection is within region of interest by verifying if both 
        bottom corners of the container are inside the ROI."""
        if roi is None or len(roi) < 3:
            logger.warning("Invalid ROI configuration")
            return False
        
        # Get the bottom left and bottom right corners of the bounding box
        x_left = box[0]  # Left edge
        x_right = box[2]  # Right edge
        y_bottom = box[3]  # Bottom edge
        
        # Test if both bottom corners are inside the polygon
        left_corner_in_roi = cv2.pointPolygonTest(roi, (x_left, y_bottom), False) >= 0
        right_corner_in_roi = cv2.pointPolygonTest(roi, (x_right, y_bottom), False) >= 0
        
        # The container is in ROI only if both bottom corners are in ROI
        result = left_corner_in_roi and right_corner_in_roi
        
        if Config.TEST_MODE and random.random() < 0.01:  # Log ~1% of checks to prevent log spam
            logger.debug(f"ROI test: bottom left ({x_left:.1f}, {y_bottom:.1f}) is {'IN' if left_corner_in_roi else 'OUT'}, "
                       f"bottom right ({x_right:.1f}, {y_bottom:.1f}) is {'IN' if right_corner_in_roi else 'OUT'}, "
                       f"container is {'IN' if result else 'OUT'} of ROI")
        
        return result

    def _is_completely_inside(self, outer_box: list, inner_box: list) -> bool:
        """Check if inner bounding box is completely contained within outer box"""
        outer_x1, outer_y1, outer_x2, outer_y2 = outer_box
        inner_x1, inner_y1, inner_x2, inner_y2 = inner_box
        
        return (inner_x1 >= outer_x1 and 
                inner_y1 >= outer_y1 and 
                inner_x2 <= outer_x2 and 
                inner_y2 <= outer_y2)

    def _get_unique_track_id(self, track_id: int, camera_id: str) -> str:
        """Generate a unique track ID combining camera ID, numerical ID, and session"""
        return f"{camera_id}_{track_id}_{self.session_id}"

    def _process_ocr(self, frame: np.ndarray, detection: dict, ocr_engine: RapidOCR):
        """Perform OCR on detected region with corrected OCR confidence extraction"""
        x1, y1, x2, y2 = map(int, detection['box'])
        cropped = frame[y1:y2, x1:x2]
        
        if cropped.size == 0:
            detection['ocr_text'] = ''
            detection['ocr_confidence'] = 0.0
            return
            
        try:
            # Convert to JPEG buffer for RapidOCR
            _, buffer = cv2.imencode('.jpg', cropped)
            result, _ = ocr_engine(buffer.tobytes(), use_det=False, use_cls=False, use_rec=True, rec_use_cuda=True)
            
            if result and isinstance(result[0], list) and len(result[0]) > 1:
                # Extract OCR text from result[0][0]
                ocr_text = result[0][0]
                filtered_text = ''.join(char for char in ocr_text if char.isdigit())
                
                # Extract confidence score directly from result[0][1]
                ocr_confidence = float(result[0][1])
                
                # Store OCR results in detection
                detection['ocr_text'] = filtered_text
                detection['ocr_confidence'] = ocr_confidence
                
                # Log OCR results
                logger.info(f"OCR result for track_id {detection.get('track_id', 'unknown')}: "
                          f"text='{filtered_text}', confidence={ocr_confidence:.3f}")
                
                # Save OCR crop in test mode but only if there's actual text detected
                if Config.TEST_MODE and filtered_text:
                    try:
                        ocr_dir = os.path.join(Config.DEBUG_DIR, 'ocr_crops')
                        os.makedirs(ocr_dir, exist_ok=True)
                        
                        # Add OCR text to the crop
                        crop_with_text = cropped.copy()
                        cv2.putText(crop_with_text, f"OCR: {ocr_text}", (5, 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(crop_with_text, f"Filtered: {filtered_text}", (5, 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(crop_with_text, f"Confidence: {ocr_confidence:.2f}", (5, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        crop_filename = f"ocr_crop_{int(time.time())}_{detection.get('track_id', 'unknown')}.jpg"
                        cv2.imwrite(os.path.join(ocr_dir, crop_filename), crop_with_text)
                    except Exception as e:
                        logger.error(f"Could not save OCR crop: {str(e)}")
            else:
                detection['ocr_text'] = ''
                detection['ocr_confidence'] = 0.0
                logger.debug(f"No valid OCR result for track_id {detection.get('track_id', 'unknown')}")
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            detection['ocr_text'] = ''
            detection['ocr_confidence'] = 0.0

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
    Config.QUEUE_LOG_INTERVAL = 5  # Log queue status every 5 seconds
if not hasattr(Config, 'RECONNECT_DELAY_MAX'):
    Config.RECONNECT_DELAY_MAX = 60  # Maximum reconnect delay in seconds

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
