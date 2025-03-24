import cv2
import threading
import torch
from ultralytics import YOLO
import numpy as np
import os
import logging
from datetime import datetime

class VideoPipeline:
    def __init__(self, camera_configs, test_mode=False):
        """
        Initialize the pipeline with camera configurations.
        Each config should include 'rtsp_url', 'camera_id', 'location', 
        'width', and 'height'.
        
        Args:
            camera_configs: List of camera configuration dictionaries
            test_mode: Boolean flag to enable test mode with visualizations
        """
        self.camera_configs = camera_configs
        self.streams = {}
        self.tracked_objects = {}
        
        # Set up CUDA device if available
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Load model on the available device
        self.model = YOLO('/Workspace/sg/Mycodes/Yolo/runs/detect/train138/weights/best.pt').to(self.device)
        
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
        
    def setup_logging(self):
        """
        Set up logging configuration for test mode.
        """
        log_file = os.path.join(self.output_dir, 'pipeline.log')
        
        # Configure root logger
        self.logger = logging.getLogger('VideoPipeline')
        self.logger.setLevel(logging.DEBUG)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler with higher level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Logging initialized")
        self.logger.info(f"YOLO model loaded from: /Workspace/sg/Mycodes/Yolo/runs/detect/train138/weights/best.pt")
        
        # Log camera configurations
        for i, config in enumerate(self.camera_configs):
            self.logger.info(f"Camera {i+1} configuration: {config}")
            
    def start_streams(self):
        """
        Start capturing video from all RTSP streams in parallel.
        """
        if self.test_mode:
            self.logger.info(f"Starting {len(self.camera_configs)} camera streams")
            
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

    def capture_stream(self, rtsp_url, camera_id):
        """
        Capture video stream from the given RTSP URL.
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
        
        # Update resolution with actual values from capture if different
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width > 0 and actual_height > 0:
            self.camera_resolutions[camera_id] = (actual_width, actual_height)
            
            if self.test_mode:
                self.logger.info(f"Camera {camera_id} resolution: {actual_width}x{actual_height}")
        
        # Get and log stream properties
        if self.test_mode:
            fps = cap.get(cv2.CAP_PROP_FPS)
            codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
            
            self.logger.info(f"Stream properties for camera {camera_id}:")
            self.logger.info(f"  - Resolution: {actual_width}x{actual_height}")
            self.logger.info(f"  - FPS: {fps}")
            self.logger.info(f"  - Codec: {codec_str}")
        
        # Initialize video writer for test mode
        if self.test_mode:
            width, height = self.camera_resolutions[camera_id]
            output_path = os.path.join(self.output_dir, f"{camera_id}_debug.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30  # Adjust as needed
            self.video_writers[camera_id] = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            self.logger.info(f"Created video writer for camera {camera_id}: {output_path}")
            
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                error_msg = f'Error: Unable to read frame from stream for camera {camera_id}'
                if self.test_mode:
                    self.logger.error(error_msg)
                    self.logger.info(f"Stream ended for camera {camera_id} after {frame_id} frames")
                else:
                    print(error_msg)
                break
            
            if self.test_mode and frame_id % 100 == 0:  # Log every 100 frames to avoid excessive logging
                self.logger.debug(f"Processing frame {frame_id} from camera {camera_id}")
            
            # Create a copy of the frame for visualization if in test mode
            if self.test_mode:
                vis_frame = frame.copy()
            else:
                vis_frame = None
                
            roi = self.define_roi()
            self.run_detection(frame, roi, frame_id, camera_id, vis_frame)
            frame_id += 1
            
            # In test mode, save to video but do not display (no display connected)
            if self.test_mode and vis_frame is not None:
                self.video_writers[camera_id].write(vis_frame)
            
        cap.release()
        
        # Clean up resources in test mode
        if self.test_mode:
            if camera_id in self.video_writers:
                self.video_writers[camera_id].release()
                self.logger.info(f"Released video writer for camera {camera_id}")
            self.logger.info(f"Stream processing completed for camera {camera_id} after {frame_id} frames")

    def define_roi(self):
        """
        Define the region of interest (ROI) as a polygon.
        """
        roi = np.array([[273,1072],[533,841],[1409,807],[1856,1066]], np.int32)
        if self.test_mode:
            self.logger.debug(f"Defined ROI with points: {roi.tolist()}")
        return roi

    def run_detection(self, frame, roi, frame_id, camera_id, vis_frame=None):
        """
        Run YOLOv8 detection with tracking on the given frame within the specified ROI.
        
        Args:
            frame: The input video frame
            roi: Region of interest polygon
            frame_id: Current frame ID
            camera_id: Camera identifier
            vis_frame: Optional frame for visualization (in test mode)
        """
        if self.test_mode and frame_id % 100 == 0:  # Limit logging frequency
            self.logger.debug(f"--- Starting detection on frame {frame_id} from camera {camera_id} ---")
            
        results = self.model.track(frame, persist=True, device=self.device)
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            # Draw ROI in test mode even if no detections
            if self.test_mode:
                self.logger.debug(f"No detections in frame {frame_id} from camera {camera_id}")
                if vis_frame is not None:
                    self.draw_roi(vis_frame, roi)
                    self.draw_predefined_point(vis_frame)
            return
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id
        
        if track_ids is None:
            # Draw ROI in test mode even if no tracking
            if self.test_mode:
                self.logger.debug(f"No tracking IDs in frame {frame_id} from camera {camera_id}")
                if vis_frame is not None:
                    self.draw_roi(vis_frame, roi)
                    self.draw_predefined_point(vis_frame)
            return
            
        track_ids = track_ids.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names  # Get class names from model
        
        # Draw ROI in test mode
        if self.test_mode and vis_frame is not None:
            self.draw_roi(vis_frame, roi)
            self.draw_predefined_point(vis_frame)
        
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
                filtered_detections.append(detection);  # Always add class ID 1 detections
                class1_in_roi_count += 1;
                if self.test_mode:
                    class_name = class_names[int(cls)] if int(cls) in class_names else f"Class {int(cls)}"
                    self.logger.debug(
                        f"Class 1 object detected - ID {int(track_id)}, {class_name}, "
                        f"conf {conf:.2f}, box [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                    )
            
            # In test mode, visualize all detections but with different colors
            if self.test_mode and vis_frame is not None:
                color = (0, 255, 0) if in_roi else (0, 0, 255);  # Green if in ROI, red if outside
                self.draw_detection(vis_frame, box, track_id, conf, cls, class_names, color);
        
        if self.test_mode:
            self.logger.debug(f"ROI Summary: Found {class0_in_roi_count} class 0 objects and {class1_in_roi_count} class 1 objects inside ROI")
        
        # Stage 2: Process associations between class 0 and class 1 objects
        if self.test_mode:
            self.logger.debug(f"STAGE 2: Processing associations between class 0 and class 1 objects")
        
        class0_detections = [d for d in filtered_detections if d['class_id'] == 0]
        class1_detections = [d for d in filtered_detections if d['class_id'] == 1]  # No ROI check for class 1
        if self.test_mode:
            self.logger.debug(f"Class 0 Detections: {len(class0_detections)}")
            self.logger.debug(f"Class 1 Detections: {len(class1_detections)}")
        association_count = 0;
        
        for class0_detection in class0_detections:
            class0_id = int(class0_detection['track_id'])
            class0_box = class0_detection['box']
            potential_associations = []

            if self.test_mode:
                self.logger.debug(f"Checking associations for Class 0 ID {class0_id}, box {[int(coord) for coord in class0_box]}")

            for class1_detection in class1_detections:
                class1_id = int(class1_detection['track_id'])
                class1_box = class1_detection['box']
                if self.test_mode:
                    self.logger.debug(f"Checking Class 1 ID {class1_id} for association")

                if self.is_completely_inside(class0_box, class1_box):
                    potential_associations.append(class1_detection)
                    if self.test_mode:
                        self.logger.debug(
                            f"Found potential association: Class 1 ID {class1_id} is inside Class 0 ID {class0_id}, "
                            f"conf {class1_detection['conf']:.2f}, box {[int(coord) for coord in class1_box]}"
                        )

            if not potential_associations:
                if self.test_mode:
                    self.logger.debug(f"No Class 1 objects found inside Class 0 ID {class0_id}")
                continue

            # Find the best association (highest confidence)
            best_association = max(potential_associations, key=lambda x: x['conf'])
            best_class1_id = int(best_association['track_id'])

            # Create or update association
            if 'associated_class_1' not in class0_detection:
                class0_detection['associated_class_1'] = best_association
                if self.test_mode:
                    self.logger.debug(
                        f"New association created: Class 0 ID {class0_id} ↔ Class 1 ID {best_class1_id}, "
                        f"conf {best_association['conf']:.2f}"
                    )
                association_count += 1
            else:
                current_conf = class0_detection['associated_class_1']['conf']
                if best_association['conf'] > current_conf:
                    if self.test_mode:
                        previous_id = int(class0_detection['associated_class_1']['track_id'])
                        self.logger.debug(
                            f"Association updated: Class 0 ID {class0_id} - "
                            f"replaced Class 1 ID {previous_id} (conf {current_conf:.2f}) with "
                            f"Class 1 ID {best_class1_id} (conf {best_association['conf']:.2f})"
                        )
                    class0_detection['associated_class_1'] = best_association
        
        if self.test_mode:
            self.logger.debug(f"Association Summary: Created {association_count} associations between class 0 and class 1 objects");
            # Log all successful associations
            for class0_detection in class0_detections:
                if 'associated_class_1' in class0_detection:
                    class0_id = int(class0_detection['track_id']);
                    class1_id = int(class0_detection['associated_class_1']['track_id']);
                    conf = class0_detection['associated_class_1']['conf'];
                    self.logger.debug(f"Final association: Class 0 ID {class0_id} ↔ Class 1 ID {class1_id}, conf {conf:.2f}");
        
        # Stage 3: Apply voter logic to get best detection
        if self.test_mode:
            self.logger.debug(f"STAGE 3: Applying voter logic to select best detection")
        
        best_detection = self.voter_logic(camera_id, filtered_detections, frame_id, vis_frame if self.test_mode else None)
        
        # Log final result
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
        
        # In test mode, highlight the best detection
        if self.test_mode and vis_frame is not None and best_detection is not None:
            box = best_detection['box']
            x1, y1, x2, y2 = box
            conf = best_detection['conf']
            cls = best_detection['class_id']
            track_id = best_detection['track_id']

            # Draw a thicker, yellow box around the best detection
            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3)
            
            # Add "BEST" label
            cv2.putText(vis_frame, "BEST", (int(x1), int(y1)-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            # Crop the best detection
            cropped_image = vis_frame[int(y1):int(y2), int(x1):int(x2)]
            if cropped_image.size > 0:  # Check if the cropped area is valid
                # Resize the cropped image to fit in the top right corner
                cropped_image_resized = cv2.resize(cropped_image, (100, 100))
                # Overlay the cropped image in the top right corner
                vis_frame[0:100, vis_frame.shape[1]-100:vis_frame.shape[1]] = cropped_image_resized
            
            # Add frame number and timestamp
            timestamp = f"Frame: {frame_id}"
            cv2.putText(vis_frame, timestamp, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def update_tracking(self, camera_id, track_id, detection, frame_id):
        """
        Update tracking data for a specific object
        """
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
            # Log tracking milestone (every 10 frames)
            track_length = len(self.tracked_objects[key]['detections'])
            if track_length % 10 == 0:
                avg_conf = sum(self.tracked_objects[key]['confidence']) / len(self.tracked_objects[key]['confidence'])
                class_id = detection['class_id']
                self.logger.debug(
                    f"Tracking milestone: Camera {camera_id}, Class {int(class_id)} ID {int(track_id)} - "
                    f"Tracked for {track_length} frames, avg confidence: {avg_conf:.3f}"
                )
        
        # Limit the history to prevent memory issues
        if len(self.tracked_objects[key]['detections']) > 30:
            self.tracked_objects[key]['detections'].pop(0)
            self.tracked_objects[key]['confidence'].pop(0)

    def voter_logic(self, camera_id, detections, frame_id, vis_frame=None):
        """
        Apply voter logic to select the best detection.
        
        Args:
            camera_id: Camera identifier
            detections: List of detection dictionaries
            frame_id: Current frame ID
            vis_frame: Optional frame for visualization
            
        Returns:
            The best detection or None if no suitable detection found
        """
        if self.test_mode:
            self.logger.debug(f"Starting voter logic for frame {frame_id}, camera {camera_id}")
        
        best_detections = {}  # Dictionary to hold best detections for each class ID 0
        class0_with_associations = [d for d in detections if d['class_id'] == 0 and 'associated_class_1' in d]
        
        if self.test_mode:
            if class0_with_associations:
                self.logger.debug(f"Voter input: {len(class0_with_associations)} class 0 objects with class 1 associations")
                for i, detection in enumerate(class0_with_associations):
                    class0_id = int(detection['track_id'])
                    class1_id = int(detection['associated_class_1']['track_id'])
                    class1_conf = detection['associated_class_1']['conf']
                    self.logger.debug(
                        f"Voter candidate {i+1}: Class 0 ID {class0_id} with "
                        f"Class 1 ID {class1_id}, conf {class1_conf:.2f}"
                    )
            else:
                self.logger.debug("Voter input: No class 0 objects with class 1 associations found")
                return None
        
        # Process each class 0 object with associations
        for detection in class0_with_associations:
            track_id = detection['track_id']
            if track_id not in best_detections:
                best_detections[track_id] = {'best_score': 0, 'best_detection': None}
            
            associated_class_1 = detection['associated_class_1']
            score = associated_class_1['conf']  # Use confidence as the score
            
            if self.test_mode:
                self.logger.debug(
                    f"Voting process: Class 0 ID {int(track_id)} with "
                    f"Class 1 ID {int(associated_class_1['track_id'])}, score {score:.3f}"
                )

            # Update the best detection if the score is higher
            if score > best_detections[track_id]['best_score']:
                if self.test_mode:
                    self.logger.debug(
                        f"New best score for Class 0 ID {int(track_id)}: {score:.3f} " 
                        f"from Class 1 ID {int(associated_class_1['track_id'])}"
                    )
                
                best_detections[track_id]['best_score'] = score
                best_detections[track_id]['best_detection'] = associated_class_1
                best_detections[track_id]['best_box'] = associated_class_1['box']
        
        # Find the overall best detection
        best_detection_result = None
        highest_score = 0
        best_track_id = None
        
        if self.test_mode:
            self.logger.debug("Determining overall best detection across all class 0 objects:")
        
        for track_id, best_detection_info in best_detections.items():
            if best_detection_info['best_detection'] is not None:
                current_score = best_detection_info['best_score']
                associated_id = int(best_detection_info['best_detection']['track_id'])
                
                if self.test_mode:
                    self.logger.debug(
                        f"Candidate: Class 0 ID {int(track_id)} with "
                        f"Class 1 ID {associated_id}, score {current_score:.3f}"
                    )
                
                if current_score > highest_score:
                    highest_score = current_score
                    best_track_id = track_id
                    best_detection_result = {
                        'box': best_detection_info['best_box'],
                        'conf': best_detection_info['best_detection']['conf'],
                        'class_id': best_detection_info['best_detection']['class_id'],
                        'track_id': best_detection_info['best_detection']['track_id'],
                        'parent_class0_id': track_id  # Store the parent class 0 ID for reference
                    }
                    
                    if self.test_mode:
                        self.logger.debug(
                            f"New leader: Class 0 ID {int(track_id)} with "
                            f"Class 1 ID {associated_id}, score {current_score:.3f}"
                        )
        
        if self.test_mode:
            if best_track_id is not None:
                associated_id = int(best_detection_result['track_id'])
                self.logger.debug(
                    f"Voter decision: Selected Class 0 ID {int(best_track_id)} with "
                    f"Class 1 ID {associated_id}, final score {highest_score:.3f}"
                )
            else:
                self.logger.debug("Voter decision: No suitable detection found")

        return best_detection_result
    
    # Helper methods for visualization in test mode
    def draw_roi(self, frame, roi):
        """Draw ROI polygon on frame"""
        pts = roi.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
        cv2.putText(frame, "ROI", (roi[0][0], roi[0][1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    def draw_predefined_point(self, frame):
        """Draw the predefined point that's used in voter logic"""
        cv2.circle(frame, self.predefined_point, 8, (255, 255, 0), -1)
        cv2.putText(frame, "Target Point", (self.predefined_point[0]+10, self.predefined_point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def draw_detection(self, frame, box, track_id, conf, cls, class_names, color=(0, 255, 0)):
        """Draw detection box and information"""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label with ID, class and confidence
        class_name = class_names[int(cls)] if int(cls) in class_names else f"Class {int(cls)}"
        label = f"ID:{int(track_id)} {class_name} {conf:.2f}"
        
        # Calculate label position and draw background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.rectangle(frame, (x1, y1-20), (x1+text_size[0], y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    def visualize_voter_logic(self, frame, scores_data):
        """Visualize voter logic scores in a table format"""
        # Draw a semi-transparent overlay for the table
        overlay = frame.copy()
        table_height = len(scores_data) * 30 + 40  # Header + rows
        table_width = 350
        cv2.rectangle(overlay, (10, 50), (10+table_width, 50+table_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw table header
        cv2.putText(frame, "ID", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Conf(30%)", (70, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Dist(70%)", (170, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Score", (270, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw horizontal line below header
        cv2.line(frame, (10, 80), (10+table_width, 80), (255, 255, 255), 1)
        
        # Sort scores by total score, highest first
        scores_data.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Draw scores for each detection
        for i, data in enumerate(scores_data):
            y_pos = 105 + i * 30
            
            # Use yellow for the highest score, white for others
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            
            cv2.putText(frame, f"{data['id']}", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"{data['conf']:.3f}", (70, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"{data['dist_score']:.3f}", (170, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, f"{data['total_score']:.3f}", (270, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # Add a visual indicator from the object to its score in the table
            if i < 3:  # Only for top 3 to avoid clutter
                cv2.line(frame, data['position'], (10, y_pos), color, 1)

        # Visualize the best associated class ID 1 detection for each class ID 0
        for data in scores_data:
            if 'best_detection' in data:
                best_detection = data['best_detection']
                box = best_detection['box']
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)  # Draw in red
                cv2.putText(frame, f"Best ID: {best_detection['track_id']}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Crop the best detection
                cropped_image = frame[box[1]:box[3], box[0]:box[2]]
                # Resize the cropped image to fit in the top right corner
                cropped_image_resized = cv2.resize(cropped_image, (100, 100))
                # Overlay the cropped image in the top right corner
                frame[0:100, frame.shape[1]-100:frame.shape[1]] = cropped_image_resized

    def is_inside_roi(self, box, roi):
        """
        Check if the bottom-left and bottom-right corners of a bounding box are inside the polygon ROI.
        
        Args:
            box: Bounding box in format [x1, y1, x2, y2]
            roi: Region of interest polygon as np.array of points
            
        Returns:
            bool: True if both bottom corners are inside the ROI, False otherwise
        """
        # Extract coordinates from box
        x1, y1, x2, y2 = box
        
        # Check only bottom-left and bottom-right corners
        bottom_left = (x1, y2)
        bottom_right = (x2, y2)
        
        # Check if both bottom corners are inside the ROI
        bottom_left_inside = cv2.pointPolygonTest(roi, bottom_left, False) >= 0
        bottom_right_inside = cv2.pointPolygonTest(roi, bottom_right, False) >= 0
        
        # Return True only if both bottom corners are inside the ROI
        return bottom_left_inside and bottom_right_inside

    def is_completely_inside(self, outer_box, inner_box):
        # Check if inner_box is completely inside outer_box
        outer_x1, outer_y1, outer_x2, outer_y2 = outer_box
        inner_x1, inner_y1, inner_x2, inner_y2 = inner_box
        return (inner_x1 >= outer_x1 and inner_y1 >= outer_y1 and inner_x2 <= outer_x2 and inner_y2 <= outer_y2)

    def associate_detections(self, class_0_detection, class_1_detection):
        # Logic to associate class ID 1 detection with class ID 0 detection
        class_0_detection['associated_class_1'] = class_1_detection

    def get_associated_class_1_detections(self, class_0_detection):
        # Retrieve associated class ID 1 detections for a given class ID 0 detection
        return class_0_detection.get('associated_class_1', [])

    def visualize_associations(self, detections):
        # Visualization logic to confirm associations
        for detection in detections:
            if 'associated_class_1' in detection:
                # Draw bounding boxes for class ID 0 detection
                self.draw_association(detection, detection['associated_class_1'])  # Draw association
                
                # Optionally add labels for associated class ID 1 detection
                associated_class_1 = detection['associated_class_1']
                box_1 = associated_class_1['box']
                cv2.putText(frame, f"Class 1 ID: {associated_class_1['track_id']}", (box_1[0], box_1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Label in red

    def draw_association(self, class_0_detection, class_1_detection):
        # Draw bounding boxes for class ID 0 and associated class ID 1 detections
        box_0 = class_0_detection['box']  # Assuming box is in format [x1, y1, x2, y2]
        box_1 = class_1_detection['box']
        color_0 = (255, 0, 0)  # Blue for associated class ID 0
        color_1 = (255, 0, 0)  # Blue for associated class ID 1

        # Draw box for class ID 0
        cv2.rectangle(frame, (box_0[0], box_0[1]), (box_0[2], box_0[3]), color_0, 2)
        # Draw box for class ID 1
        cv2.rectangle(frame, (box_1[0], box_1[1]), (box_1[2], box_1[3]), color_1, 2)
        # Optionally add labels
        cv2.putText(frame, 'Class 0', (box_0[0], box_0[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_0, 2)
        cv2.putText(frame, 'Class 1', (box_1[0], box_1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_1, 2)

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
        
        # {
        #     'rtsp_url': 'rtsp://camera2', 
        #     'camera_id': 'cam2', 
        #     'location': 'Lobby',
        #     'width': 1280,
        #     'height': 720
        # },
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