import sys
import logging
from pathlib import Path
import os
from sqlalchemy.sql import text
from datetime import datetime, timedelta
import cv2
import torch
import numpy as np
from config.config import BaseConfig as Config, ConfigurationError

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(Config.LOG_LEVEL)

# Create handlers
file_handler = logging.FileHandler(Config.LOG_PATH / 'setup_verification.log')
console_handler = logging.StreamHandler()

# Create formatter
formatter = logging.Formatter(Config.LOG_FORMAT)

# Set formatter for handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def verify_directories():
    """Verify all required directories exist and are writable"""
    required_dirs = [
        Config.BASE_DIR,
        Config.IMAGE_STORAGE_PATH,
        Config.LOG_PATH,
        Config.BASE_DIR / Config.EVIDENCE_IMAGES_DIR
    ]
    
    for path in required_dirs:
        if not path.exists():
            logger.error(f"Directory not found: {path}")
            return False
        if not os.access(path, os.W_OK):
            logger.error(f"Directory not writable: {path}")
            return False
    logger.info("Directory checks passed")
    return True

def verify_database():
    """Verify database connection and schema"""
    from database.db_operations import DatabaseManager
    
    try:
        db = DatabaseManager()
        # Test basic connection
        db.session.execute(text("SELECT 1"))
        
        # Test schema
        db.session.execute(text("SELECT COUNT(*) FROM container_data"))
        
        # Test database operations
        test_record = {
            'datetime': datetime.now(),
            'ocr_output': 'TEST123',
            'camera_id': 'CAM_1',
            'image_path': 'test.jpg',
            'track_id': 'TEST_ID',
            'confidence': 0.95
        }
        db.insert_record(test_record)
        db.session.rollback()  # Roll back test insert
        
        logger.info("Database checks passed")
        return True
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False

def verify_ml_dependencies():
    """Verify ML-related dependencies and models"""
    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        # Check OpenCV
        cv2_version = cv2.__version__
        logger.info(f"OpenCV version: {cv2_version}")
        
        # Check model file
        model_path = Config.BASE_DIR / Config.MODEL_PATH
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
            
        logger.info("ML dependency checks passed")
        return True
    except Exception as e:
        logger.error(f"ML dependency verification failed: {e}")
        return False

def verify_camera_configs():
    """Verify camera configurations"""
    try:
        # Validate default config
        default_config = Config.DEFAULT_CAMERA_CONFIG
        
        try:
            # Validate using the new validation method
            Config.validate_camera_config(default_config)
            
            # Additional checks for ROI points
            roi_points = default_config['roi_points']
            if not isinstance(roi_points, np.ndarray):
                raise ConfigurationError("ROI points must be a numpy array")
            
            if roi_points.shape[2] != 2:  # Check x,y coordinate pairs
                raise ConfigurationError("ROI points must be x,y coordinate pairs")
                
            if roi_points.shape[0] < Config.MIN_ROI_POINTS:
                raise ConfigurationError(f"ROI must have at least {Config.MIN_ROI_POINTS} points")
            
            # Test a point-in-polygon check to verify ROI format
            test_point = (1000, 1000)
            try:
                cv2.pointPolygonTest(roi_points, test_point, False)
            except Exception as e:
                raise ConfigurationError(f"ROI points failed OpenCV test: {e}")
            
            logger.info("Camera configuration checks passed")
            return True
            
        except Exception as e:
            logger.error(f"Camera config validation failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Camera config verification failed: {e}")
        return False

def verify_setup():
    """Verify complete setup with detailed error reporting"""
    try:
        checks = [
            ("Directory Structure", verify_directories),
            ("Database Connection", verify_database),
            ("ML Dependencies", verify_ml_dependencies),
            ("Camera Configurations", verify_camera_configs)
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            logger.info(f"Running {check_name} verification...")
            if not check_func():
                logger.error(f"{check_name} verification failed")
                all_passed = False
            else:
                logger.info(f"{check_name} verification passed")
        
        if all_passed:
            logger.info("All verification checks passed successfully!")
        else:
            logger.error("Some verification checks failed")
        
        return all_passed

    except Exception as e:
        logger.error(f"Unexpected error during verification: {e}")
        return False

if __name__ == "__main__":
    if verify_setup():
        print("\n✅ Setup verification completed successfully!")
        print("All components are properly configured and ready to use.")
    else:
        print("\n❌ Setup verification failed!")
        print("Please check the logs for detailed error information.")
        print(f"Log file: {Config.LOG_PATH / 'setup_verification.log'}") 