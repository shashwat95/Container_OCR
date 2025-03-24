import os
from pathlib import Path
from dotenv import load_dotenv
import secrets
import logging
import numpy as np
import re
from .exceptions import (
    ConfigurationError,
    ValidationError,
    StorageError
)

# Load environment variables from .env file
load_dotenv()

class BaseConfig:
    """Base configuration class with all system settings"""
    
    # =====================
    # Camera Settings
    # =====================
    # Camera configurations - can be single dict for one camera or list for multiple cameras
    DEFAULT_CAMERA_CONFIG = [
        {
            'rtsp_url': 'rtsp://admin:Admin-1234@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0',
            'camera_id': 'CAM_1',
            'location': 'Weightmeasure',
            'roi_points': np.array([[471,1043], [742,758], [1473,739], [1772,1002]], dtype=np.int32)
        },
        {
            'rtsp_url': 'rtsp://admin:Admin-1234@192.168.1.11:554/cam/realmonitor?channel=1&subtype=0',
            'camera_id': 'CAM_2',
            'location': 'Trainyard',
            'roi_points': np.array([[497,1077], [1533,1078], [1494,507], [577,558]], dtype=np.int32)
        },
        {
            'rtsp_url': 'rtsp://admin:Admin-1234@192.168.1.12:554/cam/realmonitor?channel=1&subtype=0',
            'camera_id': 'CAM_3',
            'location': 'Camera 3',
            'roi_points': np.array([[495,1075], [1640,1078], [1564,658], [478,619]], dtype=np.int32)
        }        
        # Add more cameras as needed
    ]

    # Camera validation settings
    CAMERA_CONFIG_REQUIRED_KEYS = ['rtsp_url', 'camera_id', 'location', 'roi_points']
    CAMERA_ID_PATTERN = r'^CAM_\d+$'
    MIN_ROI_POINTS = 3  # Minimum number of points for a valid ROI polygon
    ROI_SHAPE = (-1, 1, 2)  # Expected shape for OpenCV contour operations
    
    @classmethod
    def format_roi_points(cls, points):
        """Format ROI points to the correct numpy array shape"""
        if isinstance(points, list):
            points = np.array(points, dtype=np.int32)
        if not isinstance(points, np.ndarray):
            raise ConfigurationError("ROI points must be a numpy array or list")
        if points.dtype != np.int32:
            points = points.astype(np.int32)
        if len(points) < cls.MIN_ROI_POINTS:
            raise ConfigurationError(f"ROI must have at least {cls.MIN_ROI_POINTS} points")
        return points.reshape(cls.ROI_SHAPE)

    @classmethod
    def validate_camera_config(cls, config):
        """Validate a camera configuration"""
        if not all(k in config for k in cls.CAMERA_CONFIG_REQUIRED_KEYS):
            raise ConfigurationError(f"Missing required camera config keys: {cls.CAMERA_CONFIG_REQUIRED_KEYS}")
        if not re.match(cls.CAMERA_ID_PATTERN, config['camera_id']):
            raise ConfigurationError(f"Invalid camera_id format: {config['camera_id']}")
        try:
            config['roi_points'] = cls.format_roi_points(config['roi_points'])
        except Exception as e:
            raise ConfigurationError(f"Invalid ROI points: {str(e)}")
        return config
    
    # =====================
    # Directory Settings
    # =====================
    # Use environment variables with defaults for path configuration
    BASE_DIR = Path(os.getenv('BASE_DIR', '/app'))
    IMAGE_STORAGE_PATH = Path(BASE_DIR / 'evidence_images')
    LOG_PATH = Path(os.getenv('LOG_PATH', BASE_DIR / 'logs'))
    EVIDENCE_IMAGES_DIR = 'evidence_images'
    TEST_MODE = False
    # Temporary directory for exports
    TEMP_DIR = BASE_DIR / 'temp'
    os.makedirs(TEMP_DIR, exist_ok=True)

    # =====================
    # Database Settings
    # =====================
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://admin:admin123@localhost:5432/ocr_system')
    # Parse DATABASE_URL for individual components
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'ocr_system')
    
    # If DATABASE_URL is provided, use it directly; otherwise build it from components
    SQLALCHEMY_DATABASE_URI = DATABASE_URL or f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
    
    # Database Pool Settings
    DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '5'))
    DB_MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', '10'))
    DB_POOL_TIMEOUT = int(os.getenv('DB_POOL_TIMEOUT', '30'))
    DB_POOL_RECYCLE = int(os.getenv('DB_POOL_RECYCLE', '1800'))

    # =====================
    # Logging Settings
    # =====================
    LOG_LEVELS = {
        'development': logging.DEBUG,
        'production': logging.INFO
    }
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    LOG_LEVEL = LOG_LEVELS.get(os.getenv('FLASK_ENV', 'development'))
    LOG_FILE_MAX_BYTES = int(os.getenv('LOG_FILE_MAX_BYTES', '10000000'))  # 10MB
    LOG_FILE_BACKUP_COUNT = int(os.getenv('LOG_FILE_BACKUP_COUNT', '5'))

    # =====================
    # ML Model Settings
    # =====================
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/Container_Detector_V2.engine')
    DROP_FRAME = int(os.getenv('DROP_FRAME', '3'))  # Frame dropping configuration (1 = no drop, 2 = every second frame, etc.)
    QUEUE_SIZE = int(os.getenv('QUEUE_SIZE', '500'))  # Maximum size of frame queues
    
    # =====================
    # Stream Settings
    # =====================
    RECONNECT_DELAY_BASE = int(os.getenv('RECONNECT_DELAY_BASE', '2'))  # Base seconds for exponential backoff
    RECONNECT_DELAY_MAX = int(os.getenv('RECONNECT_DELAY_MAX', '30'))  # Maximum seconds between reconnection attempts
    PUSH_DATA = os.getenv('PUSH_DATA', 'True').lower() in ('true', '1', 't')  # Flag to control data pushing to the database
    
    # =====================
    # Monitoring Settings
    # =====================
    LOGGING_INTERVAL = int(os.getenv('LOGGING_INTERVAL', '10'))  # Interval for logging queue sizes in seconds
    TRACKING_WINDOW = int(os.getenv('TRACKING_WINDOW', '10'))  # Time window in seconds to consider tracked objects active
    METRICS_PORT = int(os.getenv('METRICS_PORT', '8000'))  # Port for Prometheus metrics
    LONG_TRACKING_WINDOW = int(os.getenv('LONG_TRACKING_WINDOW', str(12 * 3600)))

    # =====================
    # Tracking Settings
    # =====================
    TRACKING_CLEANUP_ROI = int(os.getenv('TRACKING_CLEANUP_ROI', str(12 * 3600)))  # 12 hours in seconds
    TRACKING_CLEANUP_NON_ROI = int(os.getenv('TRACKING_CLEANUP_NON_ROI', '30'))    # 30 seconds
    TRACKING_CLEANUP_DEFAULT = int(os.getenv('TRACKING_CLEANUP_DEFAULT', '60'))     # 60 seconds

    # =====================
    # Application Settings
    # =====================
    RECORDS_PER_PAGE = int(os.getenv('RECORDS_PER_PAGE', '50'))
    DATA_RETENTION_DAYS = int(os.getenv('DATA_RETENTION_DAYS', '30'))
    SECRET_KEY = os.getenv('SECRET_KEY', secrets.token_urlsafe(32))
    
    # Rate Limiting
    RATE_LIMIT_DAY = os.getenv('RATE_LIMIT_DAY', "200 per day")
    RATE_LIMIT_HOUR = os.getenv('RATE_LIMIT_HOUR', "50 per hour")

    # Cache Settings
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'simple')
    CACHE_DEFAULT_TIMEOUT = int(os.getenv('CACHE_DEFAULT_TIMEOUT', '300'))

    # Redis settings
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        required_settings = [
            'SQLALCHEMY_DATABASE_URI',
            'SECRET_KEY'
        ]
        
        # Check required settings
        for setting in required_settings:
            if not getattr(cls, setting, None):
                raise ConfigurationError(f"Missing required setting: {setting}")
        
        # Validate paths
        required_paths = [cls.IMAGE_STORAGE_PATH, cls.LOG_PATH]
        for path in required_paths:
            path.mkdir(parents=True, exist_ok=True)
            if not os.access(path, os.W_OK):
                raise ConfigurationError(f"Path not writable: {path}")

    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL with validation"""
        if not cls.SQLALCHEMY_DATABASE_URI:
            raise ConfigurationError("Database URL not configured")
        return cls.SQLALCHEMY_DATABASE_URI

