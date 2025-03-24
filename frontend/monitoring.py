from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil
import time
from datetime import datetime
import logging
from config.config import BaseConfig as Config

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(Config.LOG_LEVEL)
handler = logging.FileHandler(Config.LOG_PATH / 'monitoring.log')
handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
logger.addHandler(handler)

# Initialize start time when module is loaded
START_TIME = datetime.now()

# Metrics
REQUEST_DURATION = Histogram(
    'request_duration_seconds',
    'Request duration in seconds',
    ['endpoint']
)

DB_CONNECTIONS = Gauge(
    'database_connections',
    'Number of active database connections'
)

SYSTEM_METRICS = Gauge(
    'system_metrics',
    'System metrics',
    ['metric_type']
)

OCR_PROCESSING = Counter(
    'ocr_processing_total',
    'Total number of OCR operations',
    ['camera_id', 'status']
)

IMAGE_STORAGE = Gauge(
    'image_storage_usage',
    'Image storage usage statistics',
    ['type']
)

def start_metrics_server():
    """Start the Prometheus metrics server"""
    try:
        start_http_server(Config.METRICS_PORT)
        logger.info(f"Metrics server started on port {Config.METRICS_PORT}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

def collect_system_metrics():
    """Collect system metrics with error handling"""
    try:
        # Get basic system metrics
        uptime_delta = datetime.now() - START_TIME
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'start_time': START_TIME.isoformat(),
            'uptime': str(uptime_delta),
            'uptime_seconds': uptime_delta.total_seconds(),
        }

        # Get camera status
        camera_status = []
        for i in range(1, 6):  # Assuming 5 cameras
            camera_id = f'CAM_{i}'
            status = check_camera_status(camera_id)
            camera_status.append({'id': camera_id, 'status': status})
        
        metrics['camera_status'] = camera_status

        # Update Prometheus metrics
        SYSTEM_METRICS.labels('cpu_usage').set(metrics['cpu_percent'])
        SYSTEM_METRICS.labels('memory_usage').set(metrics['memory_percent'])
        SYSTEM_METRICS.labels('disk_usage').set(metrics['disk_percent'])

        return metrics
    except Exception as e:
        logger.error(f"Failed to collect system metrics: {e}")
        return None

def check_camera_status(camera_id):
    """Check camera status"""
    try:
        # For now, let's say CAM_1 through CAM_3 and CAM_5 are online, CAM_4 is offline
        status_map = {
            'CAM_1': True,
            'CAM_2': True,
            'CAM_3': True,
            'CAM_4': False,
            'CAM_5': True
        }
        return status_map.get(camera_id, False)
    except Exception as e:
        logger.error(f"Error checking camera status: {e}")
        return False

def update_storage_metrics():
    """Update storage-related metrics"""
    try:
        image_dir = Config.IMAGE_STORAGE_PATH
        if image_dir.exists():
            total_size = sum(f.stat().st_size for f in image_dir.glob('**/*') if f.is_file())
            file_count = sum(1 for _ in image_dir.glob('**/*') if _.is_file())
            
            IMAGE_STORAGE.labels('total_size_bytes').set(total_size)
            IMAGE_STORAGE.labels('file_count').set(file_count)
    except Exception as e:
        logger.error(f"Error updating storage metrics: {e}")

# Initialize metrics server on module load
start_metrics_server() 
