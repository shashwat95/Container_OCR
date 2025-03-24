from prometheus_client import Counter, Histogram, Gauge
import psutil
import time
from datetime import datetime, timedelta
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

def collect_system_metrics():
    """Collect system metrics with error handling"""
    try:
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
        return metrics
    except Exception as e:
        logger.error(f"Failed to collect system metrics: {e}")
        return None

def check_camera_status(camera_id):
    """Temporary function returning fixed camera status"""
    # For now, let's say CAM_01, CAM_02, CAM_03 are online, CAM_04 is offline, and CAM_05 is online
    status_map = {
        'CAM_01': True,
        'CAM_02': True,
        'CAM_03': True,
        'CAM_04': False,
        'CAM_05': True
    }
    return status_map.get(camera_id, False) 