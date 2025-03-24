from datetime import datetime, timedelta
from pathlib import Path
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Tuple
from config.config import BaseConfig as Config

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(Config.LOG_LEVEL)

# Create log directory if it doesn't exist
log_dir = Config.LOG_PATH
log_dir.mkdir(exist_ok=True)

# Set up file handler
log_file = log_dir / 'file_utils.log'
handler = RotatingFileHandler(
    log_file, 
    maxBytes=Config.LOG_FILE_MAX_BYTES, 
    backupCount=Config.LOG_FILE_BACKUP_COUNT
)
handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
logger.addHandler(handler)

def is_image_available(image_path):
    """Check if image exists and is within retention period"""
    if not image_path:
        return False
        
    # Convert /static/images/... to actual file path
    physical_path = Config.BASE_DIR / 'frontend' / image_path.lstrip('/')
    if not physical_path.exists():
        return False
        
    # Check file age
    file_time = datetime.fromtimestamp(os.path.getctime(physical_path))
    return datetime.now() - file_time <= timedelta(days=Config.DATA_RETENTION_DAYS)

def get_image_status(image_path: str) -> Tuple[bool, str, str]:
    """Check image availability and return status and full path"""
    if not image_path:
        return False, "No image path provided", ""
        
    try:
        # If image_path is just a filename, use it directly
        if '/' in image_path:
            # If it's a path, get just the filename part
            filename = Path(image_path).name
        else:
            filename = image_path
            
        full_path = Config.IMAGE_STORAGE_PATH / filename
        
        if full_path.is_file():
            return True, "Available", str(filename)
        return False, "Image not found", ""
    except Exception as e:
        logger.error(f"Error checking image status: {e}")
        return False, f"Error checking image: {e}", ""

def cleanup_old_images():
    """Remove images older than retention period"""
    try:
        image_dir = Config.IMAGE_STORAGE_PATH
        
        current_time = datetime.now()
        retention_days = timedelta(days=Config.DATA_RETENTION_DAYS)
        removed_count = 0

        if not image_dir.exists():
            logger.warning(f"Directory not found: {image_dir}")
            return 0

        for image_file in image_dir.glob('*'):
            if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                try:
                    date_str = image_file.stem.split('_')[2]
                    time_str = image_file.stem.split('_')[3]
                    file_datetime = datetime.strptime(f"{date_str}_{time_str}", '%Y%m%d_%H%M%S')
                    
                    age = current_time - file_datetime
                    
                    if age > retention_days:
                        try:
                            image_file.unlink()
                            removed_count += 1
                            logger.info(f"Removed old image: {image_file} from {file_datetime}")
                        except Exception as e:
                            logger.error(f"Error removing {image_file}: {e}")
                except (IndexError, ValueError) as e:
                    logger.error(f"Error parsing date from filename {image_file.name}: {e}")

        if removed_count > 0:
            logger.info(f"Cleanup completed. Removed {removed_count} old images.")
        return removed_count

    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        return 0 