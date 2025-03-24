from pathlib import Path
import os
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import sys

# Add project root to path
PROJECT_ROOT = Path('/Workspace/sg/Mycodes/Repos/Container_OCR')
FRONTEND_ROOT = PROJECT_ROOT / 'frontend'

# Set up logging
log_file = PROJECT_ROOT / 'logs/cleanup.log'
handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def cleanup_old_images():
    """Remove images older than 30 days"""
    try:
        # Only one directory to clean
        image_dir = FRONTEND_ROOT / 'static/images'
        
        cutoff_date = datetime.now() - timedelta(days=30)
        removed_count = 0

        if not image_dir.exists():
            logger.warning(f"Directory not found: {image_dir}")
            return 0

        logger.info(f"Checking directory: {image_dir}")
        for image_file in image_dir.glob('*'):
            if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                file_time = datetime.fromtimestamp(os.path.getctime(image_file))
                if file_time < cutoff_date:
                    try:
                        image_file.unlink()
                        removed_count += 1
                        logger.info(f"Removed old image: {image_file}")
                    except Exception as e:
                        logger.error(f"Error removing {image_file}: {e}")

        logger.info(f"Cleanup completed. Removed {removed_count} old images.")
        return removed_count

    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        return 0

if __name__ == '__main__':
    cleanup_old_images() 