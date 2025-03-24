import schedule
import time
from backup_db import create_backup
import logging
from pathlib import Path
import sys
sys.path.append('..')
from config.config import Config

logging.basicConfig(
    filename=Config.LOG_PATH / 'backup.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_old_backups():
    """Remove backups older than 30 days"""
    try:
        backup_dir = Path('backups')
        for backup_file in backup_dir.glob('*.sql'):
            if backup_file.stat().st_mtime < time.time() - 30 * 86400:
                backup_file.unlink()
                logger.info(f"Removed old backup: {backup_file}")
    except Exception as e:
        logger.error(f"Error cleaning up old backups: {str(e)}")

def scheduled_backup():
    """Create backup and cleanup old ones"""
    if create_backup():
        cleanup_old_backups()

def main():
    # Schedule daily backup at 2 AM
    schedule.every().day.at("02:00").do(scheduled_backup)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main() 