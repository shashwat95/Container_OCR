import os
import sys
import logging
from datetime import datetime
import subprocess
from pathlib import Path
sys.path.append('..')
from config.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/Workspace/sg/Mycodes/Repos/Container_OCR')
BACKUP_DIR = PROJECT_ROOT / 'backups'

def create_backup():
    """Create a database backup"""
    try:
        # Create backup directory if it doesn't exist
        BACKUP_DIR.mkdir(exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = BACKUP_DIR / f"ocr_system_backup_{timestamp}.sql"
        
        # Create backup using pg_dump
        command = [
            'pg_dump',
            '-h', Config.DB_HOST,
            '-p', Config.DB_PORT,
            '-U', Config.DB_USER,
            '-F', 'c',  # Custom format
            '-b',  # Include large objects
            '-v',  # Verbose
            '-f', str(backup_file),
            Config.DB_NAME
        ]
        
        # Set PGPASSWORD environment variable
        env = os.environ.copy()
        env['PGPASSWORD'] = Config.DB_PASSWORD
        
        # Execute backup command
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Backup created successfully: {backup_file}")
            return True
        else:
            logger.error(f"Backup failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Backup error: {str(e)}")
        return False

if __name__ == "__main__":
    create_backup() 