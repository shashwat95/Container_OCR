import os
import sys
import subprocess
from pathlib import Path
from config.config import BaseConfig as Config

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required")
        sys.exit(1)

def setup_conda_environment():
    """Setup Conda environment"""
    try:
        # Check if conda is available
        subprocess.run(['conda', '--version'], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Conda is not installed or not in PATH")
        print("Please install Miniconda or Anaconda first")
        return False

    try:
        # Get project root directory
        env_file = Config.BASE_DIR / 'environment.yml'

        if not env_file.exists():
            print(f"Error: environment.yml not found at {env_file}")
            return False

        # Check if environment exists
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        if 'Container_OCR' not in result.stdout:
            print("Creating Conda environment 'Container_OCR'...")
            subprocess.run(['conda', 'env', 'create', '-f', str(env_file)], check=True)
        else:
            print("Conda environment 'Container_OCR' already exists")
            print("Updating environment...")
            subprocess.run(['conda', 'env', 'update', '-f', str(env_file), '-n', 'Container_OCR'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up Conda environment: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    try:
        directories = [
            Config.BASE_DIR / 'frontend/static/images',
            Config.BASE_DIR / 'frontend/static/evidence',
            Config.LOG_PATH,
            Config.BASE_DIR / 'backups'
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    project_root = Path(__file__).parent.parent
    env_file = project_root / '.env'
    
    if not env_file.exists():
        env_content = """# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ocr_system
DB_USER=postgres
DB_PASSWORD=your_password

# Storage Paths
IMAGE_STORAGE_PATH=frontend/static/images
EVIDENCE_STORAGE_PATH=frontend/static/evidence
LOG_PATH=logs

# Application Settings
FLASK_ENV=development
RECORDS_PER_PAGE=50
DATA_RETENTION_DAYS=30
"""
        try:
            with open(env_file, 'w') as f:
                f.write(env_content)
            print("Created .env file")
            return True
        except Exception as e:
            print(f"Error creating .env file: {e}")
            return False
    return True

def create_pgpass():
    """Create .pgpass file if it doesn't exist"""
    pgpass_path = os.path.expanduser('~/.pgpass')
    if not os.path.exists(pgpass_path):
        try:
            with open(pgpass_path, 'w') as f:
                f.write("localhost:5432:*:postgres:your_password")
            os.chmod(pgpass_path, 0o600)
            print("Created .pgpass file")
            return True
        except Exception as e:
            print(f"Error creating .pgpass file: {e}")
            return False
    return True

def print_manual_steps():
    """Print manual setup steps"""
    print("\nManual Setup Steps:")
    print("1. Install PostgreSQL if not already installed:")
    print("   Follow PostgreSQL installation guide for your system")
    print("\n2. Create PostgreSQL database:")
    print("   $ psql -h localhost -U postgres")
    print("   postgres=# CREATE DATABASE ocr_system;")
    print("   postgres=# \\q")
    print("\n3. Edit configuration files:")
    print("   - Edit .env with your database credentials")
    print("   - Edit ~/.pgpass with your database credentials")
    print("\n4. Initialize database:")
    print("   $ psql -h localhost -U postgres -d ocr_system -f database/schema.sql")

def main():
    """Main setup function"""
    print("Starting setup...")
    
    # Check Python version
    check_python_version()

    # Create directories first (needed for environment.yml)
    if not setup_directories():
        print("Failed to create directories")
        return

    # Create configuration files
    if not create_env_file():
        print("Failed to create .env file")
        return

    if not create_pgpass():
        print("Failed to create .pgpass file")
        return

    # Setup conda environment
    if not setup_conda_environment():
        print("Failed to setup Conda environment")
        return

    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file with your database credentials")
    print("2. Edit ~/.pgpass with your database credentials")
    print("3. Activate the conda environment: conda activate Container_OCR")
    print("4. Install dependencies: conda env update -f environment.yml")
    print("5. Start the application: python frontend/app.py")
    
    print_manual_steps()

if __name__ == "__main__":
    main() 