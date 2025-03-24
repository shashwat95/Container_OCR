import random
import time
from datetime import datetime, timedelta
import sys
import logging
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path('/Workspace/sg/Mycodes/Repos/Container_OCR')
sys.path.append(str(PROJECT_ROOT))

from database.db_operations import DatabaseManager
from config.config import Config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dummy_image(path: str, text: str = "DUMMY") -> None:
    """Create a dummy image with text"""
    # Create a new image with white background
    width = 800
    height = 600
    color = 'white'
    
    # Create image and drawing context
    image = Image.new('RGB', (width, height), color)
    draw = ImageDraw.Draw(image)
    
    # Add some random shapes
    for _ in range(5):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        draw.line([(x1, y1), (x2, y2)], fill='black', width=2)
    
    # Add text
    draw.text((width//2 - 50, height//2), text, fill='black')
    
    # Save the image
    image.save(path)

def generate_container_number():
    """Generate a realistic container number"""
    shipping_lines = ['MAEU', 'CMAU', 'MSCU', 'OOLU', 'HLCU']
    letters = random.choice(shipping_lines)
    numbers = ''.join([str(random.randint(0, 9)) for _ in range(7)])
    return f"{letters}{numbers}"

class DummyOCRGenerator:
    def __init__(self):
        self.camera_ids = ['CAM_01', 'CAM_02', 'CAM_03', 'CAM_04', 'CAM_05']
        self.db_manager = DatabaseManager()

    def generate_container_number(self):
        """Generate a realistic container number."""
        shipping_lines = ['MAEU', 'CMAU', 'MSCU', 'OOLU', 'HLCU']
        letters = random.choice(shipping_lines)
        numbers = ''.join([str(random.randint(0, 9)) for _ in range(7)])
        return f"{letters}{numbers}"

    def generate_single_record(self, specific_datetime=None):
        """Generate a single OCR record."""
        if specific_datetime is None:
            specific_datetime = datetime.now() - timedelta(
                days=random.randint(1, 60),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

        # Generate paths
        image_name = f"image_{random.randint(1000,9999)}_{specific_datetime.strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = str(Config.IMAGE_STORAGE_PATH / image_name)
        
        # Create dummy image
        create_dummy_image(image_path)
        
        return {
            'datetime': specific_datetime,
            'ocr_output': self.generate_container_number(),
            'camera_id': random.choice(self.camera_ids),
            'image_path': f"/static/images/{image_name}"
        }

    def generate_batch(self, num_records=100):
        """Generate and insert a batch of OCR records."""
        records = []
        for _ in range(num_records):
            records.append(self.generate_single_record())

        try:
            self.db_manager.bulk_insert(records)
            return num_records, 0  # successful, failed
        except Exception as e:
            logger.error(f"Failed to insert batch: {e}")
            return 0, num_records  # successful, failed

    def generate_realtime_simulation(self, duration_minutes=60, interval_seconds=10):
        """
        Simulate real-time OCR data generation for a specified duration.
        
        Args:
            duration_minutes (int): How long to run the simulation
            interval_seconds (int): Interval between records
        """
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        records_generated = 0

        while datetime.now() < end_time:
            record = self.generate_single_record(datetime.now())
            if self.db_manager.insert_ocr_result(record):
                records_generated += 1
                logger.info(f"Generated record: {record['ocr_output']} from {record['camera_id']}")
            
            time.sleep(interval_seconds)

        return records_generated

def generate_dummy_data(num_records=100):
    """Generate dummy OCR data"""
    db = DatabaseManager()
    
    # Ensure directories exist with absolute paths
    os.makedirs(Config.IMAGE_STORAGE_PATH, exist_ok=True)
    
    # Generate records with absolute paths
    records = []
    current_time = datetime.now()
    
    for i in range(num_records):
        # Generate timestamps within last 60 days
        timestamp = current_time - timedelta(
            days=random.randint(1, 60),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Use absolute paths for images
        image_name = f"image_{i}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        image_path = str(Config.IMAGE_STORAGE_PATH / image_name)
        
        # Create dummy image
        create_dummy_image(image_path)
        
        # Create record
        record = {
            'datetime': timestamp,
            'ocr_output': generate_container_number(),
            'camera_id': f"CAM_{random.randint(1, 5):02d}",
            'image_path': f"/static/images/{image_name}"
        }
        records.append(record)
    
    # Insert records
    try:
        db.bulk_insert(records)
        print(f"Successfully inserted {num_records} records")
    except Exception as e:
        print(f"Error inserting records: {e}")
    finally:
        db.session.close()

if __name__ == "__main__":
    # Example usage
    generator = DummyOCRGenerator()
    
    # Generate a batch of 100 historical records
    print("Generating historical data...")
    successful, failed = generator.generate_batch(100)
    print(f"Generated {successful} records successfully ({failed} failed)")
    
    # Simulate real-time generation for 5 minutes with 10-second intervals
    print("\nStarting real-time simulation...")
    records = generator.generate_realtime_simulation(duration_minutes=5, interval_seconds=10)
    print(f"Generated {records} records in real-time simulation") 