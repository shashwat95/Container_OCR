#!/usr/bin/env python3
import sys
from pathlib import Path

# Add project root to path
from config.config import BaseConfig as Config

# Add base directory to path if needed
if str(Config.BASE_DIR) not in sys.path:
    sys.path.append(str(Config.BASE_DIR))

from utils.dummy_data import generate_dummy_data

if __name__ == "__main__":
    # Get number of records from command line argument, default to 100
    num_records = 1000
    if len(sys.argv) > 1:
        try:
            num_records = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of records: {sys.argv[1]}")
            sys.exit(1)
    
    # Generate the data
    generate_dummy_data(num_records) 