# Container OCR System

## Overview

The **Container OCR System** is a comprehensive solution designed to process and manage container-related data using Optical Character Recognition (OCR). It provides functionalities for data ingestion, processing, and visualization through a web-based dashboard. The system is built with modular components, including a frontend for user interaction, backend scripts for data processing, and utilities for maintenance and testing.

## Features

- **OCR Data Processing**: Extract and manage container numbers from images.
- **Web Dashboard**: Visualize system metrics, OCR results, and camera statuses.
- **Duplicate Handling**: Automatically remove duplicate records based on confidence levels.
- **System Metrics**: Monitor CPU usage, memory usage, uptime, and camera statuses.
- **Data Export**: Export OCR results to Excel for further analysis.
- **Maintenance Scripts**: Cleanup old images and generate dummy data for testing.
- **Docker Support**: Easily deploy and manage the system using Docker.

## Project Structure

- **frontend/**: Contains the web-based dashboard and static assets.
- **scripts/**: Includes setup and maintenance scripts.
- **utils/**: Utility scripts for generating dummy data and other helper functions.
- **Makefile**: Provides commands for setup, testing, running, and Docker management.

## Key Functionalities and Logic

### 1. **Duplicate Record Handling**
   - Implemented in `frontend/static/js/main.js`.
   - The `checkAndRemoveDuplicates` function ensures that only the record with the highest confidence is retained for each `track_id`.
   - Logic:
     - Iterate through table rows and group them by `track_id`.
     - Compare confidence levels and remove rows with lower confidence.

### 2. **System Metrics Update**
   - Implemented in `frontend/static/js/main.js`.
   - The `updateSystemMetrics` function dynamically updates CPU usage, memory usage, and system uptime on the dashboard.
   - Logic:
     - Fetch metrics data via WebSocket.
     - Update DOM elements with the latest values.

### 3. **Cleanup Old Images**
   - Implemented in `scripts/cleanup_images.py`.
   - The `cleanup_old_images` function removes images older than 30 days to free up storage.
   - Logic:
     - Traverse the image directory.
     - Compare file modification dates with the current date.
     - Delete files exceeding the 30-day threshold.

### 4. **Dummy Data Generation**
   - Implemented in `utils/dummy_data.py`.
   - The `generate_dummy_data` function creates synthetic OCR data for testing purposes.
   - Logic:
     - Generate random container numbers and timestamps.
     - Save the data in a predefined format.

### 5. **Web Dashboard**
   - Implemented in `frontend/templates/dashboard.html`.
   - Displays OCR results, system metrics, and camera statuses.
   - Features:
     - Export results to Excel.
     - Real-time updates for system metrics.
   - Access the dashboard at `http://localhost:5000`.
 
### 6. **Postgre Database Configurations**
   - DB_PORT=5433
   - POSTGRES_USER=admin
   - POSTGRES_PASSWORD=admin123
   - POSTGRES_DB=ocr_system
   - POSTGRES_DB_Table=container_data

## Maintenance

- **Cleanup Images**:
  ```bash
  python scripts/cleanup_images.py
  ```

- **Generate Dummy Data**:
  ```bash
  python utils/dummy_data.py
  ```



