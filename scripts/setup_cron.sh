#!/bin/bash

# Get the absolute path to the cleanup script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup_images.py"

# Make the cleanup script executable
chmod +x "$CLEANUP_SCRIPT"

# Add daily cleanup job at midnight
(crontab -l 2>/dev/null; echo "0 0 * * * /usr/bin/python3 $CLEANUP_SCRIPT") | crontab -

echo "Cron job installed successfully" 