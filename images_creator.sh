#!/bin/bash

# Define an array of Docker images and output file names
images=(
    "container_ocr-ocr_app ocr_app.tar"
    "nginx:1.25-alpine nginx.tar"
    "postgres:14-alpine postgres.tar"
    "redis:7-alpine redis.tar"
    "prom/prometheus:v2.44.0 prometheus.tar"
    "grafana/grafana:9.5.2 grafana.tar"
    "prom/node-exporter:latest node-exporter.tar"
)

# Create a logs directory if it doesn't exist
mkdir -p logs

# Loop through each image and save it
for entry in "${images[@]}"; do
    set -- $entry
    image="$1"
    output="$2"
    
    echo "Saving $image to $output..."
    docker save -o "$output" "$image" > "logs/$output.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Successfully saved $image to $output"
    else
        echo "Failed to save $image to $output. Check logs/$output.log for details."
    fi

done

echo "All Docker images have been processed."

