#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Define an array of tar files
images=(
    "ocr_app.tar"
    "nginx.tar"
    "postgres.tar"
    "redis.tar"
    "prometheus.tar"
    "grafana.tar"
    "node-exporter.tar"
)

# Loop through each tar file and load it into Docker
for image in "${images[@]}"; do
    echo "Loading $image..."
    docker load -i "$image"
    echo "$image loaded successfully."
done

echo "All images loaded successfully."

