#! /bin/bash

echo "Starting container and pulling latest changes..."

# Ensure we're in the right directory
cd /app/RWKV-MedSAM2

# Pull the latest changes
git pull origin main || echo "Failed to pull latest changes"

# Run the main process
exec "$@"