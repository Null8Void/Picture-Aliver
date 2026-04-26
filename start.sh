#!/bin/bash
# Picture-Aliver Start Script for Linux/Mac
# Usage: ./start.sh [image_path] [prompt]

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check arguments
if [ -z "$1" ]; then
    echo "Starting API server..."
    uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Running CLI..."
    python main.py --image "$1" --prompt "$2"
fi