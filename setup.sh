#!/bin/bash
# Picture-Aliver Setup Script for Linux/Mac
# Usage: bash setup.sh

set -e

echo "========================================"
echo "Picture-Aliver Setup"
echo "========================================"

# Check Python version
echo "[1/5] Checking Python..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $PYTHON_VERSION"

# Create virtual environment (optional)
if [ ! -d "venv" ]; then
    echo "[2/5] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "[3/5] Installing dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories
echo "[4/5] Creating directories..."
mkdir -p outputs models checkpoints debug uploads

# Download default models (if needed)
echo "[5/5] Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the app:"
echo "  python main.py --image test.png --prompt 'motion'"
echo ""
echo "To run the API server:"
echo "  uvicorn src.picture_aliver.api:app --host 0.0.0.0 --port 8000"
echo ""
echo "To run the desktop app:"
echo "  python desktop/pyqt/main.py"