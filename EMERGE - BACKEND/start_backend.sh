#!/bin/bash
# Backend startup script for EMERGE system

echo "Starting EMERGE Backend..."
echo "========================================"

# Navigate to backend directory
cd "$(dirname "$0")"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import flask, flask_cors, geopandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install flask flask-cors geopandas pandas scikit-learn shapely
fi

# Start the Flask application
echo "Starting Flask server on http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo "========================================"
python app.py
