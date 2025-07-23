# Backend startup script for EMERGE system (Windows PowerShell)

Write-Host "Starting EMERGE Backend..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Yellow

# Navigate to backend directory
Set-Location $PSScriptRoot

# Check if Python is available
try {
    python --version | Out-Null
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Check if required packages are installed
Write-Host "Checking dependencies..." -ForegroundColor Blue
try {
    python -c "import flask, flask_cors, geopandas" 2>$null
} catch {
    Write-Host "Installing required packages..." -ForegroundColor Blue
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
    } else {
        pip install flask flask-cors geopandas pandas scikit-learn shapely
    }
}

# Start the Flask application
Write-Host "Starting Flask server on http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
python app.py
