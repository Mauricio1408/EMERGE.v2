# EMERGE System Startup Script (Windows PowerShell)
# This script starts both the frontend and backend services

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "   EMERGE System Startup Script    " -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

$rootPath = $PSScriptRoot
$backendPath = Join-Path $rootPath "EMERGE - BACKEND"
$frontendPath = Join-Path $rootPath "EMERGE - FRONTEND"

# Function to start backend in a new PowerShell window
function Start-Backend {
    Write-Host "Starting Backend..." -ForegroundColor Green
    $backendScript = Join-Path $backendPath "start_backend.ps1"
    Start-Process powershell -ArgumentList "-NoExit", "-File", $backendScript
    Write-Host "Backend started in new window" -ForegroundColor Green
}

# Function to start frontend in a new PowerShell window
function Start-Frontend {
    Write-Host "Starting Frontend..." -ForegroundColor Blue
    $frontendScript = Join-Path $frontendPath "start_frontend.ps1"
    Start-Process powershell -ArgumentList "-NoExit", "-File", $frontendScript
    Write-Host "Frontend started in new window" -ForegroundColor Blue
}

# Check if directories exist
if (!(Test-Path $backendPath)) {
    Write-Host "Error: Backend directory not found at $backendPath" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $frontendPath)) {
    Write-Host "Error: Frontend directory not found at $frontendPath" -ForegroundColor Red
    exit 1
}

# Start services
Write-Host "Starting EMERGE system components..." -ForegroundColor Yellow
Start-Sleep 1

Start-Backend
Start-Sleep 3  # Give backend time to start

Start-Frontend

Write-Host ""
Write-Host "====================================" -ForegroundColor Cyan
Write-Host "System startup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Backend: http://localhost:5000" -ForegroundColor Yellow
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Close this window or press Ctrl+C to stop monitoring." -ForegroundColor Gray
Write-Host "Note: Backend and Frontend are running in separate windows." -ForegroundColor Gray
Write-Host "====================================" -ForegroundColor Cyan

# Keep the script running
try {
    while ($true) {
        Start-Sleep 10
    }
} catch {
    Write-Host "Startup script terminated." -ForegroundColor Yellow
}
