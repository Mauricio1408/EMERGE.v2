# EMERGE System - Frontend & Backend Connection Guide

This guide explains how to connect and run the EMERGE emergency management system, which consists of a Next.js frontend and a Flask backend.

## System Architecture

- **Frontend**: Next.js React application (Port 3000)
- **Backend**: Flask Python API server (Port 5000)
- **Data**: GeoSpatial data processing and risk analysis

## Prerequisites

### For Backend (Python/Flask):
- Python 3.8 or higher
- pip (Python package manager)
- Required Python packages:
  - flask
  - flask-cors
  - geopandas
  - pandas
  - scikit-learn
  - shapely

### For Frontend (Node.js/Next.js):
- Node.js 18 or higher
- npm or yarn package manager

## Quick Start

### Option 1: Automated Startup (Recommended)

1. **Open PowerShell as Administrator** in the EMERGESys directory
2. **Run the startup script**:
   ```powershell
   .\start_emerge.ps1
   ```
   This will automatically start both backend and frontend in separate windows.

### Option 2: Manual Startup

#### Start Backend:
1. Navigate to the backend directory:
   ```powershell
   cd "EMERGE - BACKEND"
   ```

2. Install Python dependencies (first time only):
   ```powershell
   pip install flask flask-cors geopandas pandas scikit-learn shapely
   ```

3. Start the Flask server:
   ```powershell
   python app.py
   ```
   Backend will be available at: http://localhost:5000

#### Start Frontend:
1. Open a new terminal and navigate to frontend directory:
   ```powershell
   cd "EMERGE - FRONTEND"
   ```

2. Install Node.js dependencies (first time only):
   ```powershell
   npm install
   ```

3. Start the development server:
   ```powershell
   npm run dev
   ```
   Frontend will be available at: http://localhost:3000

## API Endpoints

The backend provides the following API endpoints:

- **POST /api/risk-data**: Returns risk analysis data including risk points, centroids, and boundary data
- **POST /api/responsiblity**: Returns Voronoi polygon data for responder allocation

## Frontend Integration

The frontend includes:

### API Utilities (`src/utils/api.ts`):
- `fetchRiskData()`: Fetches risk analysis data
- `fetchVoronoiData()`: Fetches responder allocation data
- `checkBackendHealth()`: Checks backend connectivity

### React Hooks (`src/hooks/useApi.ts`):
- `useRiskData()`: Hook for managing risk data state
- `useVoronoiData()`: Hook for managing Voronoi data state
- `useBackendHealth()`: Hook for monitoring backend status

### Updated Pages:
- **Risk Map** (`/risk-map`): Displays real-time risk data from backend
- **Responder Allocation** (`/responder-allocation`): Shows Voronoi zones for resource allocation

## Features

### Real-time Data Integration:
- Risk scores and population data from backend analysis
- Interactive hazard filtering (Flood, Landslide, Earthquake)
- Dynamic risk statistics display

### Backend Health Monitoring:
- Automatic connection status indicators
- Retry functionality for failed requests
- Loading states and error handling

### Responsive Design:
- Works on desktop and mobile devices
- Interactive sidebars and controls
- Status notifications and feedback

## Development

### Environment Variables:
Create a `.env.local` file in the frontend directory:
```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### Debugging:
- Backend logs appear in the backend terminal
- Frontend development tools available in browser
- Network requests visible in browser dev tools

## Troubleshooting

### Backend Issues:
- **Port 5000 in use**: Change port in `app.py` and update frontend API URL
- **Missing dependencies**: Run `pip install -r requirements.txt` (if available)
- **CORS errors**: Ensure `flask-cors` is installed and configured

### Frontend Issues:
- **API connection failed**: Check if backend is running on port 5000
- **Build errors**: Run `npm install` to update dependencies
- **Port 3000 in use**: Next.js will automatically suggest an alternative port

### Data Issues:
- **No risk data**: Ensure all data files are present in `DATAV2` directory
- **Parsing errors**: Check browser console for GeoJSON parsing issues

## File Structure

```
EMERGESys/
├── EMERGE - BACKEND/
│   ├── app.py                 # Flask API server
│   ├── Backend.py             # Backend utilities
│   ├── start_backend.ps1      # Backend startup script
│   ├── models/                # Risk analysis models
│   └── DATAV2/               # Geospatial data files
├── EMERGE - FRONTEND/
│   ├── src/
│   │   ├── app/              # Next.js pages
│   │   ├── utils/api.ts      # API utilities
│   │   └── hooks/useApi.ts   # React hooks
│   ├── start_frontend.ps1    # Frontend startup script
│   └── package.json          # Node.js dependencies
└── start_emerge.ps1          # Main startup script
```

## Next Steps

1. **Map Integration**: Consider integrating Leaflet or Mapbox for interactive maps
2. **Real-time Updates**: Implement WebSocket connections for live data updates
3. **User Authentication**: Add login system for different user roles
4. **Database Integration**: Connect to a database for persistent data storage
5. **Deployment**: Configure for production deployment with proper security measures

## Support

For issues or questions:
1. Check the browser console for frontend errors
2. Check the backend terminal for Python errors
3. Verify all dependencies are installed correctly
4. Ensure both services are running on correct ports
