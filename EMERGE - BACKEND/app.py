
import os
from models.RiskClusterer import RiskClusterer
from models.RiskProcessor import RiskDataProcessor

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

boundary_path = os.path.join(BASE_DIR, "DATAV2", "Santa Barbara - Barangay Boundary.gpkg")
risk_paths = {
    "Flooding": os.path.join(BASE_DIR, "DATAV2", "Risks", "Santa Barbara - Flooding.gpkg"),
    "Landslide": os.path.join(BASE_DIR, "DATAV2", "Risks", "Santa Barbara - Landslide.gpkg"),
    "Earthquake": os.path.join(BASE_DIR, "DATAV2", "Risks", "Santa Barbara - Nearest Fault.gpkg")
}
population_path = os.path.join(BASE_DIR, "DATAV2", "Risks", "Santa Barbara - Vulnerability Data.csv")

processor = RiskDataProcessor(boundary_path, risk_paths, population_path)
processor.load_data()
risk_gdf = processor.preprocess(selected_keys=["Flooding", "Landslide"], disaster_weights={"Flooding": 1.0, "Landslide": 0.7, "Earthquake": 0.5})

clusterer = RiskClusterer(n_clusters=10)
risk_with_clusters = clusterer.cluster_and_evaluate(risk_gdf)
centroid_gdf = clusterer.get_centroid_gdf(crs=processor.boundary.crs)

@app.route('/')
def home():
    return {
        'message': 'EMERGE Backend API is running!',
        'endpoints': {
            'risk_data': '/api/risk-data (POST)',
            'voronoi_data': '/api/responsiblity (POST)'
        },
        'status': 'online'
    }
    
@app.route('/api/risk-data', methods=['POST'])
def get_risk_data():
    if risk_with_clusters is None or centroid_gdf is None:
        return {"error": "Risk data or centroids could not be generated."}, 500
    
    risk_points = risk_with_clusters.to_json()
    centroids = centroid_gdf.to_json()
    boundary_clean = processor.boundary.copy()
    
    for col in boundary_clean.select_dtypes(include=["datetime64[ns]"]):
        boundary_clean[col] = boundary_clean[col].astype(str)
    
    boundary_geojson = boundary_clean.to_json()  # <-- 🔧 Convert to GeoJSON

    return {
        'risk_points': risk_points,
        'centroids': centroids,
        'boundary': boundary_geojson
    }


@app.route('/api/responsiblity', methods=['POST'])
def get_voronoi():
    voronoi = clusterer.voronoi_partitioning(processor.boundary)
    if voronoi is None:
        return {"error": "Voronoi polygons could not be generated."}, 500
    return voronoi.to_json()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)



