# Loads the model given the particular data

import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

import os
import json
from hashlib import md5
from joblib import Parallel, delayed

class RiskDataProcessor:
    def __init__(self, boundary_path, risk_paths, population_path):
        self.boundary_path = boundary_path 
        self.risk_paths = risk_paths
        self.population_path = population_path
        self.boundary = None
        self.risk_layers = {}
        self.population = None
        self.risk_gdf = None
        
    def load_data(self):
        """Load data and ensure consistent CRS"""
        self.boundary = gpd.read_file(self.boundary_path)
        self.population = pd.read_csv(self.population_path)
        
        # Load risk layers and ensure consistent CRS
        for name, path in self.risk_paths.items():
            layer = gpd.read_file(path)
            print(f"Loading {name}: Original CRS = {layer.crs}")
            self.risk_layers[name] = layer
            
        # Set boundary CRS if not set
        if self.boundary.crs is None:
            print("⚠️  Boundary has no CRS, setting to EPSG:4326")
            self.boundary.set_crs('EPSG:4326', inplace=True)
        
        print(f"Final boundary CRS: {self.boundary.crs}")
    
        # Check if risk layers need CRS conversion
        for name, layer in self.risk_layers.items():
            if layer.crs is None:
                print(f"⚠️  {name} has no CRS, setting to EPSG:4326")
                layer.set_crs('EPSG:4326', inplace=True)
            elif layer.crs != self.boundary.crs:
                print(f"🔄 Converting {name} from {layer.crs} to {self.boundary.crs}")
                self.risk_layers[name] = layer.to_crs(self.boundary.crs)
            
    def hash_file(self, filepath):
        with open(filepath, 'rb') as f:
            return md5(f.read()).hexdigest()
        
    def should_recompute(self, file_paths, cache_path = "./.cache", cache_meta="./.cache/cache_meta.json"):
        # Check for changes in the file, will be implemening upload function soon!
        current_hashes = {p: self.hash_file(p) for p in file_paths}
        
        if not Path(cache_path).exists():
            os.makedirs(Path(cache_path), exist_ok=True)
        
        if not Path(cache_meta).exists():
            with open(cache_meta, 'w') as f:
                json.dump(current_hashes, f)
            return True
        
        with open(cache_meta, 'r') as f:
            saved_hashes = json.load(f)
            
        if current_hashes != saved_hashes:
            with open(cache_meta, 'w') as f:
                json.dump(current_hashes, f)
            return True

        return False
    
    def _clip_and_centroid(self, layer, name):
        """Clip layer to boundary and get centroids - ENSURE CONSISTENT CRS"""
        clipped = gpd.overlay(layer, self.boundary, how="intersection")
        
        # Make sure we're working in the same CRS as boundary
        if clipped.crs != self.boundary.crs:
            clipped = clipped.to_crs(self.boundary.crs)
        
        centroids = clipped.geometry.centroid
        
        # Debug: Print CRS and sample coordinates
        print(f"Layer {name} - CRS: {centroids.crs}")
        if len(centroids) > 0:
            print(f"Sample centroid: {centroids.iloc[0]}")
        
        return centroids, name
    
    def compute_weight_vulnerability(self, method="sigmoid"):
        v_raw = self.population["Children"] + self.population["Elderly"] + self.population["PWD"] - self.population["Medical Access"]
        
        if method == "minmax":
            # Fixed parentheses issue
            scaled = (v_raw - v_raw.min()) / (v_raw.max() - v_raw.min())
        elif method == "sigmoid":
            z = (v_raw - v_raw.mean()) / v_raw.std()
            scaled = 1 / (1 + np.exp(-z))
        elif method == "log":
            scaled = np.log1p(v_raw - v_raw.min()) / np.log1p(v_raw.max() - v_raw.min())
        else:
            raise ValueError("Invalid Scaling Method!")

        self.population["wV"] = scaled
        return self.population
            
    def preprocess(self, selected_keys, disaster_weights, cache_path='./.cache/risk_gdf.pkl', scaling_method="sigmoid"):
        all_paths = [self.boundary_path, self.population_path] + list(self.risk_paths.values())
        if not self.should_recompute(all_paths):
            print("🍏Cached risks is up-to-date!")
            print("🔃Loading risk_gdf from cache...")
            self.risk_gdf = joblib.load(cache_path)
            return self.risk_gdf
        
        print("Change detected! Recomputing risk_data!")
        
        # Paralellize the clipping and Centroid for each of the defined risk zones
        results = Parallel(n_jobs=-1)(delayed(self._clip_and_centroid)(self.risk_layers[k], k) for k in selected_keys)
        
        all_points, all_weights = [], []
        for centroids, key in results:
            all_points.extend(centroids)
            all_weights.extend([disaster_weights[key]] * len(centroids))
        
        gdf = gpd.GeoDataFrame({'geometry': all_points, 'weights': all_weights}, crs=self.boundary.crs)
        print("After creating GDF: risk_score NaNs =", gdf.isna().sum().sum())
        
        # Spatial join - this can introduce NaNs if points are outside boundaries
        gdf = gpd.sjoin(gdf, self.boundary[['ADM4_EN', 'geometry']], how='left', predicate='within')
        print("After spatial join: NaNs in ADM4_EN =", gdf['ADM4_EN'].isna().sum())
        
        # Compute vulnerability weights
        self.compute_weight_vulnerability(method=scaling_method)
        print("After computing vulnerability: NaNs in wV =", self.population['wV'].isna().sum())
        
        # Merge with population data - this can introduce NaNs if ADM4_EN values don't match
        gdf = gdf.merge(self.population[['ADM4_EN', 'wV']], on='ADM4_EN', how='left')
        print("After merge: NaNs in wV =", gdf['wV'].isna().sum())
        print("After merge: NaNs in weights =", gdf['weights'].isna().sum())
        
        # Calculate risk score - NaNs will propagate here
        gdf['risk_score'] = gdf['wV'] * gdf['weights']
        print("After risk calculation: NaNs in risk_score =", gdf['risk_score'].isna().sum())
        
        # Optional: Handle NaN values
        # You might want to either drop NaN rows or fill them
        initial_count = len(gdf)
        gdf = gdf.dropna(subset=['risk_score'])  # Remove rows with NaN risk scores
        print(f"Dropped {initial_count - len(gdf)} rows with NaN risk scores")
        
        self.risk_gdf = gdf
        joblib.dump(self.risk_gdf, cache_path)
        return self.risk_gdf