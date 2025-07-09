from cProfile import label
from shapely.geometry import Point
from shapely.ops import voronoi_diagram, unary_union
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from shapely.geometry import Polygon
from scipy.spatial import Voronoi


import geopandas as gpd
import pandas as pd
import numpy as np

import joblib
from pathlib import Path

class RiskClusterer:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.centroids = None
        self.labels = {}
        self.best_algo = None
        
    def _clean_coords(self, gdf):
        """ Basically, it just checks if there are empty data and if the data is within the boundary of Santa Barbara"""
        
        coords = np.array([[geom.x, geom.y, gdf.iloc[i]['wV']] for i, geom in enumerate(gdf.geometry)])
        print("Before filtering:", coords.shape)
        print(gdf.columns)
        print(gdf.head())
        mask = ~np.isnan(coords).any(axis=1)
        print("After filtering:", mask.sum())
        return coords[mask], gdf.iloc[mask].reset_index(drop=True)
    
    def cluster_and_evaluate(self, gdf, cache_path="./.cache/centroids.pkl"):
        coords, gdf_clean = self._clean_coords(gdf)
        results = {}
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=39)
        kmedoids = KMedoids(n_clusters=self.n_clusters, random_state=39)
        agglo = AgglomerativeClustering(n_clusters=self.n_clusters)
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        
        algos = {
            "KMeans": (kmeans, lambda m: m.cluster_centers_),
            "KMedoids": (kmedoids, lambda m: m.cluster_centers_),
            "Agglomerative": (agglo, lambda m: np.vstack([coords[labels == i].mean(axis=0) for i in range(self.n_clusters)])),
            "DBSCAN": (dbscan, lambda m: np.vstack([coords[labels == i].mean(axis=0) for i in set(label) if i != -1]))
        }
        
        results = {}
        labels_dict = {}

        for name, (model, _) in algos.items():
            try:
                labels = model.fit_predict(coords)
                labels_dict[name] = labels

                if len(set(labels)) <= 1 or (name == "DBSCAN" and -1 in set(labels)):
                    print(f"{name} produced invalid clustering.")
                    results[name] = None
                    continue

                sil = silhouette_score(coords, labels)
                ch = calinski_harabasz_score(coords, labels)
                results[name] = (sil, ch)

            except Exception as e:
                print(f"⚠️ {name} failed: {e}")
                results[name] = None

        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            raise RuntimeError("No valid clustering method.")

        self.best_algo = max(valid_results, key=lambda k: valid_results[k][0])
        print(f"✅ Best algorithm: {self.best_algo}")

        # Assign best model and labels
        labels = labels_dict[self.best_algo]
        self.labels[self.best_algo] = labels

        # Compute cluster centers
        if self.best_algo == "Agglomerative":
            centers = np.vstack([coords[labels == i].mean(axis=0)
                                for i in range(self.n_clusters)
                                if len(coords[labels == i]) > 0])
        elif self.best_algo == "DBSCAN":
            centers = np.vstack([coords[labels == i].mean(axis=0)
                                for i in set(labels) if i != -1 and len(coords[labels == i]) > 0])
        else:
            model, center_fn = algos[self.best_algo]
            model.fit(coords)
            centers = center_fn(model)

        # Check for NaN and create geometry
        if np.isnan(centers).any():
            raise ValueError("❌ Invalid cluster centers detected (NaN).")

        self.centroids = centers
        gdf_clean['cluster'] = labels
        
        return gdf_clean
        
    def get_centroid_gdf(self, crs, cache_path="./.cache/centroids.pkl"):
        if self.centroids is None and Path(cache_path).exists():
            self.centroids = joblib.load(cache_path)
        return gpd.GeoDataFrame(geometry=[Point(xy) for xy in self.centroids], crs=crs)
    
    def voronoi_partitioning(self, boundary_gdf):
        """
        Creates Voronoi polygons for the cluster centroids bounded by the given area.
        """
        # Prepare points
        points = np.array([[x, y] for x, y, *_ in self.centroids])
        boundary = boundary_gdf.unary_union

        # Compute Voronoi
        vor = Voronoi(points)
        polygons = []
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue  # skip infinite regions
            poly = Polygon([vor.vertices[i] for i in region])
            poly = poly.intersection(boundary)
            if not poly.is_empty:
                polygons.append(poly)

        voronoi_gdf = gpd.GeoDataFrame(geometry=polygons, crs=boundary_gdf.crs)
        return voronoi_gdf