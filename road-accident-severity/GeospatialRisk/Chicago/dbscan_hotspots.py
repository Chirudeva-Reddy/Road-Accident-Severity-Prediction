"""Step 2: density hotspots via DBSCAN (haversine), run per community area.

Reads paths.CRASHES_WITH_AREAS; writes crashes with `cluster_id` (-1 = noise),
cluster centroids, and paths.CLUSTER_RISK (the table esri_pipeline scores).
Clustering per area bounds the ball-tree memory; clusters cannot span area borders.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN

from . import paths
from .severity_pipeline import INJURY_COLUMNS, coerce_injuries, weighted_severity

EARTH_RADIUS_M = 6_371_000.0
EPS_METERS = 250.0  # dense urban grid: ~2 city blocks
MIN_SAMPLES = 12

CLUSTER_RISK_COLUMNS = [
    "sum_fatal", "sum_incapacitating", "sum_non_incapacitating", "sum_reported_not_evident", "sum_no_indication",
]  # same order as INJURY_COLUMNS


def run_dbscan_haversine(gdf, eps_meters=EPS_METERS, min_samples=MIN_SAMPLES):
    """Point geometries in EPSG:4326 -> DBSCAN labels (-1 = noise)."""
    coords_rad = np.radians(np.column_stack([gdf.geometry.y, gdf.geometry.x]))  # haversine wants (lat, lon)
    return DBSCAN(
        eps=eps_meters / EARTH_RADIUS_M, min_samples=min_samples, metric="haversine", algorithm="ball_tree",
    ).fit_predict(coords_rad)


def compute_cluster_centroids(gdf):
    """Mean lon/lat per cluster (noise excluded)."""
    clusters = gdf[gdf["cluster_id"] >= 0]
    agg = clusters.groupby("cluster_id").agg(
        centroid_lon=("geometry", lambda p: p.x.mean()),
        centroid_lat=("geometry", lambda p: p.y.mean()),
        num_points=("geometry", "size"),
    ).reset_index()
    return gpd.GeoDataFrame(agg, geometry=gpd.points_from_xy(agg.centroid_lon, agg.centroid_lat), crs="EPSG:4326")


def compute_cluster_severity(gdf):
    """Injury sums and weighted_score (severity_pipeline formula) per cluster."""
    clusters = coerce_injuries(gdf[gdf["cluster_id"] >= 0])
    agg = clusters.groupby("cluster_id").agg(
        total_crashes=("cluster_id", "size"),
        **{out: (src, "sum") for out, src in zip(CLUSTER_RISK_COLUMNS, INJURY_COLUMNS)},
    ).reset_index()
    agg["weighted_score"] = weighted_severity(*(agg[c] for c in CLUSTER_RISK_COLUMNS))
    return agg


def compute_dbscan_hotspots(gdf, group_col="community_area_number"):
    """Cluster each area separately, then renumber so cluster ids are globally unique."""
    groups = gdf.groupby(group_col, dropna=False) if group_col in gdf.columns else [(None, gdf)]

    parts, next_id = [], 0
    for _, subset in groups:
        labels = run_dbscan_haversine(subset)
        labels[labels >= 0] += next_id
        next_id = max(next_id, labels.max() + 1)
        parts.append(subset.assign(cluster_id=labels))

    clustered = pd.concat(parts, ignore_index=True)
    print(f"Clusters: {next_id}, noise points: {(clustered['cluster_id'] == -1).sum()}")
    return clustered, compute_cluster_centroids(clustered), compute_cluster_severity(clustered)


def main():
    gdf = gpd.read_file(paths.CRASHES_WITH_AREAS)
    if gdf.empty:
        raise ValueError("No crashes available for clustering")
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS")
    gdf = gdf.to_crs(epsg=4326)

    clustered, centroids, severity = compute_dbscan_hotspots(gdf)
    clustered.to_file(paths.CRASHES_WITH_CLUSTERS, driver="GeoJSON")
    centroids.to_file(paths.CLUSTER_CENTROIDS, driver="GeoJSON")
    severity.to_csv(paths.CLUSTER_RISK, index=False)

    print("Worst 10 clusters by severity score:")
    print(severity.nlargest(10, "weighted_score"))


if __name__ == "__main__":
    main()
