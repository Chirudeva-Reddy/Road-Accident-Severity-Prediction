"""Step 4 (ArcGIS): Getis-Ord Gi* on DBSCAN clusters + trauma-centre drive-time coverage.

Inputs:  paths.CLUSTER_CENTROIDS, paths.CLUSTER_RISK (from dbscan_hotspots)
Outputs: paths.ESRI_HOTSPOTS_CSV/GEOJSON, paths.ESRI_ISOCHRONES, map PNG in reports/ and report/figures/

Only the isochrones need ArcGIS Online (Network Analyst service areas, needs ARCGIS_API_KEY
from env/.env). If that call fails, EMS tiers fall back to a straight-line drive-time estimate.

Run: python -m GeospatialRisk.Chicago.esri_pipeline [--test-auth] [--api-key KEY]
"""

from __future__ import annotations

import argparse
import json
import os

import setuptools  # noqa: F401  must precede arcgis on Python 3.12+ (distutils shim)
import geopandas as gpd
import numpy as np
import pandas as pd
from arcgis.features import GeoAccessor  # noqa: F401  registers the DataFrame.spatial accessor
from arcgis.gis import GIS
import arcgis.network.analysis as network_analysis
from dotenv import load_dotenv
from scipy import stats

from . import paths

# Chicago adult Level-1 trauma centres (lat, lon).
CHICAGO_TRAUMA_CENTERS = pd.DataFrame([
    ("John H. Stroger Jr. Hospital (Cook County)", 41.8741, -87.6749),
    ("University of Chicago Medical Center", 41.7891, -87.6046),
    ("Northwestern Memorial Hospital", 41.8946, -87.6214),
    ("Advocate Illinois Masonic Medical Center", 41.9363, -87.6534),
    ("Mount Sinai Hospital", 41.8601, -87.6953),
], columns=["name", "latitude", "longitude"]).assign(facility_type="Adult Level 1 Trauma Center")

# (upper bound in minutes, label); a cluster gets the first tier it falls inside.
EMS_TIERS = [(5, "Optimal (<= 5 min)"), (8, "Golden Window (5-8 min)"), (12, "Delayed (8-12 min)")]
EMS_CRITICAL = "Critical Gap (> 12 min)"
# Offline fallback: urban ambulance ~25 km/h, road distance ~1.25x straight line.
FALLBACK_SPEED_KMH, ROUTE_CIRCUITY = 25.0, 1.25


def init_arcgis_gis(api_key: str | None = None) -> GIS:
    """API key (arg, then ARCGIS_API_KEY) -> authenticated GIS; otherwise anonymous (no routing)."""
    load_dotenv()
    key = api_key or os.getenv("ARCGIS_API_KEY")
    gis = GIS("https://www.arcgis.com", api_key=key) if key else GIS()
    user = gis.users.me.username if gis.users.me else "API key / anonymous"
    print(f"[ESRI] Connected to {gis.properties.get('portalName', 'ArcGIS Online')} as {user}")
    return gis


def load_cluster_sedf() -> pd.DataFrame:
    """Cluster centroids joined to their risk row, as a Spatially Enabled DataFrame (SHAPE column)."""
    sedf = pd.DataFrame.spatial.from_geodataframe(gpd.read_file(paths.CLUSTER_CENTROIDS))
    if paths.CLUSTER_RISK.exists():
        sedf = sedf.merge(pd.read_csv(paths.CLUSTER_RISK), on="cluster_id", how="left")
    return sedf


def sedf_to_gdf(sedf: pd.DataFrame) -> gpd.GeoDataFrame:
    """Point SEDF -> GeoDataFrame (EPSG:4326)."""
    return gpd.GeoDataFrame(
        sedf.drop(columns=["SHAPE"]),
        geometry=gpd.points_from_xy([s.x for s in sedf["SHAPE"]], [s.y for s in sedf["SHAPE"]]),
        crs="EPSG:4326",
    )


def gi_category(z: float, p: float) -> str:
    for alpha, conf in ((0.01, 99), (0.05, 95), (0.10, 90)):
        if p < alpha:
            return f"{'Hot' if z > 0 else 'Cold'} Spot ({conf}% Confidence)"
    return "Not Significant"


def compute_getis_ord_gi(df: pd.DataFrame, value_col: str = "weighted_score",
                         distance_threshold_km: float = 3.0) -> pd.DataFrame:
    """Local Getis-Ord Gi* with a fixed distance band (binary weights, self included).

    Adds gi_z_score, gi_p_value (two-sided normal) and gi_category.
    Distances are planar in degrees, with the km band converted at Chicago's latitude;
    fine for a ~40 km city, not for large extents.
    """
    result = df.copy()
    coords = np.array([(s.x, s.y) for s in result["SHAPE"]])
    deg_per_km = np.mean([1 / 111.0, 1 / (111.0 * np.cos(np.radians(41.88)))])
    threshold_deg = distance_threshold_km * deg_per_km

    x = result[value_col].fillna(0).to_numpy(dtype=float)
    n = len(x)
    s = np.std(x) or 1e-6

    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    w = (dists <= threshold_deg).astype(float)
    w_sum = w.sum(axis=1)
    numerator = w @ x - x.mean() * w_sum
    denominator = s * np.sqrt(np.clip((n * (w ** 2).sum(axis=1) - w_sum ** 2) / (n - 1), 0, None))
    z = np.divide(numerator, denominator, out=np.zeros(n), where=denominator > 0)
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))

    result["gi_z_score"] = np.round(z, 4)
    result["gi_p_value"] = np.round(p, 6)
    result["gi_category"] = [gi_category(zi, pi) for zi, pi in zip(z, p)]
    hot = result["gi_category"].str.startswith("Hot").sum()
    print(f"[Gi*] {hot}/{n} clusters are significant hot spots; z in [{z.min():.2f}, {z.max():.2f}]")
    return result


def generate_trauma_isochrones(gis: GIS, break_values: str = "5 8 12",
                               break_units: str = "Minutes") -> gpd.GeoDataFrame | None:
    """Drive-time rings around each trauma centre via ArcGIS Network Analyst (consumes credits).

    Returns polygons with a `ToBreak` column (minutes), or None if the service call fails.
    """
    facilities = pd.DataFrame.spatial.from_xy(
        CHICAGO_TRAUMA_CENTERS, x_column="longitude", y_column="latitude", sr=4326,
    ).spatial.to_featureset()
    try:
        out = network_analysis.generate_service_areas(
            facilities=facilities, break_values=break_values, break_units=break_units,
            travel_direction="Away From Facility", polygons_for_multiple_facilities="Overlapping",
            polygon_overlap_type="Rings", gis=gis,
        )
    except Exception as exc:
        print(f"[Network Analyst] Service areas failed, using fallback estimate: {exc}")
        return None
    if not (out and out.solve_succeeded):
        print("[Network Analyst] Solve did not succeed, using fallback estimate.")
        return None
    features = json.loads(out.service_areas.to_geojson)["features"]
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def _fallback_minutes(lat: float, lon: float) -> float:
    """Straight-line haversine to the nearest trauma centre -> estimated drive minutes."""
    tlat, tlon = np.radians(CHICAGO_TRAUMA_CENTERS[["latitude", "longitude"]].to_numpy()).T
    lat, lon = np.radians(lat), np.radians(lon)
    a = np.sin((tlat - lat) / 2) ** 2 + np.cos(lat) * np.cos(tlat) * np.sin((tlon - lon) / 2) ** 2
    km = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)).min()
    return km * ROUTE_CIRCUITY / FALLBACK_SPEED_KMH * 60


def evaluate_trauma_coverage(cluster_sedf: pd.DataFrame, isochrones: gpd.GeoDataFrame | None) -> pd.DataFrame:
    """Adds ems_reachability_status: isochrone point-in-polygon, else the fallback estimate."""
    result = cluster_sedf.copy()
    points = sedf_to_gdf(result).geometry

    if isochrones is not None and not isochrones.empty:
        rings = [(isochrones[isochrones["ToBreak"] == m].union_all(), label) for m, label in EMS_TIERS]
        status = [next((label for ring, label in rings if pt.within(ring)), EMS_CRITICAL) for pt in points]
    else:
        status = [
            next((label for m, label in EMS_TIERS if _fallback_minutes(pt.y, pt.x) <= m), EMS_CRITICAL)
            for pt in points
        ]
    result["ems_reachability_status"] = status

    gaps = sum(s.startswith(("Delayed", "Critical")) for s in status)
    print(f"[EMS] {gaps}/{len(status)} clusters are more than 8 min from a trauma centre")
    return result


def plot_hotspot_map(hotspots: gpd.GeoDataFrame, isochrones: gpd.GeoDataFrame | None) -> None:
    """Dark-theme static map of hot spots over isochrones, saved to both report locations."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 14), dpi=150)
    bg = "#0f172a"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    gpd.read_file(paths.COMMUNITY_AREAS).plot(ax=ax, color="#1e293b", edgecolor="#334155", linewidth=0.8, alpha=0.9)

    if isochrones is not None and not isochrones.empty:
        styles = [(12, "#38bdf8", 0.15, "12-min EMS Isochrone"),
                  (8, "#0284c7", 0.25, "8-min EMS (Golden Window)"),
                  (5, "#0369a1", 0.40, "5-min EMS Isochrone")]
        for minutes, color, alpha, label in styles:  # largest first so small rings stay visible
            ring = isochrones[isochrones["ToBreak"] == minutes]
            if not ring.empty:
                ring.plot(ax=ax, color=color, alpha=alpha, edgecolor=color, linewidth=1.2, label=label)

    is_hot = hotspots["gi_category"].str.contains("Hot Spot")
    hotspots[~is_hot].plot(ax=ax, color="#94a3b8", markersize=35, alpha=0.6, label="Not Significant / Low Risk")
    hotspots[is_hot].plot(ax=ax, color="#ef4444", markersize=90, edgecolor="#ffffff", linewidth=1.2,
                          alpha=0.95, label="Getis-Ord Gi* Hot Spot (p < 0.05)")

    ax.set_title("Chicago Crash Severity Hotspots & Level-1 Trauma Isochrones\n"
                 "ESRI ArcGIS Spatial Statistics (Getis-Ord Gi*) & Network Analyst",
                 color="#f8fafc", fontsize=14, pad=16, weight="bold")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.legend(loc="lower left", facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", fontsize=10)
    plt.tight_layout()

    for folder in (paths.REPORTS_DIR, paths.REPORT_FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / paths.ESRI_MAP_FIGURE, bbox_inches="tight", facecolor=bg)
    plt.close(fig)


def export_deliverables(hotspot_sedf: pd.DataFrame, isochrones: gpd.GeoDataFrame | None) -> None:
    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hotspot_sedf.drop(columns=["SHAPE"]).to_csv(paths.ESRI_HOTSPOTS_CSV, index=False)
    hotspots = sedf_to_gdf(hotspot_sedf)
    hotspots.to_file(paths.ESRI_HOTSPOTS_GEOJSON, driver="GeoJSON")
    if isochrones is not None:
        isochrones.to_file(paths.ESRI_ISOCHRONES, driver="GeoJSON")
    if paths.COMMUNITY_AREAS.exists():
        plot_hotspot_map(hotspots, isochrones)
    print(f"[ESRI] Deliverables saved to {paths.OUTPUT_DIR}")


def run_pipeline(api_key: str | None = None) -> pd.DataFrame:
    gis = init_arcgis_gis(api_key)
    clusters = load_cluster_sedf()
    score_col = "weighted_score" if "weighted_score" in clusters.columns else "num_points"
    hotspots = compute_getis_ord_gi(clusters, value_col=score_col, distance_threshold_km=3.0)
    isochrones = generate_trauma_isochrones(gis)
    result = evaluate_trauma_coverage(hotspots, isochrones)
    export_deliverables(result, isochrones)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESRI ArcGIS hotspot + trauma coverage pipeline")
    parser.add_argument("--test-auth", action="store_true", help="only check the ArcGIS connection")
    parser.add_argument("--api-key", default=None, help="ArcGIS API key (default: $ARCGIS_API_KEY)")
    args = parser.parse_args()
    init_arcgis_gis(args.api_key) if args.test_auth else run_pipeline(args.api_key)
