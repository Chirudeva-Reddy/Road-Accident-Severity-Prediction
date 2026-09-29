"""Every file the Chicago geospatial pipeline reads or writes, in run order."""

from pathlib import Path

CHICAGO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CHICAGO_DIR / "output"
REPORTS_DIR = CHICAGO_DIR.parents[1] / "reports"          # road-accident-severity/reports
REPORT_FIGURES_DIR = CHICAGO_DIR.parents[2] / "report" / "figures"  # LaTeX report

# Inputs (no script here produces them; COMMUNITY_AREAS is a checked-in GeoJSON export of the boundaries)
RAW_CRASHES = CHICAGO_DIR / "dataset" / "Traffic_Crashes_-_Crashes.csv"  # CSV or .xlsx content, see prepare_geodata
COMMUNITY_SHP = CHICAGO_DIR / "data" / "Boundaries" / "geo_export_273d6492-11b0-415d-8d9d-bf83ed5c6833.shp"
COMMUNITY_AREAS = OUTPUT_DIR / "chicago_community_areas.geojson"

# prepare_geodata -> dbscan_hotspots -> severity_pipeline
CRASHES_WITH_AREAS = OUTPUT_DIR / "chicago_crashes_with_areas.geojson"
CRASHES_WITH_CLUSTERS = OUTPUT_DIR / "chicago_crashes_with_clusters.geojson"
CLUSTER_CENTROIDS = OUTPUT_DIR / "chicago_cluster_centroids.geojson"
CLUSTER_RISK = OUTPUT_DIR / "cluster_risk_table.csv"
CRASHES_WITH_SEVERITY = OUTPUT_DIR / "chicago_crashes_with_severity.geojson"
COMMUNITY_RISK = OUTPUT_DIR / "community_area_risk.csv"

# esri_pipeline -> compare_hotspots / build_live_map
ESRI_HOTSPOTS_CSV = OUTPUT_DIR / "esri_chicago_hotspots_getis_ord.csv"
ESRI_HOTSPOTS_GEOJSON = OUTPUT_DIR / "esri_chicago_hotspots_getis_ord.geojson"
ESRI_ISOCHRONES = OUTPUT_DIR / "esri_trauma_isochrones.geojson"
ESRI_MAP_FIGURE = "esri_hotspots_and_trauma_isochrones.png"  # saved to REPORTS_DIR and REPORT_FIGURES_DIR
HOTSPOT_COMPARISON = REPORTS_DIR / "hotspot_comparison.csv"
