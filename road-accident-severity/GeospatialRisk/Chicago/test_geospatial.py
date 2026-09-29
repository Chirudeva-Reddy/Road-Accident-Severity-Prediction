"""Smoke checks for the shared scoring/stats logic. Run: python -m GeospatialRisk.Chicago.test_geospatial (or pytest)"""

import numpy as np
import pandas as pd
from shapely.geometry import Point

import geopandas as gpd

from .compare_hotspots import compare_hotspots
from .dbscan_hotspots import run_dbscan_haversine
from .severity_pipeline import weighted_severity


def test_weighted_severity():
    assert weighted_severity(1, 0, 0, 0, 0) == 3.0
    assert weighted_severity(0, 0, 0, 1, 1) == 0.25
    assert weighted_severity(0, 0, 0, 0, 0) == 0.0  # no injury records -> 0, not NaN
    np.testing.assert_array_equal(
        weighted_severity(pd.Series([1, 0]), pd.Series([0, 1]), 0, 0, pd.Series([1, 0])), [1.5, 2.0])


def test_dbscan_uses_true_distances():
    # 12 crashes 20 m apart north-south (all within eps=250 m) form one cluster; 12 more 400 m
    # apart stay noise. The old (lon, lat) bug shrank N-S distance ~25x and merged those too.
    m = 1 / 111_000  # degrees latitude per metre
    near = [(-87.65, 41.88 + i * 20 * m) for i in range(12)]
    far = [(-87.60, 41.88 + i * 400 * m) for i in range(12)]
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*(near + far))), crs="EPSG:4326")
    labels = run_dbscan_haversine(gdf)
    assert (labels[:12] == 0).all() and (labels[12:] == -1).all()


def test_getis_ord_flags_the_severe_cluster():
    from .esri_pipeline import compute_getis_ord_gi  # imports arcgis

    # 5 close points with high scores, 20 spread far away with low ones.
    pts = [Point(-87.6 + i * 0.001, 41.8) for i in range(5)] + [Point(-87.9 + i * 0.05, 42.0) for i in range(20)]
    df = pd.DataFrame({"SHAPE": pts, "weighted_score": [2.0] * 5 + [0.1] * 20})
    out = compute_getis_ord_gi(df)
    assert (out["gi_category"][:5] == "Hot Spot (99% Confidence)").all()
    assert not out["gi_category"][5:].str.startswith("Hot").any()


def test_compare_hotspots_quadrants():
    df = pd.DataFrame({
        "total_crashes": [100, 1], "gi_z_score": [3.0, 2.9],
        "gi_category": ["Hot Spot (99% Confidence)", "Not Significant"],
    })
    assert list(compare_hotspots(df)["methodology_insight"]) == [
        "Dual Priority: High Volume & High Severity", "Congestion Artefact: High Volume, Low Severity"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
