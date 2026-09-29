"""Step 3: per-crash severity and per-community-area risk.

Owns the severity formula shared by the pipeline (dbscan_hotspots, esri_pipeline via cluster_risk_table):
    (3*fatal + 2*incapacitating + 1*non_incapacitating + 0.5*reported_not_evident) / all injuries
Range 0..3; 0 when a crash/group has no injury records at all.
Cluster risk is written by dbscan_hotspots (same formula), not here.
"""

import numpy as np
import pandas as pd
import geopandas as gpd

from . import paths

# Order matters: matches weighted_severity()'s positional args.
INJURY_COLUMNS = [
    "INJURIES_FATAL",
    "INJURIES_INCAPACITATING",
    "INJURIES_NON_INCAPACITATING",
    "INJURIES_REPORTED_NOT_EVIDENT",
    "INJURIES_NO_INDICATION",
]


def weighted_severity(fatal, serious, moderate, minor, none):
    """Vectorised severity score; accepts scalars or aligned Series/arrays."""
    numer = 3 * fatal + 2 * serious + 1 * moderate + 0.5 * minor
    denom = fatal + serious + moderate + minor + none
    return np.divide(numer, denom, out=np.zeros(np.shape(denom), dtype=float), where=np.asarray(denom) > 0)


def coerce_injuries(df):
    """GeoJSON round-trips injury counts as strings; force numeric, NaN -> 0."""
    missing = [c for c in INJURY_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing injury columns: {missing}")
    df = df.copy()
    df[INJURY_COLUMNS] = df[INJURY_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def compute_crash_severity(gdf):
    gdf = coerce_injuries(gdf)
    gdf["weighted_severity"] = weighted_severity(*(gdf[c] for c in INJURY_COLUMNS))
    return gdf


def compute_community_risk(gdf):
    """One row per community area 1..77 (areas with no crashes get zeros)."""
    if "community_area_number" not in gdf.columns:
        raise KeyError("community_area_number column missing")

    counts = ["total_crashes", "fatal", "serious", "moderate", "minor", "none"]
    agg = gdf.groupby("community_area_number", dropna=False).agg(
        total_crashes=("geometry", "size"),
        fatal=("INJURIES_FATAL", "sum"),
        serious=("INJURIES_INCAPACITATING", "sum"),
        moderate=("INJURIES_NON_INCAPACITATING", "sum"),
        minor=("INJURIES_REPORTED_NOT_EVIDENT", "sum"),
        none=("INJURIES_NO_INDICATION", "sum"),
        mean_weighted_severity=("weighted_severity", "mean"),
    ).reset_index()

    agg["community_area_number"] = agg["community_area_number"].astype("Int64")
    all_areas = pd.DataFrame({"community_area_number": pd.array(range(1, 78), dtype="Int64")})
    agg = all_areas.merge(agg, on="community_area_number", how="left")
    agg[counts] = agg[counts].fillna(0)
    agg["mean_weighted_severity"] = agg["mean_weighted_severity"].fillna(0.0)

    agg["risk_index"] = weighted_severity(*(agg[c].astype(float) for c in counts[1:]))
    return agg


def main():
    gdf = gpd.read_file(paths.CRASHES_WITH_CLUSTERS)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    gdf = compute_crash_severity(gdf)
    community_risk = compute_community_risk(gdf)

    gdf.to_file(paths.CRASHES_WITH_SEVERITY, driver="GeoJSON")
    community_risk.to_csv(paths.COMMUNITY_RISK, index=False)

    print("Top 10 community areas by risk index:")
    print(community_risk.sort_values("risk_index", ascending=False).head(10))
    print(f"Community areas with zero crashes: {(community_risk['total_crashes'] == 0).sum()}")
    print(gdf["weighted_severity"].describe())


if __name__ == "__main__":
    main()
