"""Step 5: DBSCAN volume ranking vs Getis-Ord Gi* severity ranking, per cluster.

DBSCAN finds where crashes are *dense*; Gi* finds where *severity* clusters beyond chance.
Each cluster lands in one quadrant of (top-20 by volume) x (Gi* hot spot).
Reads paths.ESRI_HOTSPOTS_CSV, writes paths.HOTSPOT_COMPARISON.
"""

import numpy as np
import pandas as pd

from . import paths

TOP_VOLUME = 20

EXPORT_COLUMNS = [
    "cluster_id", "centroid_lat", "centroid_lon", "total_crashes", "sum_fatal", "sum_incapacitating",
    "weighted_score", "volume_rank", "gi_z_score", "gi_p_value", "gi_category", "esri_severity_rank",
    "rank_divergence", "ems_reachability_status", "methodology_insight",
]


def compare_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    """Adds volume/severity ranks (1 = highest, ties averaged then truncated) and methodology_insight."""
    df = df.copy()
    df["volume_rank"] = df["total_crashes"].rank(ascending=False).astype(int)
    df["esri_severity_rank"] = df["gi_z_score"].rank(ascending=False).astype(int)
    df["rank_divergence"] = df["volume_rank"] - df["esri_severity_rank"]  # >0: more severe than busy

    hot = df["gi_category"].astype(str).str.contains("Hot Spot")
    busy = df["volume_rank"] <= TOP_VOLUME
    df["methodology_insight"] = np.select(
        [hot & busy, hot, busy],
        ["Dual Priority: High Volume & High Severity",
         "Hidden Hazard: Low Volume, High Severity (Isolated by Gi*)",
         "Congestion Artefact: High Volume, Low Severity"],
        default="Baseline / Low Risk",
    )
    cols = [c for c in EXPORT_COLUMNS if c in df.columns]
    return df[cols].sort_values("gi_z_score", ascending=False)


def main():
    if not paths.ESRI_HOTSPOTS_CSV.exists():
        raise FileNotFoundError(f"Missing {paths.ESRI_HOTSPOTS_CSV}. Run esri_pipeline first.")
    result = compare_hotspots(pd.read_csv(paths.ESRI_HOTSPOTS_CSV))
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(paths.HOTSPOT_COMPARISON, index=False)
    print(result["methodology_insight"].value_counts().to_string())
    print(f"Saved {paths.HOTSPOT_COMPARISON}")


if __name__ == "__main__":
    main()
