"""Step 1: raw crash export -> cleaned points tagged with their community area.

Writes paths.CRASHES_WITH_AREAS (EPSG:4326). Run: python -m GeospatialRisk.Chicago.prepare_geodata
"""

import zipfile

import geopandas as gpd
import pandas as pd

from . import paths
from .cleaning import clean_chicago_crash_dataset


def convert_chicago_to_geodataframe(df):
    """Expects cleaned rows (valid numeric LATITUDE/LONGITUDE)."""
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"]), crs="EPSG:4326")


def load_community_areas(path):
    """City boundary shapefile -> EPSG:4326 with numeric `community_area_number` and `community_area_name`.

    The city's exports have used both `area_num_1` and `area_numbe` for the area id.
    """
    gdf = gpd.read_file(path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    id_col = next((c for c in ("area_num_1", "area_numbe") if c in gdf.columns), None)
    if id_col:
        gdf = gdf.rename(columns={id_col: "community_area_number"})
        gdf["community_area_number"] = pd.to_numeric(gdf["community_area_number"], errors="coerce")
    if "community" in gdf.columns and "community_area_name" not in gdf.columns:
        gdf["community_area_name"] = gdf["community"]
    return gdf


def spatial_join_chicago(crash_gdf, community_gdf):
    """Left join: crashes outside every area keep NaN area columns."""
    merged = gpd.sjoin(crash_gdf, community_gdf, how="left", predicate="within",
                       lsuffix="_crash", rsuffix="_community")
    return merged.drop(columns=[c for c in merged.columns if c.startswith("index_")])


def main():
    # The portal download is CSV; the copy used for the report is an .xlsx saved under the .csv name.
    raw = paths.RAW_CRASHES
    df_raw = pd.read_excel(raw, engine="calamine") if zipfile.is_zipfile(raw) else pd.read_csv(raw, low_memory=False)
    gdf = convert_chicago_to_geodataframe(clean_chicago_crash_dataset(df_raw))
    print(f"Crashes after cleaning: {len(gdf)} (of {len(df_raw)}), bounds {gdf.total_bounds.round(4)}")

    joined = spatial_join_chicago(gdf, load_community_areas(paths.COMMUNITY_SHP))
    area = joined["community_area_number"]
    print(f"Assigned to a community area: {area.notna().sum()}, unassigned: {area.isna().sum()}, "
          f"areas hit: {area.nunique()}")

    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joined.to_file(paths.CRASHES_WITH_AREAS, driver="GeoJSON")
    print(f"Saved {paths.CRASHES_WITH_AREAS}")


if __name__ == "__main__":
    main()
