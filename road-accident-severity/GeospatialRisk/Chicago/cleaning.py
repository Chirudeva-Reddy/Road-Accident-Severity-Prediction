"""Row filter for the raw Chicago crash export (used by prepare_geodata)."""

import pandas as pd

# Columns kept for the geospatial analysis; everything else is dropped.
REQUIRED_COLUMNS = [
    "POSTED_SPEED_LIMIT", "TRAFFIC_CONTROL_DEVICE", "DEVICE_CONDITION", "WEATHER_CONDITION",
    "LIGHTING_CONDITION", "FIRST_CRASH_TYPE", "TRAFFICWAY_TYPE", "ALIGNMENT",
    "ROADWAY_SURFACE_COND", "ROAD_DEFECT", "REPORT_TYPE", "CRASH_TYPE", "DAMAGE",
    "PRIM_CONTRIBUTORY_CAUSE", "SEC_CONTRIBUTORY_CAUSE", "NUM_UNITS", "MOST_SEVERE_INJURY",
    "INJURIES_TOTAL", "INJURIES_FATAL", "INJURIES_INCAPACITATING", "INJURIES_NON_INCAPACITATING",
    "INJURIES_REPORTED_NOT_EVIDENT", "INJURIES_NO_INDICATION", "INJURIES_UNKNOWN",
    "CRASH_HOUR", "CRASH_DAY_OF_WEEK", "CRASH_MONTH", "LATITUDE", "LONGITUDE",
]

# A row with all of these missing carries no road/environment signal.
ENV_COLUMNS = [
    "TRAFFIC_CONTROL_DEVICE", "DEVICE_CONDITION", "WEATHER_CONDITION", "LIGHTING_CONDITION",
    "TRAFFICWAY_TYPE", "ALIGNMENT", "ROADWAY_SURFACE_COND", "ROAD_DEFECT",
]

# All 7 counts incl. TOTAL/UNKNOWN (severity_pipeline.INJURY_COLUMNS is the 5 scored ones).
ALL_INJURY_COLUMNS = [c for c in REQUIRED_COLUMNS if c.startswith("INJURIES_")]


def clean_chicago_crash_dataset(df):
    """Keep rows with usable coordinates, some environment data, and not an injury-free drive-away.

    Coordinates and injury counts come back numeric (injury NaN -> 0).
    Latitude/longitude of exactly 0 are treated as missing.
    """
    df = df.copy()
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df[ALL_INJURY_COLUMNS] = df[ALL_INJURY_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)

    valid_coords = (
        df["LATITUDE"].between(-90, 90)
        & df["LONGITUDE"].between(-180, 180)
        & (df["LATITUDE"] != 0)
        & (df["LONGITUDE"] != 0)
    )  # between() is False for NaN
    has_env = df[ENV_COLUMNS].notna().any(axis=1)
    driveaway_no_injury = (df[ALL_INJURY_COLUMNS].sum(axis=1) == 0) & (df["CRASH_TYPE"] == "NO INJURY / DRIVE AWAY")

    keep = valid_coords & has_env & ~driveaway_no_injury
    return df.loc[keep, REQUIRED_COLUMNS].reset_index(drop=True)
