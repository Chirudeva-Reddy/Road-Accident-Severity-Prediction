"""
Chicago Traffic Crash Intelligence Platform - Interactive Filters Module

This module provides comprehensive filtering capabilities for the Chicago crash dataset,
including environmental, temporal, crash type, and severity filters with dynamic layer toggles.
Designed for performance with 740,000+ crash points using efficient caching and indexing.

Author: Chirudeva Reddy
Version: 1.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import dash
from dash import Dash, Input, Output, State, dcc, html, callback_context
import dash_bootstrap_components as dbc
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import Point
from datetime import datetime

# Esri Canvas Dark Gray: base tiles under the data, labels on top. Keyless public tiles; attribution required.
_ESRI_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
ESRI_DARK_BASEMAP = dict(style="white-bg", layers=[
    dict(sourcetype="raster", source=[_ESRI_TILES.format("Base")], below="traces",
         sourceattribution="Esri, HERE, Garmin, &copy; OpenStreetMap contributors"),
    dict(sourcetype="raster", source=[_ESRI_TILES.format("Reference")]),
])

# Project Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Chicago" / "output"

# =============================================================================
# GLOBAL MAPPINGS AND CONSTANTS
# =============================================================================

# Environmental condition mappings based on ACTUAL data values from debug output
WEATHER_MAPPING = {
    "CLEAR": "Clear Weather",
    "CLOUDY/OVERCAST": "Cloudy", 
    "RAIN": "Rain",
    "SNOW": "Snow",
    "BLOWING SNOW": "Blowing Snow",
    "BLOWING SAND, SOIL, DIRT": "Blowing Sand/Dirt",
    "FOG/SMOKE/HAZE": "Fog/Smoke/Haze",
    "FREEZING RAIN/DRIZZLE": "Freezing Rain",
    "SLEET/HAIL": "Sleet/Hail",
    "SEVERE CROSS WIND GATE": "Severe Wind",
    "OTHER": "Other",
    "UNKNOWN": "Unknown"
}

LIGHTING_MAPPING = {
    "DAYLIGHT": "Daylight",
    "DAWN": "Dawn", 
    "DUSK": "Dusk",
    "DARKNESS": "Darkness - No Lights",
    "DARKNESS, LIGHTED ROAD": "Darkness - Street Lights",
    "UNKNOWN": "Unknown"
}

SURFACE_MAPPING = {
    "DRY": "Dry Surface",
    "WET": "Wet Surface", 
    "SNOW OR SLUSH": "Snow/Slush",
    "ICE": "Ice",
    "SAND, MUD, DIRT": "Sand/Mud/Dirt",
    "OTHER": "Other",
    "UNKNOWN": "Unknown"
}

# Crash type mappings based on ACTUAL data values
FIRST_CRASH_TYPE_MAPPING = {
    "REAR END": "Rear End",
    "ANGLE": "Angle",
    "SIDESWIPE SAME DIRECTION": "Sideswipe - Same Direction",
    "SIDESWIPE OPPOSITE DIRECTION": "Sideswipe - Opposite Direction",
    "HEAD ON": "Head On",
    "TURNING": "Turning",
    "PEDESTRIAN": "Pedestrian",
    "PEDALCYCLIST": "Cyclist",
    "FIXED OBJECT": "Fixed Object",
    "PARKED MOTOR VEHICLE": "Parked Vehicle",
    "OVERTURNED": "Overturned",
    "OTHER OBJECT": "Other Object",
    "OTHER NONCOLLISION": "Non-Collision",
    "ANIMAL": "Animal",
    "REAR TO FRONT": "Rear to Front",
    "REAR TO SIDE": "Rear to Side", 
    "REAR TO REAR": "Rear to Rear",
    "TRAIN": "Train"
}

DAMAGE_MAPPING = {
    1: "$1-500", 2: "$501-1,500", 3: "$1,501-2,500", 4: "$2,501-5,000",
    5: "$5,001-10,000", 6: "$10,001+", 7: "Unknown"
}

# Severity categories for filtering
SEVERITY_CATEGORIES = {
    "Fatal": lambda df: df['fatal'] > 0,
    "Injury": lambda df: (df['serious'] > 0) | (df['moderate'] > 0),
    "Property Damage": lambda df: (df['minor'] > 0) | (df['none'] > 0)
}

# Risk color scheme: Critical (High Risk) to Low Risk
RISK_COLORS = {
    "Critical": "#DC143C",      # Crimson - Highest risk
    "High": "#FF6347",          # Tomato - High risk  
    "Medium": "#FFD700",        # Gold - Medium risk
    "Low": "#90EE90",           # Light Green - Low risk
    "No Data": "#D3D3D3"        # Light Gray - No data
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_geospatial_data() -> Dict[str, Any]:
    """Load all geospatial data with optimized indexing for performance."""
    print("Loading geospatial data...")
    
    # Load community areas
    community_areas = gpd.read_file(DATA_DIR / "chicago_community_areas.geojson")
    community_areas = community_areas.to_crs(epsg=4326)
    
    # Load crashes with areas
    crashes = gpd.read_file(DATA_DIR / "chicago_crashes_with_areas.geojson")
    crashes = crashes.to_crs(epsg=4326)
    
    # Load clusters
    clusters = gpd.read_file(DATA_DIR / "chicago_cluster_centroids.geojson")
    clusters = clusters.to_crs(epsg=4326)
    
    # Load risk tables
    community_risk = pd.read_csv(DATA_DIR / "community_area_risk.csv")
    cluster_risk = pd.read_csv(DATA_DIR / "cluster_risk_table.csv")
    
    # Optimize data types and create indexes for performance
    # The crash data already has temporal columns: CRASH_HOUR, CRASH_DAY_OF_WEEK, CRASH_MONTH
    crashes['crash_hour'] = pd.to_numeric(crashes['CRASH_HOUR'], errors='coerce').fillna(0)
    crashes['crash_day_of_week'] = pd.to_numeric(crashes['CRASH_DAY_OF_WEEK'], errors='coerce').fillna(0)
    crashes['crash_month'] = pd.to_numeric(crashes['CRASH_MONTH'], errors='coerce').fillna(1)
    
    # Create year column (assuming recent data, use 2023 as default)
    crashes['crash_year'] = 2023
    
    # Map injury columns to expected names
    crashes['fatal'] = pd.to_numeric(crashes['INJURIES_FATAL'], errors='coerce').fillna(0)
    crashes['serious'] = pd.to_numeric(crashes['INJURIES_INCAPACITATING'], errors='coerce').fillna(0)
    crashes['moderate'] = pd.to_numeric(crashes['INJURIES_NON_INCAPACITATING'], errors='coerce').fillna(0)
    crashes['minor'] = pd.to_numeric(crashes['INJURIES_REPORTED_NOT_EVIDENT'], errors='coerce').fillna(0)
    crashes['none'] = pd.to_numeric(crashes['INJURIES_NO_INDICATION'], errors='coerce').fillna(0)
    
    # Create categorical indexes for faster filtering
    for col in ['WEATHER_CONDITION', 'LIGHTING_CONDITION', 'ROADWAY_SURFACE_COND',
                'TRAFFICWAY_TYPE', 'ALIGNMENT', 'ROAD_DEFECT', 'FIRST_CRASH_TYPE']:
        if col in crashes.columns:
            crashes[col] = crashes[col].astype('category')
    
    # Merge community areas with risk data
    geojson_key = None
    for col in community_areas.columns:
        if 'area' in col.lower() and 'num' in col.lower():
            geojson_key = col
            break
    
    if geojson_key and 'community_area_number' in community_risk.columns:
        community_areas = community_areas.merge(
            community_risk, 
            left_on=geojson_key, 
            right_on='community_area_number', 
            how='left'
        )
    
    # Add cluster coordinates
    clusters['centroid_lon'] = clusters.geometry.x
    clusters['centroid_lat'] = clusters.geometry.y
    
    # Convert all GeoDataFrames to JSON-serializable structures
    community_areas_json = json.loads(community_areas.to_json())
    crashes_json = json.loads(crashes.to_json())
    clusters_json = json.loads(clusters.drop(columns="geometry").to_json())
    
    print(f"Loaded {len(crashes):,} crashes, {len(community_areas)} areas, {len(clusters)} clusters")
    
    return {
        'crashes': crashes_json,
        'community_areas': community_areas_json,
        'clusters': clusters_json,
        'community_risk': community_risk.to_dict('records'),
        'cluster_risk': cluster_risk.to_dict('records'),
        'geojson_key': geojson_key,
        'total_crashes': len(crashes)
    }

def create_data_store(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert data to JSON-serializable format for dcc.Store."""
    # All data is already JSON-serializable from load_geospatial_data
    return data_dict

# =============================================================================
# FILTER LOGIC FUNCTIONS
# =============================================================================

def apply_filters(crashes_df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply all filters to crash data using optimized pandas queries."""
    
    # Start with full dataset
    filtered_df = crashes_df.copy()
    
    # Temporal filters
    if filters.get('hours') and filters['hours'] != list(range(24)):
        filtered_df = filtered_df[filtered_df['crash_hour'].isin(filters['hours'])]
    
    if filters.get('days') and filters['days'] != list(range(7)):
        filtered_df = filtered_df[filtered_df['crash_day_of_week'].isin(filters['days'])]
    
    if filters.get('months') and filters['months'] != list(range(1, 13)):
        filtered_df = filtered_df[filtered_df['crash_month'].isin(filters['months'])]
    
    if filters.get('years') and filters['years'] != sorted(filtered_df['crash_year'].unique()):
        filtered_df = filtered_df[filtered_df['crash_year'].isin(filters['years'])]
    
    # Environmental filters - use actual column names
    if filters.get('weather') and filters['weather'] != list(WEATHER_MAPPING.keys()):
        filtered_df = filtered_df[filtered_df['WEATHER_CONDITION'].isin(filters['weather'])]
    
    if filters.get('lighting') and filters['lighting'] != list(LIGHTING_MAPPING.keys()):
        filtered_df = filtered_df[filtered_df['LIGHTING_CONDITION'].isin(filters['lighting'])]
    
    if filters.get('surface') and filters['surface'] != list(SURFACE_MAPPING.keys()):
        filtered_df = filtered_df[filtered_df['ROADWAY_SURFACE_COND'].isin(filters['surface'])]
    
    # Crash type filters
    if filters.get('crash_types') and filters['crash_types'] != list(FIRST_CRASH_TYPE_MAPPING.keys()):
        filtered_df = filtered_df[filtered_df['FIRST_CRASH_TYPE'].isin(filters['crash_types'])]
    
    # Severity filters
    severity_filters = filters.get('severity', [])
    if severity_filters and severity_filters != list(SEVERITY_CATEGORIES.keys()):
        severity_mask = pd.Series(False, index=filtered_df.index)
        for severity in severity_filters:
            if severity in SEVERITY_CATEGORIES:
                severity_mask |= SEVERITY_CATEGORIES[severity](filtered_df)
        filtered_df = filtered_df[severity_mask]
    
    # Risk score range filter
    if filters.get('risk_range'):
        min_risk, max_risk = filters['risk_range']
        if 'risk_score' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['risk_score'] >= min_risk) & 
                (filtered_df['risk_score'] <= max_risk)
            ]
    
    return filtered_df

def compute_filtered_statistics(filtered_df: pd.DataFrame, total_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comparative statistics for filtered vs total dataset."""
    
    stats = {
        'filtered_count': len(filtered_df),
        'total_count': len(total_df),
        'percentage': (len(filtered_df) / len(total_df) * 100) if len(total_df) > 0 else 0,
        'fatal_filtered': filtered_df['fatal'].sum() if 'fatal' in filtered_df.columns else 0,
        'fatal_total': total_df['fatal'].sum() if 'fatal' in total_df.columns else 0,
        'injury_filtered': (filtered_df[['serious', 'moderate']].sum().sum()) if all(col in filtered_df.columns for col in ['serious', 'moderate']) else 0,
        'injury_total': (total_df[['serious', 'moderate']].sum().sum()) if all(col in total_df.columns for col in ['serious', 'moderate']) else 0,
    }
    
    # Compute mean weighted severity (risk score) - use actual column name
    if 'mean_weighted_severity' in filtered_df.columns:
        stats['mean_risk_filtered'] = filtered_df['mean_weighted_severity'].mean()
        stats['mean_risk_total'] = total_df['mean_weighted_severity'].mean()
    else:
        # Fallback: compute from injury columns
        stats['mean_risk_filtered'] = (
            3 * stats['fatal_filtered'] + 
            2 * filtered_df['serious'].sum() + 
            1 * filtered_df['moderate'].sum()
        ) / max(len(filtered_df), 1)
        
        stats['mean_risk_total'] = (
            3 * stats['fatal_total'] + 
            2 * total_df['serious'].sum() + 
            1 * total_df['moderate'].sum()
        ) / max(len(total_df), 1)
    
    # Compute risk increase/decrease
    if stats['mean_risk_total'] > 0:
        stats['risk_change'] = ((stats['mean_risk_filtered'] - stats['mean_risk_total']) / stats['mean_risk_total']) * 100
    else:
        stats['risk_change'] = 0
    
    return stats

# =============================================================================
# FIGURE BUILDING FUNCTIONS
# =============================================================================

def create_interactive_choropleth(community_areas_df: pd.DataFrame, 
                                 filters: Dict[str, Any],
                                 show_choropleth: bool = True) -> go.Figure:
    """Create dynamic choropleth map with filtered data using proper risk-based coloring."""
    
    fig = go.Figure()
    
    if show_choropleth and len(community_areas_df) > 0:
        # Calculate risk quantiles for proper coloring
        if 'mean_weighted_severity' in community_areas_df.columns:
            risk_scores = community_areas_df['mean_weighted_severity'].dropna()
            if len(risk_scores) > 0:
                critical_threshold = risk_scores.quantile(0.9)
                high_threshold = risk_scores.quantile(0.75)
                medium_threshold = risk_scores.quantile(0.5)
            else:
                critical_threshold = high_threshold = medium_threshold = 0
        else:
            critical_threshold = high_threshold = medium_threshold = 0
        
        # Add community areas with dynamic risk-based coloring
        for _, area in community_areas_df.iterrows():
            if hasattr(area, 'geometry') and area.geometry.geom_type == 'Polygon':
                coords = list(area.geometry.exterior.coords)
                lons, lats = zip(*coords)
                
                # Risk-based color assignment (Red = High Risk, Green = Low Risk)
                risk_score = area.get('mean_weighted_severity', 0)
                if pd.isna(risk_score) or risk_score == 0:
                    color = RISK_COLORS["No Data"]  # Gray - No data
                elif risk_score >= critical_threshold:
                    color = RISK_COLORS["Critical"]  # Crimson - Highest risk
                elif risk_score >= high_threshold:
                    color = RISK_COLORS["High"]      # Tomato - High risk  
                elif risk_score >= medium_threshold:
                    color = RISK_COLORS["Medium"]    # Gold - Medium risk
                else:
                    color = RISK_COLORS["Low"]       # Light Green - Low risk
                
                fig.add_trace(go.Scattermapbox(
                    lat=lats, lon=lons,
                    mode='lines',
                    line=dict(width=1, color='white'),
                    fill='toself',
                    fillcolor=color,
                    hovertemplate=(
                        f"<b>{area.get('community', 'Unknown Area')}</b><br>"
                        f"Risk Score: {risk_score:.3f}<br>"
                        f"Risk Level: {'Critical' if risk_score >= critical_threshold else 'High' if risk_score >= high_threshold else 'Medium' if risk_score >= medium_threshold else 'Low' if risk_score > 0 else 'No Data'}<br>"
                        f"Total Crashes: {area.get('total_crashes', 0):,}<br>"
                        f"Fatalities: {area.get('fatal', 0)}<extra></extra>"
                    ),
                    name='Community Areas',
                    showlegend=False
                ))
    
    return fig

def create_crash_points_layer(crashes_df: pd.DataFrame, 
                             max_points: int = 2000) -> go.Figure:
    """Create crash points layer with performance optimization."""
    
    fig = go.Figure()
    
    if len(crashes_df) > 0:
        # Aggressive sampling for performance and network efficiency
        if len(crashes_df) > max_points:
            # Stratified sampling to preserve severity distribution
            fatal_crashes = crashes_df[crashes_df['fatal'] > 0]
            injury_crashes = crashes_df[(crashes_df['serious'] > 0) | (crashes_df['moderate'] > 0)]
            other_crashes = crashes_df[(crashes_df['fatal'] == 0) & (crashes_df['serious'] == 0) & (crashes_df['moderate'] == 0)]
            
            # Sample proportionally
            sample_size = max_points - min(len(fatal_crashes), 200)  # Reserve space for fatalities
            injury_sample = injury_crashes.sample(min(len(injury_crashes), sample_size//2), random_state=42)
            other_sample = other_crashes.sample(min(len(other_crashes), sample_size//2), random_state=42)
            
            display_crashes = pd.concat([fatal_crashes, injury_sample, other_sample])
        else:
            display_crashes = crashes_df
        
        # Extract coordinates as lists (no Shapely objects)
        lats = display_crashes.geometry.y.tolist()
        lons = display_crashes.geometry.x.tolist()
        
        # Color by severity - use dark blue for better visibility
        colors = []
        sizes = []
        for _, crash in display_crashes.iterrows():
            if crash.get('fatal', 0) > 0:
                colors.append('#000080')  # Dark blue - fatal crashes
                sizes.append(8)
            elif crash.get('serious', 0) > 0 or crash.get('moderate', 0) > 0:
                colors.append('#0000CD')  # Medium blue - injury crashes
                sizes.append(6)
            else:
                colors.append('#4169E1')  # Royal blue - property damage
                sizes.append(4)
        
        # Create date string from components with error handling
        date_strings = []
        severity_strings = []
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        for _, crash in display_crashes.iterrows():
            try:
                month_idx = int(crash['crash_month']) - 1
                day_idx = int(crash['crash_day_of_week'])
                
                if 0 <= month_idx < len(month_names) and 0 <= day_idx < len(day_names):
                    month_name = month_names[month_idx]
                    day_name = day_names[day_idx]
                    date_strings.append(f"{month_name} {day_name} {int(crash['crash_hour']):02d}:00")
                else:
                    date_strings.append("Invalid Date")
            except (ValueError, IndexError, KeyError):
                date_strings.append("Invalid Date")
            
            if crash.get('fatal', 0) > 0:
                severity_strings.append('Fatal')
            elif crash.get('serious', 0) > 0 or crash.get('moderate', 0) > 0:
                severity_strings.append('Injury')
            else:
                severity_strings.append('Property')
        
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.7
            ),
            hovertemplate=(
                "Crash ID: %{text}<br>"
                "Date: %{customdata[0]}<br>"
                "Severity: %{customdata[1]}<extra></extra>"
            ),
            text=[f"Crash {i}" for i in range(len(display_crashes))],
            customdata=[list(pair) for pair in zip(date_strings, severity_strings)],
            name='Crash Points',
            showlegend=True
        ))
    
    return fig

def create_cluster_heatmap(clusters_df: pd.DataFrame, 
                          cluster_risk_df: pd.DataFrame) -> go.Figure:
    """Create cluster heatmap overlay."""
    
    fig = go.Figure()
    
    if len(clusters_df) > 0 and len(cluster_risk_df) > 0:
        # Merge clusters with risk data (clusters_df already has coordinates)
        merged_clusters = clusters_df.merge(cluster_risk_df, on='cluster_id', how='left')
        
        fig.add_trace(go.Scattermapbox(
            lat=merged_clusters['centroid_lat'],
            lon=merged_clusters['centroid_lon'],
            mode='markers',
            marker=dict(
                size=np.sqrt(merged_clusters['total_crashes']) * 2,
                color=merged_clusters['weighted_score'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Cluster Risk Score"),
                opacity=0.8
            ),
            hovertemplate=(
                "<b>Cluster %{text}</b><br>"
                "Total Crashes: %{marker.size:.0f}<br>"
                "Risk Score: %{marker.color:.2f}<extra></extra>"
            ),
            text=merged_clusters['cluster_id'],
            name='Risk Clusters',
            showlegend=True
        ))
    
    return fig

# =============================================================================
# DASHBOARD LAYOUT COMPONENTS
# =============================================================================

def create_filter_panel() -> dbc.Card:
    """Create comprehensive filter control panel."""
    
    return dbc.Card([
        dbc.CardHeader(html.H5("Filters & Controls", className="mb-0")),
        dbc.CardBody([
            # Layer Controls
            html.H6("Map Layers", className="text-primary"),
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id="layer-toggles",
                        options=[
                            {"label": "Community Areas", "value": "choropleth"},
                            {"label": "Risk Clusters", "value": "clusters"},
                            {"label": "Crash Points", "value": "crashes"}
                        ],
                        value=["choropleth", "clusters"],
                        inline=True
                    )
                ], width=12)
            ], className="mb-3"),
            
            html.Hr(),
            
            # Temporal Filters
            html.H6("Temporal Filters", className="text-primary"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Time of Day"),
                    dcc.RangeSlider(
                        id="hour-range",
                        min=0, max=23,
                        step=1,
                        value=[0, 23],
                        marks={i: f"{i:02d}:00" for i in range(0, 24, 3)},
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], width=12),
                dbc.Col([
                    dbc.Label("Day of Week"),
                    dcc.Dropdown(
                        id="day-filter",
                        options=[
                            {"label": "Monday", "value": 0},
                            {"label": "Tuesday", "value": 1},
                            {"label": "Wednesday", "value": 2},
                            {"label": "Thursday", "value": 3},
                            {"label": "Friday", "value": 4},
                            {"label": "Saturday", "value": 5},
                            {"label": "Sunday", "value": 6}
                        ],
                        value=list(range(7)),
                        multi=True,
                        placeholder="Select days..."
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Month"),
                    dcc.Dropdown(
                        id="month-filter",
                        options=[
                            {"label": datetime(2024, i, 1).strftime("%B"), "value": i}
                            for i in range(1, 13)
                        ],
                        value=list(range(1, 13)),
                        multi=True,
                        placeholder="Select months..."
                    )
                ], width=6)
            ], className="mb-3"),
            
            html.Hr(),
            
            # Environmental Filters
            html.H6("Environmental Conditions", className="text-primary"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Weather"),
                    dcc.Dropdown(
                        id="weather-filter",
                        options=[
                            {"label": WEATHER_MAPPING[k], "value": k}
                            for k in sorted(WEATHER_MAPPING.keys())
                        ],
                        value=list(WEATHER_MAPPING.keys()),
                        multi=True,
                        placeholder="Select weather conditions..."
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Lighting"),
                    dcc.Dropdown(
                        id="lighting-filter",
                        options=[
                            {"label": LIGHTING_MAPPING[k], "value": k}
                            for k in sorted(LIGHTING_MAPPING.keys())
                        ],
                        value=list(LIGHTING_MAPPING.keys()),
                        multi=True,
                        placeholder="Select lighting conditions..."
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Road Surface"),
                    dcc.Dropdown(
                        id="surface-filter",
                        options=[
                            {"label": SURFACE_MAPPING[k], "value": k}
                            for k in sorted(SURFACE_MAPPING.keys())
                        ],
                        value=list(SURFACE_MAPPING.keys()),
                        multi=True,
                        placeholder="Select surface conditions..."
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Crash Type"),
                    dcc.Dropdown(
                        id="crash-type-filter",
                        options=[
                            {"label": FIRST_CRASH_TYPE_MAPPING[k], "value": k}
                            for k in sorted(FIRST_CRASH_TYPE_MAPPING.keys())
                        ],
                        value=list(FIRST_CRASH_TYPE_MAPPING.keys()),
                        multi=True,
                        placeholder="Select crash types..."
                    )
                ], width=6)
            ], className="mb-3"),
            
            html.Hr(),
            
            # Severity Filters
            html.H6("Severity Filters", className="text-primary"),
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id="severity-filter",
                        options=[
                            {"label": "Fatal Crashes", "value": "Fatal"},
                            {"label": "Injury Crashes", "value": "Injury"},
                            {"label": "Property Damage", "value": "Property Damage"}
                        ],
                        value=["Fatal", "Injury", "Property Damage"],
                        inline=True
                    )
                ], width=12)
            ], className="mb-3"),
            
            # Action Buttons
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Apply Filters", id="apply-filters", color="primary", className="me-2"),
                    dbc.Button("Reset Filters", id="reset-filters", color="secondary", outline=True)
                ], width=12)
            ])
        ])
    ], className="mb-4")

def create_statistics_panel() -> dbc.Card:
    """Create dynamic statistics panel."""
    
    return dbc.Card([
        dbc.CardHeader(html.H5("Filter Statistics", className="mb-0")),
        dbc.CardBody([
            html.Div(id="filter-stats-content", children=[
                html.P("Apply filters to see statistics...", className="text-muted")
            ])
        ])
    ], className="mb-4")


# MAIN DASHBOARD

def create_interactive_dashboard() -> Dash:
    """Create the main interactive dashboard."""
    
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Load and cache data
    data_dict = load_geospatial_data()
    store_data = create_data_store(data_dict)
    
    # Layout
    app.layout = dbc.Container([
        # Data store
        dcc.Store(id='data-store', data=store_data),
        
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Chicago Traffic Crash Intelligence Platform", 
                       className="text-center mb-4",
                       style={'color': '#2c3e50', 'fontWeight': 'bold'}),
                html.H5("Interactive Geospatial Analytics & Filtering System", 
                       className="text-center mb-4 text-muted"),
                html.Hr()
            ])
        ]),
        
        # Main Content
        dbc.Row([
            # Left Panel - Filters
            dbc.Col([
                create_filter_panel(),
                create_statistics_panel()
            ], width=4),
            
            # Right Panel - Map
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Interactive Map", className="mb-0")),
                    dbc.CardBody([
                        dcc.Graph(
                            id="main-map",
                            style={'height': '800px'},
                            config={'displayModeBar': True, 'scrollZoom': True}
                        )
                    ])
                ])
            ], width=8)
        ])
    ], fluid=True)
    
    # CALLBACKS
    
    @app.callback(
        [Output('main-map', 'figure'),
         Output('filter-stats-content', 'children')],
        [Input('apply-filters', 'n_clicks'),
         Input('reset-filters', 'n_clicks')],
        [State('data-store', 'data'),
         State('layer-toggles', 'value'),
         State('hour-range', 'value'),
         State('day-filter', 'value'),
         State('month-filter', 'value'),
         State('weather-filter', 'value'),
         State('lighting-filter', 'value'),
         State('surface-filter', 'value'),
         State('crash-type-filter', 'value'),
         State('severity-filter', 'value')]
    )
    def update_dashboard(apply_clicks, reset_clicks, store_data, 
                        layers, hour_range, days, months, weather, lighting, 
                        surface, crash_types, severity):
        
        try:
            try:
                ctx = callback_context
                trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if (ctx and getattr(ctx, 'triggered', None)) else None
            except Exception:
                trigger_id = None
            
            # Reset filters if reset button clicked
            if trigger_id == 'reset-filters':
                hour_range = [0, 23]
                days = list(range(7))
                months = list(range(1, 13))
                weather = list(WEATHER_MAPPING.keys())
                lighting = list(LIGHTING_MAPPING.keys())
                surface = list(SURFACE_MAPPING.keys())
                crash_types = list(FIRST_CRASH_TYPE_MAPPING.keys())
                severity = ["Fatal", "Injury", "Property Damage"]
            
            # Convert stored data back to DataFrames (reconstruct GeoDataFrames from JSON)
            crashes_df = gpd.GeoDataFrame.from_features(store_data['crashes']['features'], crs='EPSG:4326')
            for col in ['fatal', 'serious', 'moderate', 'minor', 'none', 'crash_hour', 'crash_day_of_week', 'crash_month', 'crash_year']:
                if col in crashes_df.columns:
                    crashes_df[col] = pd.to_numeric(crashes_df[col], errors='coerce').fillna(0)

            community_areas_df = gpd.GeoDataFrame.from_features(store_data['community_areas']['features'], crs='EPSG:4326')
            for col in ['total_crashes', 'fatal', 'serious', 'moderate', 'minor', 'none', 'mean_weighted_severity', 'risk_index']:
                if col in community_areas_df.columns:
                    community_areas_df[col] = pd.to_numeric(community_areas_df[col], errors='coerce').fillna(0)

            clusters_df = pd.DataFrame(store_data['clusters'])  # Already processed, no geometry
            for col in ['centroid_lat', 'centroid_lon', 'total_crashes', 'weighted_score']:
                if col in clusters_df.columns:
                    clusters_df[col] = pd.to_numeric(clusters_df[col], errors='coerce').fillna(0)
            
            # Build filter dictionary
            filters = {
                'hours': list(range(hour_range[0], hour_range[1] + 1)) if hour_range else list(range(24)),
                'days': days or [],
                'months': months or [],
                'weather': weather or [],
                'lighting': lighting or [],
                'surface': surface or [],
                'crash_types': crash_types or [],
                'severity': severity or []
            }
            
            # Apply filters
            filtered_crashes = apply_filters(crashes_df, filters)
            
            # Compute statistics
            stats = compute_filtered_statistics(filtered_crashes, crashes_df)
            
            # Create statistics content
            stats_content = [
                html.H6("Filter Results", className="text-primary"),
                html.P(f"Showing {stats['filtered_count']:,} of {stats['total_count']:,} crashes "
                      f"({stats['percentage']:.1f}%)", className="fw-bold"),
                html.Hr(),
                html.P(f"Fatalities: {stats['fatal_filtered']} of {stats['fatal_total']} "
                      f"({(stats['fatal_filtered']/stats['fatal_total']*100 if stats['fatal_total'] > 0 else 0):.1f}%)"),
                html.P(f"Injuries: {stats['injury_filtered']:,} of {stats['injury_total']:,} "
                      f"({(stats['injury_filtered']/stats['injury_total']*100 if stats['injury_total'] > 0 else 0):.1f}%)"),
                html.P(f"Mean Risk Score: {stats['mean_risk_filtered']:.2f} (vs {stats['mean_risk_total']:.2f} baseline)"),
                html.P(f"Risk Change: {stats['risk_change']:+.1f}%", 
                      className="fw-bold text-success" if stats['risk_change'] < 0 else "text-danger")
            ]
            
            # Build map figure
            fig = go.Figure()
            
            # Add base map layers based on toggles
            # Only add layers that are explicitly selected
            if layers and 'choropleth' in layers:
                try:
                    choropleth_fig = create_interactive_choropleth(community_areas_df, filters, True)
                    for trace in choropleth_fig.data:
                        fig.add_trace(trace)
                except Exception as e:
                    print(f"Error creating choropleth: {e}")
            
            if layers and 'clusters' in layers:
                try:
                    cluster_fig = create_cluster_heatmap(clusters_df, pd.DataFrame(store_data['cluster_risk']))
                    for trace in cluster_fig.data:
                        fig.add_trace(trace)
                except Exception as e:
                    print(f"Error creating clusters: {e}")
            
            if layers and 'crashes' in layers:
                try:
                    crash_fig = create_crash_points_layer(filtered_crashes, max_points=2000)  # Reduced from 5000
                    for trace in crash_fig.data:
                        fig.add_trace(trace)
                except Exception as e:
                    print(f"Error creating crash points: {e}")
            
            # Update layout
            fig.update_layout(
                mapbox=dict(
                    **ESRI_DARK_BASEMAP,
                    center=dict(lat=41.8781, lon=-87.6298),
                    zoom=10
                ),
                height=800,
                margin=dict(l=20, r=20, t=40, b=20),
                title=dict(
                    text=f"Chicago Crash Analysis - {stats['filtered_count']:,} Filtered Results",
                    x=0.5,
                    font=dict(size=18, color="#2c3e50")
                ),
                legend=dict(
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=0.02,
                    bgcolor="rgba(255,255,255,0.9)"
                ),
                showlegend=True
            )
            
            return fig, stats_content
            
        except Exception as e:
            # Return error figure and message
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"Error loading dashboard: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            
            error_content = [
                html.H6("Error", className="text-danger"),
                html.P(f"An error occurred: {str(e)}", className="text-danger")
            ]
            
            return error_fig, error_content
    
    return app


# MAIN EXECUTION

if __name__ == "__main__":
    print("Initializing Chicago Traffic Crash Intelligence Platform...")
    
    # Disable debug mode to reduce network overhead
    import os
    os.environ['DASH_DEBUG_MODE'] = 'false'
    
    # Create the dashboard app
    app = create_interactive_dashboard()
    print("Dashboard ready. Starting server...")
    
    # Run with optimized settings
    app.run(debug=False, dev_tools_ui=False, dev_tools_props_check=False)
