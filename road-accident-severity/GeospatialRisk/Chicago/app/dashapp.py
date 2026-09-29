from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import dash
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Chicago" / "output"

# Load data
print("Loading data...")
community_areas = gpd.read_file(DATA_DIR / "chicago_community_areas.geojson")
community_risk = pd.read_csv(DATA_DIR / "community_area_risk.csv")
crashes_gdf = gpd.read_file(DATA_DIR / "chicago_crashes_with_areas.geojson")

print("Data loaded successfully!")
print("Community areas columns:", community_areas.columns.tolist())
print("Community risk columns:", community_risk.columns.tolist())

# Ensure CRS is correct
community_areas = community_areas.to_crs(epsg=4326)
crashes_gdf = crashes_gdf.to_crs(epsg=4326)

# Merge community areas with risk data - use correct column names
# The risk CSV uses 'community_area_number', find matching column in GeoJSON
geojson_key = None
for col in community_areas.columns:
    if 'area' in col.lower() and 'num' in col.lower():
        geojson_key = col
        break

if geojson_key is None:
    # Try common variations
    for col in ['area_num', 'area_numbe', 'area_number', 'community_area_number']:
        if col in community_areas.columns:
            geojson_key = col
            break

if geojson_key is None:
    raise ValueError(f"Could not find area number column in community_areas. Available columns: {community_areas.columns.tolist()}")

print(f"Using merge key: {geojson_key} -> community_area_number")
community_areas = community_areas.merge(community_risk, left_on=geojson_key, right_on='community_area_number', how='left')

# Calculate risk categories for styling
def get_risk_category(score):
    if pd.isna(score):
        return "No Data"
    elif score >= community_risk['mean_weighted_severity'].quantile(0.9):
        return "Critical"
    elif score >= community_risk['mean_weighted_severity'].quantile(0.75):
        return "High"
    elif score >= community_risk['mean_weighted_severity'].quantile(0.5):
        return "Medium"
    else:
        return "Low"

community_areas['risk_category'] = community_areas['mean_weighted_severity'].apply(get_risk_category)

# Create professional color scale
RISK_COLORS = {
    "No Data": "#f0f0f0",
    "Low": "#2E8B57",      # Sea green
    "Medium": "#FFD700",   # Gold
    "High": "#FF6347",     # Tomato
    "Critical": "#DC143C"  # Crimson
}

def create_professional_choropleth(show_crashes=False, metric="weighted_score"):
    """Create professional choropleth map of Chicago community areas"""
    fig = go.Figure()
    
    # Add community area choropleth
    for _, area in community_areas.iterrows():
        if area.geometry.geom_type == 'Polygon':
            coords = list(area.geometry.exterior.coords)
            lons, lats = zip(*coords)
            
            # Get color based on risk category
            risk_cat = area['risk_category']
            color = RISK_COLORS[risk_cat]
            
            # Create hover text
            hover_text = (
                f"<b>{area.get('community', 'Unknown Area')}</b><br>"
                f"Area Code: {area[geojson_key]}<br>"
                f"Total Crashes: {area.get('total_crashes', 0):,}<br>"
                f"Risk Score: {area.get('mean_weighted_severity', 0):.2f}<br>"
                f"Fatalities: {area.get('fatal', 0)}<br>"
                f"Risk Category: {risk_cat}"
            )
            
            fig.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(width=1, color='white'),
                fill='toself',
                fillcolor=color,
                hovertemplate=hover_text + "<extra></extra>",
                name=risk_cat,
                showlegend=False
            ))
    
    # Optional: Add crash points as overlay
    if show_crashes:
        # Sample crashes for performance
        sample_size = min(2000, len(crashes_gdf))
        sample_crashes = crashes_gdf.sample(sample_size)
        
        fig.add_trace(go.Scattermapbox(
            lat=sample_crashes.geometry.y,
            lon=sample_crashes.geometry.x,
            mode='markers',
            marker=dict(
                size=2,
                color='black',
                opacity=0.4
            ),
            hovertemplate='Crash Location<extra></extra>',
            name='Crash Points',
            showlegend=True
        ))
    
    # Create custom legend
    legend_traces = []
    for category, color in RISK_COLORS.items():
        if category != "No Data" or community_areas['risk_category'].value_counts().get(category, 0) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=[None], lon=[None],  # Invisible points for legend
                mode='markers',
                marker=dict(size=10, color=color),
                name=f"{category} ({community_areas['risk_category'].value_counts().get(category, 0)} areas)",
                showlegend=True
            ))
    
    # Professional layout
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=41.8781, lon=-87.6298),
            zoom=10
        ),
        height=800,
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(
            text="Chicago Traffic Crash Risk Analysis by Community Area",
            x=0.5,
            font=dict(size=24, color="#2c3e50")
        ),
        legend=dict(
            title=dict(text="Risk Categories"),
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

def create_risk_distribution_chart():
    """Create risk distribution by community area"""
    fig = go.Figure()
    
    # Count areas by risk category
    risk_counts = community_areas['risk_category'].value_counts()
    risk_counts = risk_counts.reindex(["Critical", "High", "Medium", "Low", "No Data"])
    
    colors = [RISK_COLORS[cat] for cat in risk_counts.index]
    
    fig.add_trace(go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker=dict(color=colors),
        text=risk_counts.values,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Distribution of Risk Categories Across Community Areas",
        xaxis_title="Risk Category",
        yaxis_title="Number of Community Areas",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

def create_top_areas_table():
    """Create table of highest risk community areas"""
    top_areas = community_areas.nlargest(10, 'mean_weighted_severity').copy()
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Community Area', 'Area Code', 'Total Crashes', 'Risk Score', 'Fatalities', 'Risk Category'],
            fill_color='#2c3e50',
            font=dict(color='white', size=11),
            align='left',
            height=40
        ),
        cells=dict(
            values=[
                top_areas['community'],
                top_areas[geojson_key],
                top_areas['total_crashes'],
                top_areas['mean_weighted_severity'].round(2),
                top_areas['fatal'],
                top_areas['risk_category']
            ],
            fill_color=[
                [RISK_COLORS[cat] for cat in top_areas['risk_category']],
                ['white'] * len(top_areas)
            ],
            align='left',
            font=dict(size=10),
            height=35
        )
    )])
    
    fig.update_layout(
        title="Top 10 Highest Risk Community Areas",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Create app with professional theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Chicago Traffic Crash Risk Analysis", 
                   className="text-center mb-4",
                   style={'color': '#2c3e50', 'fontWeight': 'bold'}),
            html.H5("Professional Geographic Risk Assessment by Community Area", 
                   className="text-center mb-4 text-muted"),
            html.Hr()
        ])
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Map Controls", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Show Crash Points"),
                            dbc.Switch(
                                id="show-crashes-switch",
                                label="Display individual crash locations",
                                value=False
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Risk Metric"),
                            dcc.Dropdown(
                                id="metric-dropdown",
                                options=[
                                    {'label': 'Mean Weighted Severity', 'value': 'mean_weighted_severity'},
                                    {'label': 'Total Crashes', 'value': 'total_crashes'},
                                    {'label': 'Fatalities', 'value': 'fatal'}
                                ],
                                value='mean_weighted_severity',
                                clearable=False
                            )
                        ], width=6)
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Main Map
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id="choropleth-map",
                figure=create_professional_choropleth(),
                style={'height': '800px'}
            )
        ])
    ]),
    
    # Analytics Section
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id="risk-distribution",
                figure=create_risk_distribution_chart()
            )
        ], width=6),
        dbc.Col([
            dcc.Graph(
                id="top-areas-table",
                figure=create_top_areas_table()
            )
        ], width=6)
    ], className="mt-4"),
    
    # Summary Statistics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Analysis Summary", className="card-title"),
                    html.Div([
                        html.P(f"Total Community Areas: {len(community_areas)}"),
                        html.P(f"Areas with Crash Data: {community_areas['total_crashes'].notna().sum()}"),
                        html.P(f"Total Crashes Analyzed: {community_areas['total_crashes'].sum():,}"),
                        html.P(f"Highest Risk Area: {community_areas.loc[community_areas['mean_weighted_severity'].idxmax(), 'community']}"),
                        html.P(f"Average Risk Score: {community_areas['mean_weighted_severity'].mean():.2f}"),
                        html.P(f"Total Fatalities: {community_areas['fatal'].sum()}")
                    ])
                ])
            ])
        ], width=12)
    ], className="mt-4")
    
], fluid=True)

# Callbacks
@app.callback(
    Output("choropleth-map", "figure"),
    [Input("show-crashes-switch", "value"),
     Input("metric-dropdown", "value")]
)
def update_map(show_crashes, metric):
    return create_professional_choropleth(show_crashes=show_crashes, metric=metric)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
