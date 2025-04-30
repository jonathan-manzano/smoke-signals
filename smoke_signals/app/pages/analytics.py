# analytics.py
from app.data_loader import load_and_preprocess_data
import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import os
import sys



dash.register_page(__name__, path="/analytics", name="Methods")

# Load data
df, locations, location_map = load_and_preprocess_data()

layout = html.Div([

    # Hero
    html.Section(
        html.Div([
            html.H1("Analytics Dashboard"),
            html.P("Interactive charts to explore your data", className="lead")
        ], className="container"),
        className="hero"
    ),

    # Time Series Section
    html.Section(
        html.Div([
            html.H2("PM2.5 Time Series Analysis"),
            html.Div([
                html.Label("Select Monitoring Site:"),
                dcc.Dropdown(
                    id='location-dropdown',
                    options=locations,
                    value=locations[0]['value'] if locations else None,
                    clearable=False
                ),
                html.Label("Select Date Range:"),
                dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed=df['timestamp'].min().date(),
                    max_date_allowed=df['timestamp'].max().date(),
                    start_date=df['timestamp'].min().date(),
                    end_date=df['timestamp'].max().date()
                ),
                html.Label("Select Data to Display:"),
                dcc.Checklist(
                    id='data-checklist',
                    options=[
                        {'label': 'Observed PM2.5', 'value': 'observed'},
                        {'label': 'Predicted Net PM2.5', 'value': 'predicted_net'},
                        {'label': 'Predicted Ambient PM2.5', 'value': 'predicted_ambient'},
                        {'label': 'Estimated Fire-Specific PM2.5', 'value': 'fire_specific'}
                    ],
                    value=['observed', 'predicted_net']
                )
            ]),
            dcc.Graph(id='time-series-graph')
        ], className="container")
    )
])

@callback(
    Output('time-series-graph', 'figure'),
    Input('location-dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('data-checklist', 'value')
)
def update_time_series(selected_location_idx, start_date, end_date, selected_data):
    if selected_location_idx is None:
        return go.Figure().update_layout(title="Please select a location")

    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)

    filtered_df = df[
        (df['location_id'] == selected_location_idx) &
        (df['timestamp'] >= start_datetime) &
        (df['timestamp'] <= end_datetime)
    ]

    if filtered_df.empty:
        return go.Figure().update_layout(title="No data available for the selected range")

    fig = go.Figure()
    location_label = location_map.get(selected_location_idx, f"Location {selected_location_idx}")

    if 'observed' in selected_data:
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['observed'], mode='lines', name='Observed PM2.5'))
    if 'predicted_net' in selected_data:
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['predicted_net'], mode='lines', name='Predicted Net PM2.5'))
    if 'predicted_ambient' in selected_data:
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['predicted_ambient'], mode='lines', name='Predicted Ambient PM2.5'))
    if 'fire_specific' in selected_data:
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['fire_specific'], mode='lines', name='Fire-Specific PM2.5'))

    fig.update_layout(
        title=f"PM2.5 Time Series for {location_label}",
        xaxis_title="Timestamp",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        hovermode="x unified"
    )

    return fig
