import os  # Used for checking file existence
from datetime import datetime
from pathlib import Path

import dash
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output

# --- Configuration: Replace with your actual file paths ---
PROJ_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJ_ROOT / "models"
TRAIN00_DIR = MODELS_DIR / "train" / "00"
PREDICT_NET_FILE = TRAIN00_DIR / "predict.npy"
PM25GNN_AMBIENT_DIR = MODELS_DIR / "train-ambient" / "00"
PREDICT_AMBIENT_FILE = PM25GNN_AMBIENT_DIR / "predict.npy"
LABEL_FILE = TRAIN00_DIR / "label.npy"
TIME_FILE = TRAIN00_DIR / "time.npy"
RAW_DATA_DIR = PROJ_ROOT / "data" / "raw"
LOCATIONS_FILE = RAW_DATA_DIR / "locations.txt"
FEATURES_FILE = RAW_DATA_DIR / "dataset_fire_wind_aligned.npy"
PM25_FILE = RAW_DATA_DIR / "dataset_fire_wind_aligned.npy"

# --- Constants ---
PRED_LEN = 48  # Needed if features/labels have sample/sequence dims
# *** Define Feature Names (CRITICAL STEP) ***
# This list MUST match the column order in your FEATURES_FILE
# Update this list based on your actual feature columns and their order.
# Include features derived in dataset.py if they are in FEATURES_FILE.
# Example based on Table 1 and dataset.py processing:
FEATURE_NAMES = [
    '100m_u_component_of_wind',
    '100m_v_component_of_wind',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'boundary_layer_height',
    'total_precipitation',
    'surface_pressure',
    'u_component_of_wind+950',
    'v_component_of_wind+950',
]

FEATURE_USE = [
    '100m_u_component_of_wind',
    '100m_v_component_of_wind',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'boundary_layer_height',
    'total_precipitation',
    'surface_pressure',
    'u_component_of_wind+950',
    'v_component_of_wind+950',
]
# Add 'Observed_PM25' for the correlation matrix
ALL_VARS_NAMES = FEATURE_NAMES + ["Observed_PM25"]


# --- Data Loading and Preprocessing ---
print("Loading data...")


# --- Helper function to load data ---
def load_npy_data(filepath):
    # ... (same helper function as before) ...
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        raw_data = np.load(filepath)
        if filepath is FEATURES_FILE:
            data = raw_data[:, :, :9]
        elif filepath is PM25_FILE:
            data = raw_data[:, :, -1:]
        else:
            data = raw_data
        print(f"Successfully loaded {filepath}, shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# --- Load data ---
feature_data_raw = load_npy_data(FEATURES_FILE)
label_data_raw = load_npy_data(LABEL_FILE)
time_data_raw = load_npy_data(TIME_FILE)
pm25_data_raw = load_npy_data(PM25_FILE)

# --- Load location data ---
locations = []
location_map = {}
if os.path.exists(LOCATIONS_FILE):
    # ... (same location loading as before) ...
    try:
        with open(LOCATIONS_FILE, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    loc_id = int(parts[0])
                    loc_name = parts[1]
                    label = f"{loc_name} ({loc_id})"
                    locations.append({"label": label, "value": loc_id})
                    location_map[loc_id] = label
        print(f"Loaded {len(locations)} locations from {LOCATIONS_FILE}")
    except Exception as e:
        print(f"Error reading {LOCATIONS_FILE}: {e}")
else:
    print(f"Error: Locations file not found at {LOCATIONS_FILE}")

# --- Exit if essential data failed to load ---
if (
    feature_data_raw is None
    or label_data_raw is None
    or time_data_raw is None
    or pm25_data_raw is None
    or not locations
):
    print(
        "Essential data files (features, label, time, locations) could not be loaded. Exiting."
    )
    exit()

# --- Data Structure Adaptation (Simpler for aligned data) ---
print("Adapting data structure (assuming aligned full time series)...")
try:
    # Step 0: Handle Potential Length Mismatch
    min_len = min(feature_data_raw.shape[0], pm25_data_raw.shape[0], time_data_raw.shape[0])
    feature_data_aligned = feature_data_raw[:min_len]
    pm25_data_aligned = pm25_data_raw[:min_len]
    time_data_aligned = time_data_raw[:min_len]

    num_locations_data = feature_data_aligned.shape[1] # Should be 112
    num_features_data = feature_data_aligned.shape[2] # Should be 17

    # --- Validate Feature Names ---
    if num_features_data != len(FEATURE_NAMES): # Check against 17
        print(f"FATAL ERROR: Feature count mismatch.")
        exit()
    # --- Validate Location Names ---
    if num_locations_data != len(locations): # Check against 112
         print(f"Warning: Location count mismatch! Using data count.")
         locations = [{'label': f"Loc {i}", 'value': i} for i in range(num_locations_data)]
         location_map = {i: f"Loc {i}" for i in range(num_locations_data)}

    # Step 1: Process Timestamps
    time_data_dt = pd.to_datetime(time_data_aligned, unit='s') # Assuming unix seconds

    # Step 2: Data is already aligned (no complex reshaping needed)
    # feature_actual will be feature_data_aligned
    # label_actual will be pm25_data_aligned[:, :, 0] # Remove last dimension

    print("Data preprocessing complete.")

except Exception as e:
    print(f"An error occurred during data structure adaptation: {e}")
    exit()

# --- Dash App Initialization ---
# Use JupyterDash for Colab:
# from jupyter_dash import JupyterDash
# app = JupyterDash(__name__)
app = dash.Dash(__name__)  # Use this for local running
server = app.server  # Expose server for deployment

# --- App Layout ---
app.layout = html.Div(
    [
        html.H1("PM2.5 vs. Meteorology Correlation Heatmap"),
        html.Div(
            [
                html.Label("Select Monitoring Site:", style={"marginRight": "10px"}),
                dcc.Dropdown(
                    id="meteo-heatmap-location-dropdown",
                    options=locations,
                    value=locations[0]["value"]
                    if locations
                    else None,  # Default to first location
                    clearable=False,
                    style={
                        "width": "300px",
                        "display": "inline-block",
                        "marginRight": "20px",
                    },
                ),
                html.Label("Select Date Range:", style={"marginRight": "10px"}),
                dcc.DatePickerRange(
                    id="meteo-heatmap-date-picker-range",
                    min_date_allowed=time_data_dt.min().date(),
                    max_date_allowed=time_data_dt.max().date(),
                    start_date=time_data_dt.min().date(),
                    end_date=time_data_dt.max().date(),
                    display_format="YYYY-MM-DD",
                    style={"display": "inline-block"},
                ),
            ],
            style={"marginBottom": "20px", "display": "flex", "alignItems": "center"},
        ),
        dcc.Loading(
            id="loading-meteo-heatmap-graph",
            type="circle",
            children=dcc.Graph(
                id="meteo-correlation-heatmap", style={"height": "80vh"}
            ),  # Adjust height
        ),
    ]
)


# --- App Callback ---
@app.callback(
    Output("meteo-correlation-heatmap", "figure"),
    Input("meteo-heatmap-location-dropdown", "value"),
    Input("meteo-heatmap-date-picker-range", "start_date"),
    Input("meteo-heatmap-date-picker-range", "end_date"),
)
def update_meteo_heatmap(selected_location_id, start_date, end_date):
    print(
        f"Updating meteo heatmap for Loc ID: {selected_location_id}, Date: {start_date} to {end_date}"
    )

    if selected_location_id is None:
        return go.Figure().update_layout(title="Please select a location")

    # Filter data based on date range
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59
    )
    time_mask = (time_data_dt >= start_datetime) & (time_data_dt <= end_datetime)

    if not np.any(time_mask):
        return go.Figure().update_layout(
            title="No data available for the selected date range"
        )

    # Get the observed PM2.5 and features for the selected location and time period
    # Location ID directly corresponds to the index in the second dimension
    location_index = selected_location_id
    filtered_labels_loc = pm25_data_aligned[time_mask, location_index, 0]
    filtered_features_loc = feature_data_aligned[time_mask, location_index, :]

    # Check if enough data remains
    if filtered_labels_loc.shape[0] < 2:
        return go.Figure().update_layout(
            title="Not enough time points in the selected range for this location"
        )

    # Create a DataFrame for correlation calculation for this specific location
    data_for_corr = {
        name: filtered_features_loc[:, i] for i, name in enumerate(FEATURE_NAMES)
    }
    data_for_corr["Observed_PM25"] = filtered_labels_loc
    corr_df = pd.DataFrame(data_for_corr)

    # Calculate the full correlation matrix
    print(f"Calculating correlation for location {selected_location_id}...")
    try:
        # Handle potential columns with zero variance before calculating correlation
        std_devs = corr_df.std()
        valid_cols = std_devs[std_devs > 1e-6].index.tolist()

        if len(valid_cols) < 2:
            return go.Figure().update_layout(
                title="Not enough varying data columns for correlation"
            )

        corr_matrix = corr_df[valid_cols].corr()
        print(f"Correlation matrix calculated, shape: {corr_matrix.shape}")
        axis_labels = valid_cols  # Use only the names of columns used in correlation

    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return go.Figure().update_layout(title="Error calculating correlation matrix")

    # Create Heatmap figure
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,  # Use .values to get numpy array
            x=axis_labels,
            y=axis_labels,
            colorscale="RdBu",  # Red-Blue colorscale is good for correlation (-1 to 1)
            zmin=-1,  # Enforce range for correlation
            zmax=1,
            colorbar=dict(title="Pearson Corr."),
            # Add text labels inside cells
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",  # Format to 2 decimal places
            textfont={"size": 10},
        )
    )

    location_label = location_map.get(
        selected_location_id, f"Location {selected_location_id}"
    )
    fig.update_layout(
        title=f"PM2.5 vs. Meteorology Correlation - {location_label} ({start_date} to {end_date})",
        xaxis_title="Variable",
        yaxis_title="Variable",
        xaxis_tickangle=-45,
        margin=dict(l=150, r=50, b=150, t=100),  # Adjust margins
        yaxis_autorange="reversed",
    )
    # Ensure square aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


# --- Run the App ---
if __name__ == "__main__":
    print("Starting Dash server...")
    # For Colab, use app.run(mode='inline', port=8050) after changing app initialization
    app.run(debug=True, host="0.0.0.0", port=8050)
