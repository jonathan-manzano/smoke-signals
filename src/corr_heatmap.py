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
PM25GNN_DIR = PROJ_ROOT / "models" / "pm25gnn" / "00"
PREDICT_NET_FILE = PM25GNN_DIR / "predict.npy"
PM25GNN_AMBIENT_DIR = PROJ_ROOT / "models" / "pm25gnn-ambient" / "00"
PREDICT_AMBIENT_FILE = PM25GNN_AMBIENT_DIR / "predict.npy"
LABEL_FILE = PM25GNN_DIR / "label.npy"
TIME_FILE = PM25GNN_DIR / "time.npy"
RAW_DATA_DIR = PROJ_ROOT / "data" / "raw"
LOCATIONS_FILE = RAW_DATA_DIR / "locations.txt"
FEATURES_FILE = RAW_DATA_DIR / "dataset_fire_wind_aligned.npy"


# --- Constants ---
PRED_LEN = 48  # Needed if features/labels have sample/sequence dims
# *** Define Feature Names (CRITICAL STEP) ***
# This list MUST match the column order in your FEATURES_FILE
# Update this list based on your actual feature columns and their order.
# Include features derived in dataset.py if they are in FEATURES_FILE.
# Example based on Table 1 and dataset.py processing:
FEATURE_NAMES = [
    "100m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "boundary_layer_height",
    "total_precipitation",
    "surface_pressure",
    "u_component_of_wind+950",
    "v_component_of_wind+950",
    "frp_25km_idw",
    "frp_50km_idw",
    "frp_100km_idw",
    "frp_500km_idw",
    "numfires",
    "interp_flag",
    "julian_date",
    "time_of_day",
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
        data = np.load(filepath)
        print(f"Successfully loaded {filepath}, shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


# --- Load data ---
feature_data_raw = load_npy_data(FEATURES_FILE)
label_data_raw = load_npy_data(LABEL_FILE)
time_data_raw = load_npy_data(TIME_FILE)

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
    or not locations
):
    print(
        "Essential data files (features, label, time, locations) could not be loaded. Exiting."
    )
    exit()

# --- Data Structure Adaptation ---
print("Adapting data structure...")
try:
    # Step 0: Handle Shape Mismatch
    min_len = min(
        feature_data_raw.shape[0], label_data_raw.shape[0], time_data_raw.shape[0]
    )
    feature_data_trunc = feature_data_raw[:min_len]
    label_data_trunc = label_data_raw[:min_len]
    time_data_trunc = time_data_raw[:min_len]

    # Get number of locations and features
    num_locations_data = label_data_raw.shape[
        -2
    ]  # Assuming location is second-to-last dim
    num_features_data = feature_data_raw.shape[-1]  # Assuming features is last dim

    # --- Validate Feature Names ---
    if num_features_data != len(FEATURE_NAMES):
        print(
            f"FATAL ERROR: Number of features in data ({num_features_data}) does not match length of FEATURE_NAMES list ({len(FEATURE_NAMES)})."
        )
        print(
            "Please update the FEATURE_NAMES list in the script to match your data columns."
        )
        exit()
    else:
        print(f"Data has {num_features_data} features, matching FEATURE_NAMES list.")

    if num_locations_data != len(locations):
        print(
            f"Warning: Location count mismatch! Data has {num_locations_data}, file has {len(locations)}. Using data count."
        )
        locations = [
            {"label": f"Loc {i}", "value": i} for i in range(num_locations_data)
        ]
        location_map = {i: f"Loc {i}" for i in range(num_locations_data)}

    # Step 1: Extract Relevant Horizon & Reshape Data
    if len(label_data_trunc.shape) == 4:  # Assumes (samples, seq, loc, 1) for label
        sequence_length = label_data_trunc.shape[1]
        if PRED_LEN > sequence_length:
            raise ValueError("PRED_LEN > sequence length")

        # Extract label part corresponding to predictions
        label_pred = label_data_trunc[
            :, -PRED_LEN:, :, 0
        ]  # Shape: (samples, PRED_LEN, loc)

        # Extract corresponding feature part
        # Assuming features file has same sample/seq/loc structure, but last dim is num_features
        if feature_data_trunc.shape[:3] != label_data_trunc.shape[:3]:
            raise ValueError(
                "Feature and Label sample/sequence/location dimensions mismatch"
            )
        feature_pred = feature_data_trunc[
            :, -PRED_LEN:, :, :
        ]  # Shape: (samples, PRED_LEN, loc, feat)

        num_samples = label_pred.shape[0]
        num_locs = label_pred.shape[2]
        num_feats = feature_pred.shape[3]

        # Reshape to (time*pred_len, locations, features/labels)
        label_actual = label_pred.reshape(
            num_samples * PRED_LEN, num_locs
        )  # Shape: (time, loc)
        feature_actual = feature_pred.reshape(
            num_samples * PRED_LEN, num_locs, num_feats
        )  # Shape: (time, loc, feat)

        # Process time data
        if time_data_trunc.shape[0] == num_samples:
            time_data_processed = np.repeat(time_data_trunc, PRED_LEN)
        else:
            raise ValueError("Time data shape mismatch")

    elif len(label_data_trunc.shape) == 2:  # Assumes (time, location)
        label_actual = label_data_trunc  # Shape: (time, loc)
        if feature_data_trunc.shape[:2] != label_actual.shape[:2]:
            raise ValueError("Feature and Label time/location dimensions mismatch")
        feature_actual = feature_data_trunc  # Shape: (time, loc, feat)
        time_data_processed = time_data_trunc
    else:
        raise ValueError("Unsupported label data shape")

    # Step 2: Process Timestamps
    if time_data_processed.shape[0] != label_actual.shape[0]:
        raise ValueError("Timestamp array shape incompatible")
    time_data_dt = pd.to_datetime(time_data_processed, unit="s")

    print("Data preprocessing complete.")
    print(f"Processed label data shape: {label_actual.shape}")
    print(f"Processed feature data shape: {feature_actual.shape}")
    print(f"Processed time data length: {len(time_data_dt)}")

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
    filtered_labels_loc = label_actual[time_mask, location_index]
    filtered_features_loc = feature_actual[time_mask, location_index, :]

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
