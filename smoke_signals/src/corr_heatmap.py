import dash
import re
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime
import os # Used for checking file existence
from pathlib import Path # Use pathlib for better path handling

# --- Configuration: Replace with your actual file paths ---
# Determine project root relative to this script file
PROJ_ROOT = Path(__file__).resolve().parents[2] # Assumes script is in src/
# Or define explicitly:
# PROJ_ROOT = Path('/home/jon/smoke-signals') # Example explicit path

MODELS_DIR = PROJ_ROOT / "models"
TRAIN00_DIR = MODELS_DIR / "train" / "00"
PM25GNN_AMBIENT_DIR = MODELS_DIR / "train-ambient" / "00"
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Define file paths using pathlib
FEATURES_FILE = RAW_DATA_DIR / "dataset_fire_wind_aligned.npy" # Essential: Raw feature data
LABEL_FILE = TRAIN00_DIR / "label.npy" # Needed to align data
TIME_FILE = TRAIN00_DIR / "time.npy" # Essential: Time alignment
# *** Use the new locations CSV file ***
PROC_DATA_DIR = DATA_DIR / "processed"
LOCATIONS_FILE = PROC_DATA_DIR / "locations-names.csv"
PM25_FILE = RAW_DATA_DIR / "dataset_fire_wind_aligned.npy" # Source for PM2.5 column

# --- Output Directory for Saved HTMLs ---
SAVE_DIR = PROJ_ROOT / "heatmap_exports"
SAVE_DIR.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

# --- Constants ---
PRED_LEN = 48 # Needed if features/labels have sample/sequence dims
# *** Define Feature Names (CRITICAL STEP) ***
# This list MUST match the first N columns in your FEATURES_FILE / PM25_FILE
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
    # Add other feature names here if dataset_fire_wind_aligned.npy contains more than 9 + PM2.5
]
# *** Add 'Observed_PM25' back for the correlation matrix calculation ***
ALL_VARS_NAMES = FEATURE_NAMES + ["Observed_PM25"]


# --- Data Loading and Preprocessing ---
print("Loading data...")

# --- Helper function to load data ---
def load_npy_data(filepath, description="data"):
    filepath = Path(filepath) # Ensure it's a Path object
    if not filepath.exists():
        print(f"Error: File not found at {filepath}")
        return None
    try:
        data = np.load(filepath)
        print(f"Successfully loaded {description} from {filepath}, shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# --- Load data ---
# Load the combined file which contains both features and PM2.5
combined_data_raw = load_npy_data(FEATURES_FILE, description="combined features and PM2.5")
# Load time file for alignment
time_data_raw = load_npy_data(TIME_FILE, description="time data")


# --- Load location data from CSV ---
locations = []
location_map = {} # To map index (0,1,..) to display Label (City Name)
if LOCATIONS_FILE.exists():
    try:
        locations_df = pd.read_csv(LOCATIONS_FILE)
        print(f"CSV Columns read by pandas: {locations_df.columns.tolist()}") # Helps debugging

        # *** Define expected column names in locations-names.csv ***
        index_col_name = 'location_id' # Column with 0, 1, 2...
        city_name_col = 'city_name'      # Column with city names

        # --- Verify the expected columns exist ---
        if index_col_name not in locations_df.columns:
            raise ValueError(f"Column '{index_col_name}' not found in {LOCATIONS_FILE}. Found columns: {locations_df.columns.tolist()}")
        if city_name_col not in locations_df.columns:
             raise ValueError(f"Column '{city_name_col}' not found in {LOCATIONS_FILE}. Found columns: {locations_df.columns.tolist()}")

        # Populate the locations list for the dropdown
        for _, row in locations_df.iterrows():
            loc_index = int(row[index_col_name]) # Get the numerical index (0, 1, ...)
            loc_name = row[city_name_col]      # Get the city name
            label = loc_name
            value = loc_index
            locations.append({'label': label, 'value': value})
            location_map[value] = label # Map index to City Name

        # Sort locations alphabetically by label for the dropdown
        locations = sorted(locations, key=lambda x: x['label'])
        print(f"Loaded {len(locations)} locations from {LOCATIONS_FILE}")

    except Exception as e:
        print(f"Error reading locations CSV {LOCATIONS_FILE}: {e}")
        locations = [] # Ensure locations is empty list on error
else:
    print(f"Error: Locations file not found at {LOCATIONS_FILE}")


# --- Exit if essential data failed to load ---
if combined_data_raw is None or time_data_raw is None or not locations:
    print("Essential data files (combined features/PM2.5, time, locations) could not be loaded. Exiting.")
    exit()

# --- Data Structure Adaptation ---
print("Adapting data structure...")
try:
    # Assuming combined_data_raw is (time, location, feature_and_pm25_columns)
    # And time_data_raw is (time,)

    # Step 0: Handle Potential Length Mismatch
    min_len = min(combined_data_raw.shape[0], time_data_raw.shape[0])
    combined_data_aligned = combined_data_raw[:min_len]
    time_data_aligned = time_data_raw[:min_len]

    # Extract Features and PM2.5 based on FEATURE_NAMES length
    num_features = len(FEATURE_NAMES)
    num_total_cols = combined_data_aligned.shape[2]

    if num_total_cols <= num_features:
        raise ValueError(f"Combined data file has {num_total_cols} columns, but expected at least {num_features} features + 1 PM2.5 column.")

    feature_data_aligned = combined_data_aligned[:, :, :num_features]
    # Extract PM2.5 column (assuming it's the one AFTER the listed features)
    pm25_data_aligned = combined_data_aligned[:, :, num_features:num_features+1]

    print(f"Extracted features shape: {feature_data_aligned.shape}")
    print(f"Extracted PM2.5 shape: {pm25_data_aligned.shape}")


    num_locations_data = feature_data_aligned.shape[1] # Should be 112
    num_features_data = feature_data_aligned.shape[2] # Should match len(FEATURE_NAMES)

    # --- Validate Feature Names ---
    if num_features_data != len(FEATURE_NAMES):
        print(f"FATAL ERROR: Extracted feature count ({num_features_data}) does not match length of FEATURE_NAMES list ({len(FEATURE_NAMES)}).")
        exit()
    # --- Validate Location Count ---
    if num_locations_data != len(locations):
         print(f"Warning: Location count mismatch! Data has {num_locations_data}, CSV file has {len(locations)}. Using data count for processing.")
         locations = [{'label': f"Location Index {i}", 'value': i} for i in range(num_locations_data)]
         location_map = {i: f"Location Index {i}" for i in range(num_locations_data)}

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
app = dash.Dash(__name__) # Use this for local running
server = app.server # Expose server for deployment

# --- App Layout ---
app.layout = html.Div([
    html.H1("PM2.5 vs. Meteorology Correlation Heatmap"),

    html.Div([
        html.Label("Select Monitoring Site:", style={'marginRight': '10px'}),
        dcc.Dropdown(
            id='meteo-heatmap-location-dropdown',
            options=locations, # Populated from CSV
            value=locations[0]['value'] if locations else None, # Default to first location index
            clearable=False,
            style={'width': '300px', 'display': 'inline-block', 'marginRight': '20px'}
        ),

        html.Label("Select Date Range:", style={'marginRight': '10px'}),
        dcc.DatePickerRange(
            id='meteo-heatmap-date-picker-range',
            min_date_allowed=time_data_dt.min().date(),
            max_date_allowed=time_data_dt.max().date(),
            start_date=time_data_dt.min().date(),
            end_date=time_data_dt.max().date(),
            display_format='YYYY-MM-DD',
            style={'display': 'inline-block'}
        ),
    ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),


    dcc.Loading(
        id="loading-meteo-heatmap-graph",
        type="circle",
        children=dcc.Graph(id='meteo-correlation-heatmap', style={'height': '80vh'}) # Adjust height
    )
])

# --- Callback Function ---
@app.callback(
    Output('meteo-correlation-heatmap', 'figure'),
    Input('meteo-heatmap-location-dropdown', 'value'), # Value is the location index (0, 1, ...)
    Input('meteo-heatmap-date-picker-range', 'start_date'),
    Input('meteo-heatmap-date-picker-range', 'end_date'),
)
def update_meteo_heatmap(selected_location_idx, start_date, end_date):
    print(f"Updating meteo heatmap for Loc Index: {selected_location_idx}, Date: {start_date} to {end_date}")

    # Initialize figure in case of early return
    fig = go.Figure()

    if selected_location_idx is None:
        return fig.update_layout(title="Please select a location")

    # Filter data based on date range
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
    time_mask = (time_data_dt >= start_datetime) & (time_data_dt <= end_datetime)

    if not np.any(time_mask):
        return fig.update_layout(title="No data available for the selected date range")

    # Get the features AND PM2.5 for the selected location index and time period
    location_index = selected_location_idx
    filtered_features_loc = feature_data_aligned[time_mask, location_index, :]
    # Extract the corresponding PM2.5 data, removing the last dimension
    filtered_pm25_loc = pm25_data_aligned[time_mask, location_index, 0]

    # Check if enough data remains
    if filtered_features_loc.shape[0] < 2:
        return fig.update_layout(title="Not enough time points in the selected range for this location")

    # Create a DataFrame for correlation calculation including features AND PM2.5
    data_for_corr = {name: filtered_features_loc[:, i] for i, name in enumerate(FEATURE_NAMES)}
    data_for_corr["Observed_PM25"] = filtered_pm25_loc
    corr_df = pd.DataFrame(data_for_corr)

    # Calculate the full correlation matrix (including PM2.5)
    print(f"Calculating feature and PM2.5 correlation for location index {selected_location_idx}...")
    corr_matrix = None # Initialize
    axis_labels = []
    try:
        # Handle potential columns with zero variance before calculating correlation
        std_devs = corr_df.std()
        valid_cols = std_devs[std_devs > 1e-6].index.tolist()

        if len(valid_cols) < 2:
             return fig.update_layout(title="Not enough varying data columns for correlation")

        # Calculate correlation on valid columns
        corr_matrix = corr_df[valid_cols].corr()
        print(f"Correlation matrix calculated, shape: {corr_matrix.shape}")
        # Axis labels include all valid columns (features + Observed_PM25)
        axis_labels = valid_cols

    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return fig.update_layout(title="Error calculating correlation matrix")

    # axis_labels = [
    #     '100m u-Component of Wind',
    #     '100m v-Component of Wind',
    #     '2m Dewpoint Temperature',
    #     '2m Temperature',
    #     'Planetary Boundary Layer Height',
    #     'Total Precipitation',
    #     'Surface Pressure',
    #     'u-Component of Wind at 950 hPa',
    #     'v-Component of Wind at 950 hPa',
    #     # Add other feature names here if dataset_fire_wind_aligned.npy contains more than 9 + PM2.5
    # ]

    # --- Create and Save Heatmap ---
    if corr_matrix is not None and len(axis_labels) > 0:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, # Use .values to get numpy array
            x=axis_labels,
            y=axis_labels,
            colorscale='RdBu', # Red-Blue colorscale is good for correlation (-1 to 1)
            zmin=-1, # Enforce range for correlation
            zmax=1,
            colorbar=dict(title='Pearson Corr.'),
            # Add text labels inside cells
            text=corr_matrix.values,
            texttemplate="%{text:.2f}", # Format to 2 decimal places
            textfont={"size":10}
        ))

        # Get the City Name using the location_map
        location_label = location_map.get(selected_location_idx, f"Location_Index_{selected_location_idx}")
        # Clean label for filename
        clean_location_label = re.sub(r'[^\w\-]+', '_', location_label) # Replace non-alphanumeric with underscore

        fig.update_layout(
            title=f'PM2.5 vs. Meteorology Correlation - {location_label} ({start_date} to {end_date})',
            xaxis_title='Variable',
            yaxis_title='Variable',
            xaxis_tickangle=-45,
            margin=dict(l=150, r=50, b=150, t=100), # Adjust margins
            yaxis_autorange='reversed'
        )
        # Ensure square aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # *** Save the figure as HTML ***
        try:
            filename = f"corr_heatmap_{clean_location_label}_{start_date}_to_{end_date}.html"
            filepath = SAVE_DIR / filename
            fig.write_html(str(filepath))
            print(f"Successfully saved heatmap to: {filepath}")
        except Exception as e:
            print(f"Error saving heatmap HTML: {e}")
            # Continue to return the figure even if saving fails
    else:
        # Handle case where correlation matrix calculation failed but didn't raise exception
         return fig.update_layout(title="Could not generate correlation matrix")


    return fig # Return the figure to the dcc.Graph component

# --- Run the App ---
if __name__ == '__main__':
    print(f"Saving HTML files to: {SAVE_DIR}")
    print("Starting Dash server...")
    # For Colab, use app.run(mode='inline', port=8050) after changing app initialization
    app.run(debug=True, host='0.0.0.0', port=8050)
