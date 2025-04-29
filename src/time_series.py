import os # Used for checking file existence
from datetime import datetime
from pathlib import Path

import dash
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dash import dcc, html, Input, Output, State

# --- Configuration: Replace with your actual file paths ---
# Make sure you have two predict.npy files: one from the pm25gnn (net) model
# and one from the pm25gnn-ambient model.
# PREDICT_NET_FILE = '/home/jon/smoke-signals/models/pm25gnn/00/predict.npy' # Example path
# PREDICT_AMBIENT_FILE = '/home/jon/smoke-signals/models/pm25gnn-ambient/00/predict.npy' # Example path
# LABEL_FILE = '/home/jon/smoke-signals/models/pm25gnn/00/label.npy' # Example path
# TIME_FILE = '/home/jon/smoke-signals/models/pm25gnn/00/time.npy' # Example path
# LOCATIONS_FILE = '/home/jon/smoke-signals/data/raw/locations.txt' # Example path

PROJ_ROOT = Path(__file__).resolve().parents[1]
PM25GNN_DIR = PROJ_ROOT / "models" / "pm25gnn" / "00"
PREDICT_NET_FILE = (
    PM25GNN_DIR / "predict.npy"
)  # Replace with path to pm25gnn predictions
PM25GNN_AMBIENT_DIR = PROJ_ROOT / "models" / "pm25gnn-ambient" / "00"
PREDICT_AMBIENT_FILE = (
    PM25GNN_AMBIENT_DIR
    / "predict.npy"  # Replace with path to pm25gnn-ambient predictions
)
LABEL_FILE = PM25GNN_DIR / "label.npy"  # Replace with path to label.npy
TIME_FILE = PM25GNN_DIR / "time.npy"  # Replace with path to time.npy (see notes below)
RAW_DATA_DIR = PROJ_ROOT / "data" / "raw"
LOCATIONS_FILE = RAW_DATA_DIR / "locations.txt"  # Replace with path to locations.txt


# --- Constants (Update if necessary based on your training config) ---
# These might be needed if your data includes history/prediction length dimensions
# Check your config.yaml or train.py for these values
HIST_LEN = 240 # Example: History length used in training
PRED_LEN = 48  # Example: Prediction length used in training (Set to your actual value)

# --- Data Loading and Preprocessing ---
print("Loading data...")

# --- Helper function to load data with error handling ---
def load_npy_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        print("Please ensure the file path is correct and the file exists.")
        return None
    try:
        data = np.load(filepath)
        print(f"Successfully loaded {filepath}, shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# --- Load prediction, label, and time data ---
predict_net_data_raw = load_npy_data(PREDICT_NET_FILE)
predict_ambient_data_raw = load_npy_data(PREDICT_AMBIENT_FILE)
label_data_raw = load_npy_data(LABEL_FILE)
time_data_raw = load_npy_data(TIME_FILE) # This might be indices or timestamps

# --- Load location data ---
locations = []
if os.path.exists(LOCATIONS_FILE):
    try:
        with open(LOCATIONS_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    loc_id = int(parts[0])
                    loc_name = parts[1]
                    # Combine ID and name for clarity in dropdown
                    locations.append({'label': f"{loc_name} (ID: {loc_id})", 'value': loc_id})
        print(f"Loaded {len(locations)} locations from {LOCATIONS_FILE}")
    except Exception as e:
        print(f"Error reading {LOCATIONS_FILE}: {e}")
else:
    print(f"Error: Locations file not found at {LOCATIONS_FILE}")

# --- Exit if essential data failed to load ---
if predict_net_data_raw is None or \
   predict_ambient_data_raw is None or \
   label_data_raw is None or \
   time_data_raw is None or not locations:
    print("Essential data files could not be loaded. Exiting.")
    exit()

# --- Data Structure Adaptation (CRITICAL STEP) ---
# Handles shape mismatches and extracts relevant data.

print("Adapting data structure (Verify this section based on your .npy files)...")
try:
    # --- Step 0: Handle Potential Shape Mismatch ---
    # Find the minimum length along the first dimension (samples/time)
    min_len = min(predict_net_data_raw.shape[0],
                  predict_ambient_data_raw.shape[0],
                  label_data_raw.shape[0],
                  time_data_raw.shape[0]) # Include time_data_raw here

    print(f"Detected minimum length across primary arrays: {min_len}")

    # Truncate all arrays to the minimum length to ensure consistency
    predict_net_data_trunc = predict_net_data_raw[:min_len]
    predict_ambient_data_trunc = predict_ambient_data_raw[:min_len]
    label_data_trunc = label_data_raw[:min_len]
    time_data_trunc = time_data_raw[:min_len] # Also truncate time data

    print(f"Truncated predict_net shape: {predict_net_data_trunc.shape}")
    print(f"Truncated predict_ambient shape: {predict_ambient_data_trunc.shape}")
    print(f"Truncated label shape: {label_data_trunc.shape}")
    print(f"Truncated time shape: {time_data_trunc.shape}")


    # Get number of locations from data (more reliable than file if file is wrong)
    # Assuming location is the second-to-last dim in the *original* shape
    num_locations_data = predict_net_data_raw.shape[-2]
    if num_locations_data != len(locations):
         print(f"Warning: Location count mismatch! Data has {num_locations_data}, file has {len(locations)}. Using data count.")
         # Fallback: Use index if location file is mismatched/missing
         locations = [{'label': f"Location Index {i}", 'value': i} for i in range(num_locations_data)]


    # --- Step 1: Extract Prediction Horizon ---
    # Assuming the *last* `pred_len` steps of the sequence dimension are the predictions.
    # Your data shape is (samples, sequence_length, num_locations, 1)
    # Sequence length seems to be 288 based on your output.
    # Let's assume PRED_LEN is 48 (as in the paper), so history is 240.
    # We need to extract the prediction part from the sequence dimension (index 1)
    # And remove the last dimension (index 3, size 1)

    # Check if the assumed sequence dimension exists
    if len(predict_net_data_trunc.shape) == 4:
        sequence_length = predict_net_data_trunc.shape[1]
        print(f"Detected sequence length: {sequence_length}")
        # Assuming PRED_LEN is defined correctly above
        if PRED_LEN > sequence_length:
             raise ValueError(f"PRED_LEN ({PRED_LEN}) cannot be greater than sequence length ({sequence_length})")

        # Extract the prediction part (last PRED_LEN steps) and remove the last dimension
        predict_net_pred = predict_net_data_trunc[:, -PRED_LEN:, :, 0]
        predict_ambient_pred = predict_ambient_data_trunc[:, -PRED_LEN:, :, 0]
        # Extract the corresponding labels
        label_pred = label_data_trunc[:, -PRED_LEN:, :, 0]

        print(f"Shape after extracting prediction horizon ({PRED_LEN} steps): {predict_net_pred.shape}")

        # Now, reshape to flatten the sample and prediction steps into a single time dimension
        # New shape: (num_samples * PRED_LEN, num_locations)
        num_samples = predict_net_pred.shape[0]
        num_locs = predict_net_pred.shape[2] # Locations dimension

        predict_net_actual = predict_net_pred.reshape(num_samples * PRED_LEN, num_locs)
        predict_ambient_actual = predict_ambient_pred.reshape(num_samples * PRED_LEN, num_locs)
        label_actual = label_pred.reshape(num_samples * PRED_LEN, num_locs)

        print(f"Shape after reshaping (time*pred_len, locations): {predict_net_actual.shape}")

        # Time data needs careful handling: Repeat each timestamp PRED_LEN times
        if time_data_trunc.shape[0] == num_samples:
             time_data_processed = np.repeat(time_data_trunc, PRED_LEN)
             print(f"Processed time shape after repeating: {time_data_processed.shape}")
        else:
             raise ValueError(f"Time data shape {time_data_trunc.shape} does not match number of samples {num_samples}. Cannot repeat timestamps.")

    elif len(predict_net_data_trunc.shape) == 2: # Assumes (time, location) - Less likely given your output
         print("Warning: Data shape is (time, location). Assuming PRED_LEN=1 and no history dimension.")
         predict_net_actual = predict_net_data_trunc
         predict_ambient_actual = predict_ambient_data_trunc
         label_actual = label_data_trunc
         time_data_processed = time_data_trunc # Assumes time_data_raw is 1D array of timestamps/indices
    else:
         raise ValueError(f"Unsupported prediction data shape: {predict_net_data_trunc.shape}. Please adapt the slicing.")

    # --- Step 2: Process Timestamps ---
    # Convert to datetime objects (assuming Unix timestamps)
    print("Processing timestamps...")
    if time_data_processed.shape[0] != predict_net_actual.shape[0]:
         raise ValueError(f"Final timestamp array shape {time_data_processed.shape} incompatible with final prediction shape {predict_net_actual.shape}.")

    time_data_dt = pd.to_datetime(time_data_processed, unit='s')
    print(f"Processed time range: {time_data_dt.min()} to {time_data_dt.max()}")

    # --- Step 3: Combine into a DataFrame ---
    print("Creating DataFrame...")
    num_times_final = predict_net_actual.shape[0]
    num_locs_final = predict_net_actual.shape[1]

    df_data = {
        'timestamp': np.repeat(time_data_dt, num_locs_final), # Repeat time for each location
        'location_id': np.tile(np.arange(num_locs_final), num_times_final), # Tile locations for each time step
        'observed': label_actual.flatten(),
        'predicted_net': predict_net_actual.flatten(),
        'predicted_ambient': predict_ambient_actual.flatten()
    }
    df = pd.DataFrame(df_data)
    df['fire_specific'] = df['predicted_net'] - df['predicted_ambient']
    # Optional: Clip negative fire-specific values
    # df['fire_specific_clipped'] = df['fire_specific'].clip(lower=0)

    print("Data preprocessing complete.")
    print(df.head())
    print(f"DataFrame shape: {df.shape}")


except Exception as e:
    print(f"An error occurred during data structure adaptation: {e}")
    print("Please carefully check the NumPy file structures and adjust the slicing/timestamp logic in the 'Data Structure Adaptation' section.")
    exit()


# --- Dash App Initialization ---
# Use JupyterDash for Colab:
# from jupyter_dash import JupyterDash
# app = JupyterDash(__name__)
app = dash.Dash(__name__) # Use this for local running
server = app.server # Expose server for deployment

# --- App Layout ---
app.layout = html.Div([
    html.H1("PM2.5 Time Series Comparison and Fire Contribution"),

    html.Div([
        html.Label("Select Monitoring Site:", style={'marginRight': '10px'}),
        dcc.Dropdown(
            id='location-dropdown',
            options=locations,
            value=locations[0]['value'] if locations else None, # Default to first location
            clearable=False,
            style={'width': '300px', 'display': 'inline-block', 'marginRight': '20px'}
        ),

        html.Label("Select Date Range:", style={'marginRight': '10px'}),
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=time_data_dt.min().date(),
            max_date_allowed=time_data_dt.max().date(),
            start_date=time_data_dt.min().date(),
            end_date=time_data_dt.max().date(), # Default to full range initially
            display_format='YYYY-MM-DD',
            style={'display': 'inline-block'}
        ),
    ], style={'marginBottom': '20px', 'display': 'flex', 'alignItems': 'center'}),

    html.Div([
        html.Label("Select Data to Display:"),
        dcc.Checklist(
            id='data-checklist',
            options=[
                {'label': 'Observed PM2.5', 'value': 'observed'},
                {'label': 'Predicted Net PM2.5 (pm25gnn)', 'value': 'predicted_net'},
                {'label': 'Predicted Ambient PM2.5 (pm25gnn-ambient)', 'value': 'predicted_ambient'},
                {'label': 'Estimated Fire-Specific PM2.5', 'value': 'fire_specific'}
            ],
            value=['observed', 'predicted_net'], # Default selections
            inline=True,
            labelStyle={'display': 'inline-block', 'marginRight': '15px'}
        )
    ], style={'marginBottom': '20px'}),

    dcc.Loading( # Add a loading indicator
        id="loading-graph",
        type="circle",
        children=dcc.Graph(id='pm25-time-series-graph')
    )
])

# --- App Callback ---
@app.callback(
    Output('pm25-time-series-graph', 'figure'),
    Input('location-dropdown', 'value'),
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('data-checklist', 'value')
)
def update_graph(selected_location_id, start_date, end_date, selected_data):
    # print(f"Updating graph for location ID: {selected_location_id}, Date Range: {start_date} to {end_date}, Data: {selected_data}") # Keep for debugging if needed

    if selected_location_id is None:
        return go.Figure().update_layout(title="Please select a location")

    # Filter DataFrame based on selections
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    # Include the whole end day
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)

    # Efficiently filter the DataFrame
    filtered_df = df[
        (df['location_id'] == selected_location_id) &
        (df['timestamp'] >= start_datetime) &
        (df['timestamp'] <= end_datetime)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    if filtered_df.empty:
        return go.Figure().update_layout(title=f"No data available for Location ID {selected_location_id} in the selected date range")

    # Create figure
    fig = go.Figure()
    location_label = next((loc['label'] for loc in locations if loc['value'] == selected_location_id), f"Location Index {selected_location_id}")

    # Add traces based on checklist selection
    if 'observed' in selected_data:
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['observed'], mode='lines', name='Observed PM2.5', line=dict(color='black')))
    if 'predicted_net' in selected_data:
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['predicted_net'], mode='lines', name='Predicted Net (pm25gnn)', line=dict(color='red')))
    if 'predicted_ambient' in selected_data:
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['predicted_ambient'], mode='lines', name='Predicted Ambient (pm25gnn-ambient)', line=dict(color='blue', dash='dash')))
    if 'fire_specific' in selected_data:
        # You might want to plot the clipped version depending on interpretation
        # y_fire = filtered_df['fire_specific_clipped']
        y_fire = filtered_df['fire_specific']
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=y_fire, mode='lines', name='Estimated Fire-Specific', line=dict(color='orange', dash='dot')))


    # Update layout
    fig.update_layout(
        title=f'PM2.5 Concentration Comparison for {location_label}',
        xaxis_title='Date and Time',
        yaxis_title='PM2.5 Concentration (µg/m³)', # Assuming standard units from paper
        legend_title='Data Series',
        hovermode='x unified', # Improves tooltip display
        xaxis_range=[start_datetime, end_datetime] # Ensure plot respects date picker range
    )

    # Optional: Add range slider for zooming
    fig.update_xaxes(rangeslider_visible=True)

    return fig

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Dash server...")
    # Set host='0.0.0.0' to make it accessible on your network
    # Use debug=True for development, False for production
    # For Colab, use app.run_server(mode='inline', port=8050) after changing app initialization
    app.run(debug=True, host='0.0.0.0', port=8050)
