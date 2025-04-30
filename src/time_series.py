import os  # Used for checking file existence
from datetime import datetime
from pathlib import Path

import dash
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dash import dcc, html, Input, Output

# --- Configuration ---
PROJ_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJ_ROOT / "models"
PM25GNN_DIR = MODEL_DIR / "train" / "00"
PREDICT_NET_FILE = PM25GNN_DIR / "predict.npy"
PM25GNN_AMBIENT_DIR = MODEL_DIR / "train-ambient" / "00"
PREDICT_AMBIENT_FILE = PM25GNN_AMBIENT_DIR / "predict.npy"
LABEL_FILE = PM25GNN_DIR / "label.npy"
TIME_FILE = PM25GNN_DIR / "time.npy"
DATA_DIR = PROJ_ROOT / "data"
PROC_DATA_DIR = DATA_DIR / "processed"
LOCATIONS_FILE = PROC_DATA_DIR / "locations-names.csv"

# --- Constants (Update if necessary based on your training config) ---
PRED_LEN = 48  # Example: Prediction length used in training (Set to your actual value)

# --- Data Loading and Preprocessing ---
print("Loading data...")

# --- Helper function to load data ---
def load_npy_data(filepath):
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

# --- Load prediction, label, and time data ---
predict_net_data_raw = load_npy_data(PREDICT_NET_FILE)
predict_ambient_data_raw = load_npy_data(PREDICT_AMBIENT_FILE)
label_data_raw = load_npy_data(LABEL_FILE)
time_data_raw = load_npy_data(TIME_FILE)

# --- Load location data from the UPDATED CSV ---
locations = []
location_map = {} # To map index (0,1,..) to display Label
if os.path.exists(LOCATIONS_FILE):
    try:
        # *** Read the updated CSV ***
        # Use index_col=0 to handle the first unnamed column
        locations_df = pd.read_csv(LOCATIONS_FILE, index_col=0)
        print(f"CSV Columns read by pandas: {locations_df.columns.tolist()}") # Helps debugging

        # --- Define the column names we expect based on the CSV header ---
        # Column containing 'city0', 'city1', etc.
        loc_id_col = 'location_id'
        # Column containing 'Carmel Valley', 'Poleta', etc.
        city_name_col = 'city_name'

        # --- Verify the expected columns exist ---
        if loc_id_col not in locations_df.columns:
            raise ValueError(f"Column '{loc_id_col}' not found in CSV. Found columns: {locations_df.columns.tolist()}")
        if city_name_col not in locations_df.columns:
             raise ValueError(f"Column '{city_name_col}' not found in CSV. Found columns: {locations_df.columns.tolist()}")

        # Populate the locations list for the dropdown
        # Use the DataFrame index (0, 1, 2...) as the dropdown 'value'
        # because this index corresponds to the NumPy array columns
        for index, row in locations_df.iterrows():
            loc_index = index # This is the numerical index (0, 1, 2...)
            loc_id_str = row[loc_id_col] # Get value from 'location_id' column (e.g., 'city0')
            loc_name = row[city_name_col] # Get value from 'city_name' column (e.g., 'Carmel Valley')

            # *** Construct the CORRECT label ***
            label = f"{loc_name}" # Display label: "Carmel Valley (city0)"

            locations.append({'label': label, 'value': loc_index}) # Use index as value
            location_map[loc_index] = label # Map index to display label

        print(f"Loaded {len(locations)} locations from {LOCATIONS_FILE}")

    except Exception as e:
        print(f"Error reading locations CSV {LOCATIONS_FILE}: {e}")
        locations = [] # Ensure locations is empty list on error
else:
    print(f"Error: Locations file not found at {LOCATIONS_FILE}")


# --- Exit if essential data failed to load ---
if predict_net_data_raw is None or \
   predict_ambient_data_raw is None or \
   label_data_raw is None or \
   time_data_raw is None or not locations: # Check if locations list is populated
    print("Essential data files (predictions, label, time, locations) could not be loaded. Exiting.")
    exit()

# --- Data Structure Adaptation (CRITICAL STEP - Same as before) ---
# (Assuming this part is correct based on previous iterations)
print("Adapting data structure (Verify this section based on your .npy files)...")
try:
    # Step 0: Handle Potential Shape Mismatch
    min_len = min(predict_net_data_raw.shape[0],
                  predict_ambient_data_raw.shape[0],
                  label_data_raw.shape[0],
                  time_data_raw.shape[0])
    predict_net_data_trunc = predict_net_data_raw[:min_len]
    predict_ambient_data_trunc = predict_ambient_data_raw[:min_len]
    label_data_trunc = label_data_raw[:min_len]
    time_data_trunc = time_data_raw[:min_len]

    # Get number of locations from data (more reliable than file if file is wrong)
    num_locations_data = predict_net_data_raw.shape[-2] # Index for location dimension
    if num_locations_data != len(locations):
         print(f"Warning: Location count mismatch! Data has {num_locations_data}, CSV file has {len(locations)}. Dropdown will use CSV, but array processing uses data shape.")
         # Decide how to handle mismatch - maybe filter locations list? For now, just warn.

    # Step 1: Extract Prediction Horizon & Reshape
    if len(predict_net_data_trunc.shape) == 4:
        sequence_length = predict_net_data_trunc.shape[1]
        if PRED_LEN > sequence_length: raise ValueError("PRED_LEN > sequence length")
        predict_net_pred = predict_net_data_trunc[:, -PRED_LEN:, :, 0]
        predict_ambient_pred = predict_ambient_data_trunc[:, -PRED_LEN:, :, 0]
        label_pred = label_data_trunc[:, -PRED_LEN:, :, 0]
        num_samples = predict_net_pred.shape[0]
        num_locs = num_locations_data # Use actual locations dim from data
        predict_net_actual = predict_net_pred.reshape(num_samples * PRED_LEN, num_locs)
        predict_ambient_actual = predict_ambient_pred.reshape(num_samples * PRED_LEN, num_locs)
        label_actual = label_pred.reshape(num_samples * PRED_LEN, num_locs)
        if time_data_trunc.shape[0] == num_samples:
             time_data_processed = np.repeat(time_data_trunc, PRED_LEN)
        else: raise ValueError("Time data shape mismatch")
    elif len(predict_net_data_trunc.shape) == 2: # Assumes (time, location)
         num_locs = num_locations_data
         predict_net_actual = predict_net_data_trunc
         predict_ambient_actual = predict_ambient_data_trunc
         label_actual = label_data_trunc
         time_data_processed = time_data_trunc
    else: raise ValueError("Unsupported prediction data shape")

    # Step 2: Process Timestamps
    if time_data_processed.shape[0] != predict_net_actual.shape[0]:
         raise ValueError("Timestamp array shape incompatible")
    time_data_dt = pd.to_datetime(time_data_processed, unit='s')

    # Step 3: Combine into a DataFrame
    print("Creating DataFrame...")
    num_times_final = predict_net_actual.shape[0]
    num_locs_final = predict_net_actual.shape[1] # Use actual num_locs from data
    df_list = []

    # Create DataFrame using the location INDEX (0, 1, 2...)
    # This index corresponds to the dropdown 'value' and the numpy array columns
    for data_loc_idx in range(num_locs_final):
        loc_df = pd.DataFrame({
            'timestamp': time_data_dt,
            'location_id': data_loc_idx, # Use the index (0, 1, ...)
            'observed': label_actual[:, data_loc_idx],
            'predicted_net': predict_net_actual[:, data_loc_idx],
            'predicted_ambient': predict_ambient_actual[:, data_loc_idx]
        })
        df_list.append(loc_df)

    df = pd.concat(df_list, ignore_index=True)
    df['fire_specific'] = df['predicted_net'] - df['predicted_ambient']
    df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure correct dtype

    print("Data preprocessing complete.")
    print(df.head())
    print(f"DataFrame shape: {df.shape}")

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
    html.H1("PM2.5 Time Series Comparison and Fire Contribution"),

    html.Div([
        html.Label("Select Monitoring Site:", style={'marginRight': '10px'}),
        dcc.Dropdown(
            id='location-dropdown',
            options=locations, # Populated from CSV, value is index (0,1,...)
            value=locations[0]['value'] if locations else None, # Default to first location index
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
    Input('location-dropdown', 'value'), # This value is the location INDEX (0, 1, ...)
    Input('date-picker-range', 'start_date'),
    Input('date-picker-range', 'end_date'),
    Input('data-checklist', 'value')
)
def update_graph(selected_location_idx, start_date, end_date, selected_data):
    # print(f"Updating graph for location index: {selected_location_idx}, Date Range: {start_date} to {end_date}, Data: {selected_data}") # Debugging

    if selected_location_idx is None:
        return go.Figure().update_layout(title="Please select a location")

    # Filter DataFrame based on selections
    # The DataFrame's 'location_id' column now holds the index (0, 1, ...)
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)

    filtered_df = df[
        (df['location_id'] == selected_location_idx) &
        (df['timestamp'] >= start_datetime) &
        (df['timestamp'] <= end_datetime)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning


    if filtered_df.empty:
        return go.Figure().update_layout(title=f"No data available for Location Index {selected_location_idx} in the selected date range")

    # Create figure
    fig = go.Figure()
    # Find the display label corresponding to the selected index using the location_map
    location_label = location_map.get(selected_location_idx, f"Location Index {selected_location_idx}")

    # Add traces based on checklist selection
    if 'observed' in selected_data:
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['observed'], mode='lines', name='Observed PM2.5', line=dict(color='black')))
    if 'predicted_net' in selected_data:
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['predicted_net'], mode='lines', name='Predicted Net (pm25gnn)', line=dict(color='red')))
    if 'predicted_ambient' in selected_data:
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['predicted_ambient'], mode='lines', name='Predicted Ambient (pm25gnn-ambient)', line=dict(color='blue', dash='dash')))
    if 'fire_specific' in selected_data:
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
    # For Colab, use app.run(mode='inline', port=8050) after changing app initialization
    app.run(debug=True, host='0.0.0.0', port=8050)
