import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime
import os # Used for checking file existence

# --- Configuration: Replace with your actual file paths ---
# PREDICT_NET_FILE = '/path/to/your/predict_pm25gnn.npy' # Not strictly needed for this heatmap
# PREDICT_AMBIENT_FILE = '/path/to/your/predict_pm25gnn_ambient.npy' # Not strictly needed
LABEL_FILE = '/path/to/your/label.npy' # Essential for observed PM2.5
TIME_FILE = '/path/to/your/time.npy' # Essential for time filtering
LOCATIONS_FILE = '/path/to/your/locations.txt' # Essential for labels

# --- Use placeholder paths if you haven't set them above ---
# PREDICT_NET_FILE = os.environ.get('PREDICT_NET_FILE', 'predict_pm25gnn.npy')
# PREDICT_AMBIENT_FILE = os.environ.get('PREDICT_AMBIENT_FILE', 'predict_pm25gnn_ambient.npy')
LABEL_FILE = os.environ.get('LABEL_FILE', 'label.npy')
TIME_FILE = os.environ.get('TIME_FILE', 'time.npy')
LOCATIONS_FILE = os.environ.get('LOCATIONS_FILE', 'locations.txt')


# --- Constants (Update if necessary based on your training config) ---
PRED_LEN = 48  # Example: Prediction length used in training (Set to your actual value)

# --- Data Loading and Preprocessing (Reusing logic) ---
print("Loading data...")

# --- Helper function to load data with error handling ---
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
# Only label and time are strictly required for this heatmap
label_data_raw = load_npy_data(LABEL_FILE)
time_data_raw = load_npy_data(TIME_FILE)

# --- Load location data ---
locations = []
location_map = {} # To map ID to Name/Label for heatmap axes
if os.path.exists(LOCATIONS_FILE):
    try:
        with open(LOCATIONS_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    loc_id = int(parts[0])
                    loc_name = parts[1]
                    label = f"{loc_name} ({loc_id})"
                    locations.append({'label': label, 'value': loc_id})
                    location_map[loc_id] = label # Store mapping for later use
        print(f"Loaded {len(locations)} locations from {LOCATIONS_FILE}")
    except Exception as e:
        print(f"Error reading {LOCATIONS_FILE}: {e}")
else:
    print(f"Error: Locations file not found at {LOCATIONS_FILE}")

# --- Exit if essential data failed to load ---
if label_data_raw is None or time_data_raw is None or not locations:
    print("Essential data files (label, time, locations) could not be loaded. Exiting.")
    exit()

# --- Data Structure Adaptation (Focus on Label and Time) ---
print("Adapting data structure (Verify this section based on your .npy files)...")
try:
    # Step 0: Handle Potential Shape Mismatch (only need label and time)
    min_len = min(label_data_raw.shape[0], time_data_raw.shape[0])
    label_data_trunc = label_data_raw[:min_len]
    time_data_trunc = time_data_raw[:min_len]

    # Get number of locations
    num_locations_data = label_data_raw.shape[-2]
    if num_locations_data != len(locations):
         print(f"Warning: Location count mismatch! Data has {num_locations_data}, file has {len(locations)}. Using data count.")
         locations = [{'label': f"Loc {i}", 'value': i} for i in range(num_locations_data)]
         location_map = {i: f"Loc {i}" for i in range(num_locations_data)}


    # Step 1: Extract Relevant Horizon & Reshape Label Data
    if len(label_data_trunc.shape) == 4:
        sequence_length = label_data_trunc.shape[1]
        if PRED_LEN > sequence_length: raise ValueError("PRED_LEN > sequence length")
        # Extract the label part corresponding to predictions
        label_pred = label_data_trunc[:, -PRED_LEN:, :, 0]
        num_samples = label_pred.shape[0]
        num_locs = label_pred.shape[2]
        # Reshape to (time*pred_len, locations)
        label_actual = label_pred.reshape(num_samples * PRED_LEN, num_locs)

        # Process time data
        if time_data_trunc.shape[0] == num_samples:
             time_data_processed = np.repeat(time_data_trunc, PRED_LEN)
        else: raise ValueError("Time data shape mismatch")

    elif len(label_data_trunc.shape) == 2: # Assumes (time, location)
         label_actual = label_data_trunc
         time_data_processed = time_data_trunc
    else: raise ValueError("Unsupported label data shape")

    # Step 2: Process Timestamps
    if time_data_processed.shape[0] != label_actual.shape[0]:
         raise ValueError("Timestamp array shape incompatible")
    time_data_dt = pd.to_datetime(time_data_processed, unit='s')

    # Step 3: Create DataFrame (Optional but can be convenient)
    # We primarily need the label_actual numpy array and time_data_dt for correlation
    print("Data preprocessing complete.")
    print(f"Processed label data shape: {label_actual.shape}")
    print(f"Processed time data length: {len(time_data_dt)}")


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
    html.H1("PM2.5 Observed Values Correlation Heatmap"),

    html.Div([
         html.Label("Select Date Range for Correlation Calculation:"),
         dcc.DatePickerRange(
             id='heatmap-date-picker-range',
             min_date_allowed=time_data_dt.min().date(),
             max_date_allowed=time_data_dt.max().date(),
             start_date=time_data_dt.min().date(),
             end_date=time_data_dt.max().date(),
             display_format='YYYY-MM-DD',
         )
    ], style={'marginBottom': '20px'}),

    # Optional: Location subset selection (can be added if heatmap is too large)
    # html.Div([
    #     html.Label("Select Locations Subset (Optional):"),
    #     dcc.Dropdown(
    #         id='heatmap-location-subset',
    #         options=locations,
    #         value=[loc['value'] for loc in locations], # Default to all
    #         multi=True,
    #         placeholder="Select locations..."
    #     )
    # ], style={'marginBottom': '20px'}),

    dcc.Loading(
        id="loading-heatmap-graph",
        type="circle",
        children=dcc.Graph(id='correlation-heatmap', style={'height': '80vh'}) # Adjust height as needed
    )
])

# --- App Callback ---
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('heatmap-date-picker-range', 'start_date'),
    Input('heatmap-date-picker-range', 'end_date'),
    # Optional Input:
    # Input('heatmap-location-subset', 'value')
)
def update_heatmap(start_date, end_date): # Add selected_subset if using the dropdown
    print(f"Updating heatmap for date range: {start_date} to {end_date}")

    # Filter data based on date range
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)

    time_mask = (time_data_dt >= start_datetime) & (time_data_dt <= end_datetime)

    if not np.any(time_mask):
        return go.Figure().update_layout(title="No data available for the selected date range")

    # Get the observed data for the filtered time period
    filtered_labels = label_actual[time_mask, :]

    # Optional: Filter by location subset
    # if selected_subset:
    #     subset_indices = [loc['value'] for loc in locations if loc['value'] in selected_subset] # Get indices
    #     if not subset_indices:
    #          return go.Figure().update_layout(title="No locations selected in subset")
    #     filtered_labels = filtered_labels[:, subset_indices]
    #     axis_labels = [location_map[i] for i in subset_indices]
    # else:
    #     axis_labels = [location_map[i] for i in range(filtered_labels.shape[1])] # Use all labels

    # Use all locations for now
    axis_labels = [location_map.get(i, f"Loc {i}") for i in range(filtered_labels.shape[1])]

    # Check if enough data remains after filtering time
    if filtered_labels.shape[0] < 2:
        return go.Figure().update_layout(title="Not enough time points in the selected range to calculate correlation")

    # Calculate correlation matrix
    # np.corrcoef expects variables as rows, so transpose the data
    # Handle potential columns with zero variance (e.g., constant PM2.5) which cause NaN in corrcoef
    print(f"Calculating correlation for data shape: {filtered_labels.shape}")
    try:
        # Calculate correlation, ignoring columns with NaNs or zero variance temporarily
        valid_cols_mask = np.std(filtered_labels, axis=0) > 1e-6 # Check for non-zero standard deviation
        if not np.any(valid_cols_mask):
             return go.Figure().update_layout(title="No locations with varying data in selected range")

        valid_labels = filtered_labels[:, valid_cols_mask]
        corr_matrix_valid = np.corrcoef(valid_labels, rowvar=False) # Calculate on valid columns

        # Create full matrix filled with NaNs initially
        num_locs_total = filtered_labels.shape[1]
        corr_matrix = np.full((num_locs_total, num_locs_total), np.nan)

        # Fill in the calculated correlations for valid columns
        valid_indices = np.where(valid_cols_mask)[0]
        for i, row_idx in enumerate(valid_indices):
            for j, col_idx in enumerate(valid_indices):
                corr_matrix[row_idx, col_idx] = corr_matrix_valid[i, j]

        print(f"Correlation matrix calculated, shape: {corr_matrix.shape}")

    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return go.Figure().update_layout(title="Error calculating correlation matrix")


    # Create Heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=axis_labels,
        y=axis_labels,
        colorscale='RdBu', # Red-Blue colorscale is good for correlation (-1 to 1)
        zmin=-1, # Enforce range for correlation
        zmax=1,
        colorbar=dict(title='Pearson Corr.')
    ))

    fig.update_layout(
        title=f'Correlation of Observed PM2.5 Between Locations ({start_date} to {end_date})',
        xaxis_title='Location',
        yaxis_title='Location',
        xaxis_tickangle=-45, # Angle labels if they overlap
        # Adjust margins if labels are cut off
        margin=dict(l=150, r=50, b=150, t=50), # Increase left/bottom margins
        yaxis_autorange='reversed' # Often conventional for matrices
    )
    # Ensure square aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)


    return fig

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Dash server...")
    # For Colab, use app.run(mode='inline', port=8050) after changing app initialization
    app.run(debug=True, host='0.0.0.0', port=8050)
