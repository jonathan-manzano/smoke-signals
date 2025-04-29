import os
from datetime import datetime
from pathlib import Path

import dash
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dash import dcc, html, Input, Output, State
from sklearn.metrics import r2_score # For R-squared calculation

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


# --- Constants (Update if necessary based on your training config) ---
PRED_LEN = 48  # Example: Prediction length used in training (Set to your actual value)

# --- Data Loading and Preprocessing (Reusing logic from previous example) ---
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
predict_net_data_raw = load_npy_data(PREDICT_NET_FILE)
predict_ambient_data_raw = load_npy_data(PREDICT_AMBIENT_FILE)
label_data_raw = load_npy_data(LABEL_FILE)
time_data_raw = load_npy_data(TIME_FILE)

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

# --- Data Structure Adaptation (CRITICAL STEP - Same as previous example) ---
print("Adapting data structure (Verify this section based on your .npy files)...")
try:
    # Step 0: Handle Shape Mismatch
    min_len = min(predict_net_data_raw.shape[0],
                  predict_ambient_data_raw.shape[0],
                  label_data_raw.shape[0],
                  time_data_raw.shape[0])
    predict_net_data_trunc = predict_net_data_raw[:min_len]
    predict_ambient_data_trunc = predict_ambient_data_raw[:min_len]
    label_data_trunc = label_data_raw[:min_len]
    time_data_trunc = time_data_raw[:min_len]

    # Get number of locations
    num_locations_data = predict_net_data_raw.shape[-2]
    if num_locations_data != len(locations):
         print(f"Warning: Location count mismatch! Data has {num_locations_data}, file has {len(locations)}. Using data count.")
         locations = [{'label': f"Location Index {i}", 'value': i} for i in range(num_locations_data)]

    # Step 1: Extract Prediction Horizon & Reshape
    if len(predict_net_data_trunc.shape) == 4:
        sequence_length = predict_net_data_trunc.shape[1]
        if PRED_LEN > sequence_length: raise ValueError("PRED_LEN > sequence length")
        predict_net_pred = predict_net_data_trunc[:, -PRED_LEN:, :, 0]
        predict_ambient_pred = predict_ambient_data_trunc[:, -PRED_LEN:, :, 0]
        label_pred = label_data_trunc[:, -PRED_LEN:, :, 0]
        num_samples = predict_net_pred.shape[0]
        num_locs = predict_net_pred.shape[2]
        predict_net_actual = predict_net_pred.reshape(num_samples * PRED_LEN, num_locs)
        predict_ambient_actual = predict_ambient_pred.reshape(num_samples * PRED_LEN, num_locs)
        label_actual = label_pred.reshape(num_samples * PRED_LEN, num_locs)
        if time_data_trunc.shape[0] == num_samples:
             time_data_processed = np.repeat(time_data_trunc, PRED_LEN)
        else: raise ValueError("Time data shape mismatch")
    elif len(predict_net_data_trunc.shape) == 2: # Assumes (time, location)
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
    num_locs_final = predict_net_actual.shape[1]
    df_list = []
    for loc_idx in range(num_locs_final):
        loc_df = pd.DataFrame({
            'timestamp': time_data_dt,
            'location_id': loc_idx,
            'observed': label_actual[:, loc_idx],
            'predicted_net': predict_net_actual[:, loc_idx],
            'predicted_ambient': predict_ambient_actual[:, loc_idx]
        })
        df_list.append(loc_df)
    df = pd.concat(df_list, ignore_index=True)
    # Ensure timestamp is datetime type
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("Data preprocessing complete.")
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
    html.H1("Observed vs. Predicted PM2.5 Scatter Plot"),

    html.Div([
        # Model Selection
        html.Div([
            html.Label("Select Model:"),
            dcc.RadioItems(
                id='model-selector',
                options=[
                    {'label': 'Net (pm25gnn)', 'value': 'predicted_net'},
                    {'label': 'Ambient (pm25gnn-ambient)', 'value': 'predicted_ambient'},
                ],
                value='predicted_net', # Default model
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'marginBottom': '15px'}),

        # Location Filter
        html.Div([
            html.Label("Filter by Location(s): (Optional - leave blank for all)"),
            dcc.Dropdown(
                id='location-filter-dropdown',
                options=locations,
                value=[], # Default to no filter (all locations)
                multi=True, # Allow multiple selections
                placeholder="Select locations to include..."
            )
        ], style={'marginBottom': '15px'}),

        # Date Filter
        html.Div([
            html.Label("Filter by Date Range:"),
            dcc.DatePickerRange(
                id='scatter-date-picker-range',
                min_date_allowed=time_data_dt.min().date(),
                max_date_allowed=time_data_dt.max().date(),
                start_date=time_data_dt.min().date(),
                end_date=time_data_dt.max().date(),
                display_format='YYYY-MM-DD',
            )
        ], style={'marginBottom': '15px'}),

        # Log Scale Toggle
        html.Div([
            html.Label("Axis Scale:"),
            dcc.RadioItems(
                id='scale-selector',
                options=[
                    {'label': 'Linear', 'value': 'linear'},
                    {'label': 'Logarithmic', 'value': 'log'},
                ],
                value='linear', # Default scale
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'marginBottom': '15px'}),

    ]),

    # Metrics Display
    html.Div(id='metrics-display', style={'marginTop': '10px', 'fontWeight': 'bold'}),

    # Scatter Plot
    dcc.Loading(
        id="loading-scatter-graph",
        type="circle",
        children=dcc.Graph(id='observed-vs-predicted-scatter')
    )
])

# --- App Callback ---
@app.callback(
    Output('observed-vs-predicted-scatter', 'figure'),
    Output('metrics-display', 'children'),
    Input('model-selector', 'value'),
    Input('location-filter-dropdown', 'value'),
    Input('scatter-date-picker-range', 'start_date'),
    Input('scatter-date-picker-range', 'end_date'),
    Input('scale-selector', 'value')
)
def update_scatter_plot(selected_model_col, selected_locations, start_date, end_date, selected_scale):
    print(f"Updating scatter plot for model: {selected_model_col}, locations: {selected_locations}, date: {start_date}-{end_date}, scale: {selected_scale}")

    # Filter DataFrame based on selections
    filtered_df = df.copy() # Start with the full dataframe

    # Apply location filter if any locations are selected
    if selected_locations:
        filtered_df = filtered_df[filtered_df['location_id'].isin(selected_locations)]

    # Apply date filter
    if start_date and end_date:
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
        filtered_df = filtered_df[
            (filtered_df['timestamp'] >= start_datetime) &
            (filtered_df['timestamp'] <= end_datetime)
        ]

    if filtered_df.empty:
        return go.Figure().update_layout(title="No data available for the selected filters"), "No data points to calculate metrics."

    # Extract observed and selected predicted values
    observed_values = filtered_df['observed']
    predicted_values = filtered_df[selected_model_col]

    # Remove NaN or infinite values if any (important for R2 and log scale)
    valid_mask = np.isfinite(observed_values) & np.isfinite(predicted_values)
    if selected_scale == 'log':
        # Also remove non-positive values for log scale
        valid_mask &= (observed_values > 0) & (predicted_values > 0)

    observed_values = observed_values[valid_mask]
    predicted_values = predicted_values[valid_mask]

    if len(observed_values) < 2: # Need at least 2 points for R2
         return go.Figure().update_layout(title="Not enough valid data points for the selected filters/scale"), "Not enough valid data points to calculate metrics."

    # Calculate R-squared
    try:
        r2 = r2_score(observed_values, predicted_values)
        metrics_text = f"R² Score: {r2:.4f} (based on {len(observed_values):,} filtered points)"
    except Exception as e:
        print(f"Error calculating R2 score: {e}")
        metrics_text = "Could not calculate R² score."
        r2 = None

    # Create Scatter Plot (using Scattergl for performance with many points)
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=observed_values,
        y=predicted_values,
        mode='markers',
        name='Data Points',
        marker=dict(
            size=5,
            opacity=0.6,
            # Optional: Color by density or another variable if desired
            # color='blue'
        )
    ))

    # Add 1:1 reference line
    min_val = min(observed_values.min(), predicted_values.min()) if len(observed_values)>0 else 0
    max_val = max(observed_values.max(), predicted_values.max()) if len(observed_values)>0 else 1
    # Adjust range slightly for visibility if linear scale
    if selected_scale == 'linear':
        plot_min = min_val - 0.05 * (max_val - min_val)
        plot_max = max_val + 0.05 * (max_val - min_val)
    else: # For log scale, use the actual min/max positive values
        plot_min = max(min_val, 1e-1) # Avoid zero or negative for log
        plot_max = max_val

    fig.add_trace(go.Scattergl(
        x=[plot_min, plot_max],
        y=[plot_min, plot_max],
        mode='lines',
        name='1:1 Line',
        line=dict(color='grey', dash='dash')
    ))

    # Update layout
    model_name = "Net (pm25gnn)" if selected_model_col == 'predicted_net' else "Ambient (pm25gnn-ambient)"
    fig.update_layout(
        title=f'Observed vs. Predicted PM2.5 ({model_name})',
        xaxis_title='Observed PM2.5 (µg/m³)',
        yaxis_title=f'Predicted PM2.5 ({model_name}) (µg/m³)',
        xaxis_type=selected_scale, # Set scale type
        yaxis_type=selected_scale, # Set scale type
        # Ensure axes ranges match for 1:1 comparison, especially if linear
        xaxis_range=[plot_min, plot_max] if selected_scale=='linear' else None, # Let log scale adjust automatically
        yaxis_range=[plot_min, plot_max] if selected_scale=='linear' else None,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    # Ensure square aspect ratio for better visual comparison against 1:1 line
    fig.update_yaxes(scaleanchor="x", scaleratio=1)


    return fig, metrics_text

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Dash server...")
    # For Colab, use app.run(mode='inline', port=8050) after changing app initialization
    app.run(debug=True, host='0.0.0.0', port=8050)

