from app.data_loader import load_and_preprocess_data
import dash
from dash import dcc, html, Input, Output
from datetime import datetime
import plotly.graph_objects as go

# --- Load and preprocess data ---
df, locations, location_map = load_and_preprocess_data()

# --- Dash App Initialization ---
app = dash.Dash(__name__)  # Use this for local running
server = app.server  # Expose server for deployment

# --- App Layout ---
app.layout = html.Div([
    html.H1("PM2.5 Time Series Comparison and Fire Contribution"),

    html.Div([
        html.Label("Select Monitoring Site:", style={'marginRight': '10px'}),
        dcc.Dropdown(
            id='location-dropdown',
            options=locations,
            value=locations[0]['value'] if locations else None,
            clearable=False,
            style={'width': '300px', 'display': 'inline-block', 'marginRight': '20px'}
        ),

        html.Label("Select Date Range:", style={'marginRight': '10px'}),
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=df['timestamp'].min().date(),
            max_date_allowed=df['timestamp'].max().date(),
            start_date=df['timestamp'].min().date(),
            end_date=df['timestamp'].max().date(),
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
            value=['observed', 'predicted_net'],
            inline=True,
            labelStyle={'display': 'inline-block', 'marginRight': '15px'}
        )
    ], style={'marginBottom': '20px'}),

    dcc.Loading(
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
def update_graph(selected_location_idx, start_date, end_date, selected_data):
    if selected_location_idx is None:
        return go.Figure().update_layout(title="Please select a location")

    # Filter DataFrame based on selections
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)

    filtered_df = df[
        (df['location_id'] == selected_location_idx) &
        (df['timestamp'] >= start_datetime) &
        (df['timestamp'] <= end_datetime)
    ].copy()

    if filtered_df.empty:
        return go.Figure().update_layout(title=f"No data available for Location Index {selected_location_idx} in the selected date range")

    # Create figure
    fig = go.Figure()
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
        yaxis_title='PM2.5 Concentration (µg/m³)',
        legend_title='Data Series',
        hovermode='x unified',
        xaxis_range=[start_datetime, end_datetime]
    )

    fig.update_xaxes(rangeslider_visible=True)

    return fig

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Dash server...")
    app.run(debug=True, host='0.0.0.0', port=8050)
