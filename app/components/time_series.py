from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
from modules.data_loader import load_and_preprocess_data

# load data once
df, locations, location_map = load_and_preprocess_data()

def layout_time_series():
    return html.Section(
      html.Div([
        html.H2("PM2.5 Time Series Analysis", style={"textAlign": "center"}),
        html.Div([
          # site dropdown + date picker
          html.Div([
            html.Label("Site:", style={'marginRight':'10px'}),
            dcc.Dropdown(
              id='location-dropdown',
              options=locations,
              value=locations[0]['value'] if locations else None,
              clearable=False,
              style={'width':'250px'}
            )
          ], style={'display':'inline-block','marginRight':'20px'}),
          html.Div([
            html.Label("Date Range:"),
            dcc.DatePickerRange(
              id='date-picker-range',
              min_date_allowed=df['timestamp'].min().date(),
              max_date_allowed=df['timestamp'].max().date(),
              start_date=df['timestamp'].min().date(),
              end_date=df['timestamp'].max().date(),
              display_format='YYYY-MM-DD'
            )
          ], style={'display':'inline-block'})
        ], style={'textAlign':'center','marginBottom':'1rem'}),
        html.Div([
          html.Label("Show:", style={'marginRight':'10px'}),
          dcc.Checklist(
            id='data-checklist',
            options=[
              {'label':'Observed','value':'observed'},
              {'label':'Predicted Net','value':'predicted_net'},
              {'label':'Predicted Ambient','value':'predicted_ambient'},
              {'label':'Fire-specific','value':'fire_specific'},
            ],
            value=['observed','predicted_net'],
            inline=True,
            labelStyle={'marginRight':'15px'}
          )
        ], style={'textAlign':'center','marginBottom':'1rem'}),
        dcc.Loading(dcc.Graph(id='pm25-time-series-graph'))
      ], className="container"),
      className="container"
    )

# register callback onto whichever Dash app is running
@callback(
    Output('pm25-time-series-graph','figure'),
    Input('location-dropdown','value'),
    Input('date-picker-range','start_date'),
    Input('date-picker-range','end_date'),
    Input('data-checklist','value'),
)
def update_ts(site, start, end, series):
    if not site:
        return go.Figure().update_layout(title="Select a site")
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    df2 = df[(df.location_id==site)&(df.timestamp>=start)&(df.timestamp<=end)]
    if df2.empty:
        return go.Figure().update_layout(title="No data")
    fig=go.Figure()
    label = location_map.get(site, site)
    if 'observed' in series:
        fig.add_trace(go.Scatter(x=df2.timestamp,y=df2.observed,mode='lines', name='Observed'))
    if 'predicted_net' in series:
        fig.add_trace(go.Scatter(x=df2.timestamp,y=df2.predicted_net,mode='lines',name='Predicted Net'))
    if 'predicted_ambient' in series:
        fig.add_trace(go.Scatter(x=df2.timestamp,y=df2.predicted_ambient,mode='lines',name='Pred Ambient'))
    if 'fire_specific' in series:
        fig.add_trace(go.Scatter(x=df2.timestamp,y=df2.fire_specific,mode='lines',name='Fire-specific'))
    fig.update_layout(title=f"PM2.5 @ {label}", xaxis_title="Time", yaxis_title="µg/m³",hovermode='x')
    return fig
