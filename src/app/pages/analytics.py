import dash
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import pandas as pd
from utils import get_csv_from_gcs
import dash_bootstrap_components as dbc

dash.register_page(__name__, name="Analytics")

# Load the dataset
df = get_csv_from_gcs("smoke-signal-bucket", "gapminder2007.csv")

layout = dbc.Container(
    [
        html.H1("Analytics Dashboard", className="text-center my-4"),
        html.P(
            "Analyze PM2.5 data trends and explore the impact of wildfires on air quality.",
            className="lead text-center",
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    dcc.RadioItems(
                        options=["pop", "lifeExp", "gdpPercap"],
                        value="lifeExp",
                        id="analytics-radio-item",
                        className="mb-4",
                    ),
                    width=4,
                ),
                dbc.Col(
                    dcc.Graph(id="analytics-graph"),
                    width=8,
                ),
            ]
        ),
    ],
    className="p-4",
)

@callback(
    Output("analytics-graph", "figure"),
    Input("analytics-radio-item", "value"),
)
def update_graph(col_chosen):
    fig = px.histogram(df, x="continent", y=col_chosen, histfunc="avg", title=f"Average {col_chosen} by Continent")
    return fig