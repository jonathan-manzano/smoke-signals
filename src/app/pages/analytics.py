import dash
from dash import html, dcc, dash_table, callback, Input, Output
import plotly.express as px
import pandas as pd

# Register the page with a custom title
dash.register_page(__name__, name="Analytics")

# Sample dataset (replace with actual data if needed)
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv")

# Define the analytics page layout
layout = html.Div(
    [
        html.H2("Analytics Page"),
        dcc.RadioItems(
            options=["pop", "lifeExp", "gdpPercap"],
            value="lifeExp",
            id="analytics-radio-item",
        ),
        dcc.Graph(id="analytics-graph"),
    ]
)

# Callback to update the graph based on selected radio item
@callback(
    Output("analytics-graph", "figure"),
    Input("analytics-radio-item", "value"),
)
def update_graph(col_chosen):
    fig = px.histogram(df, x="continent", y=col_chosen, histfunc="avg")
    return fig