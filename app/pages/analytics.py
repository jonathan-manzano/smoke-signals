# analytics.py

import dash
from dash import html
from components.time_series import layout_time_series

dash.register_page(__name__, path="/analytics", name="Findings")

layout = html.Div([

    # Navbar + Hero (no TS code here!)
    html.Section(
        html.Div([
            html.H1("Analytics Dashboard"),
            html.P("Interactive charts to explore your data", className="lead")
        ], className="container"),
        className="hero"
    ),

    # ← embeds all the time series controls & graph
    layout_time_series()

])
