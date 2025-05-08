# analytics.py
import dash
from dash import html, dcc, callback, Input, Output
from utils import get_csv_from_gcs
from Nav_bar import create_navbar

dash.register_page(__name__, path="/analytics", name="Findings")

layout = html.Div([
    # Hero Section wrapped in a blue box
    html.Section(
        html.Div([
            html.H1("Analytics Dashboard"),
            html.P("Explore the data and uncover insights with interactive visualizations.", className="lead")
        ], className="container"),
        className="hero hero--blue"  # Apply the blue box styling
    ),

    # Metric cards if any…
    html.Section(
        html.Div([
            # e.g. html.Div([...], className="metric"), …
        ], className="metrics container")
    ),

    # Correlation Heatmaps Section
    html.Section(
        html.Div([
            html.H2("Correlation Heatmaps", style={"textAlign": "center"}),
            html.Div([
                html.Iframe(
                    src="/assets/corre_maps/corr_heatmap_Fresno_2021-05-01_to_2021-12-30.html",
                    style={"width": "100%", "height": "600px", "border": "none"}
                ),
                html.Iframe(
                    src="/assets/corre_maps/corr_heatmap_Glendora_2021-05-01_to_2021-12-30.html",
                    style={"width": "100%", "height": "600px", "border": "none", "marginTop": "20px"}
                ),
                # Add more iframes for other heatmaps as needed
            ], className="container"),
        ], className="container"),
    ),
])