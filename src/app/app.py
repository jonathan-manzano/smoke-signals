# app.py
import os
import dash
from dash import Dash, html
import dash_bootstrap_components as dbc
from utils import get_csv_from_gcs
from Nav_bar import create_navbar

# Initialize the Dash app with Bootstrap and custom styles
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server

# Define the app layout
app.layout = html.Div(
    className="dashboard-frame",
    children=[
        # Header Section
        html.Header(
            html.H1(
                "Smoke Signals: Time Series Forecasting of PM2.5 Amid California Wildfires",
                className="header text-center my-4",
            ),
            className="bg-dark text-white py-3",
        ),

        # Navigation Bar
        create_navbar(project_name="Exploring PM2.5 Data"),

        # Main Content Section
        html.Main(
            html.Div(dash.page_container, className="container mt-4"),
            className="main",
        ),

        # Footer Section
        html.Footer(
            html.P(
                "© 2023 Smoke Signals. All rights reserved.",
                className="text-center text-muted my-3",
            ),
            className="footer bg-light py-3",
        ),
    ],
)

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True)
