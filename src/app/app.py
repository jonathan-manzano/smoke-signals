import os
from io import StringIO

import pandas as pd
import plotly.express as px
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from google.cloud import storage
import dash
from utils import get_csv_from_gcs

# Initialize the Dash app with multipage support
app = Dash(__name__, use_pages=True)
server = app.server  # Expose the Flask server for WSGI.

# BUCKET_NAME = os.environ.get("BUCKET_NAME")
# if BUCKET_NAME:
#     try:
#         df2 = get_csv_from_gcs(BUCKET_NAME, "gapminder2007.csv")
#     except Exception as e:
#         print(f"Error loading CSV from GCS: {e}")
#         df2 = pd.DataFrame()  # Fallback to an empty DataFrame.
# else:
#     print("BUCKET_NAME environment variable not set; skipping Cloud Storage demo.")
#     df2 = pd.DataFrame()

# Define the app layout
app.layout = html.Div(
    [
        # Header
        html.Header(
            html.H1("Smoke Signals:Time Series Forecasting of PM2.5 Amid California Wildfires", style={"margin": "0", "padding": "20px", "text-align": "center", "background-color": "#ac7c34", "color": "white"}),
        ),
        # Navigation Bar
        html.Nav(
            [
                html.Div(
                    html.A(page["name"], href=page["relative_path"], style={"margin-right": "15px", "font-weight": "bold", "color": "white", "text-decoration": "none"}),
                    style={"display": "inline-block"}
                ) for page in dash.page_registry.values()
            ],
            style={"background-color": "#444", "padding": "10px", "text-align": "center"}
        ),
        # Page Content
        html.Div(
            dash.page_container,
            style={"padding": "20px"}
        ),
    ]
)

# Run the app using the proper host and port configuration.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True)
