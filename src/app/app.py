import os
from io import StringIO

import pandas as pd
import plotly.express as px
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from google.cloud import storage
import dash

# Initialize the Dash app with multipage support
app = Dash(__name__, use_pages=True)
server = app.server  # Expose the Flask server for WSGI.

def get_csv_from_gcs(bucket_name, source_blob_name):
    """
    Download a CSV file from Google Cloud Storage and return a DataFrame.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_text()
    return pd.read_csv(StringIO(data))


# Load primary dataset from an external URL.
df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv"
)

# Cloud Storage demo: Try to load a second dataset.
BUCKET_NAME = os.environ.get("smoke-signal-bucket")
if BUCKET_NAME:
    try:
        df2 = get_csv_from_gcs(BUCKET_NAME, "customers-100-simple.csv")
    except Exception as e:
        print(f"Error loading CSV from GCS: {e}")
        df2 = pd.DataFrame()  # Fallback to an empty DataFrame.
else:
    print("BUCKET_NAME environment variable not set; skipping Cloud Storage demo.")
    df2 = pd.DataFrame()

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
