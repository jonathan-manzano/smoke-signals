import os
from io import StringIO

import pandas as pd
import plotly.express as px
from dash import Dash, html, dash_table, dcc, callback, Output, Input
from google.cloud import storage


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

# Initialize the Dash app.
app = Dash(__name__)
server = app.server  # Expose the Flask server for WSGI.

# Cloud Storage demo: Try to load a second dataset.
BUCKET_NAME = os.environ.get("BUCKET_NAME")
if BUCKET_NAME:
    try:
        df2 = get_csv_from_gcs(BUCKET_NAME, "customers-100-simple.csv")
    except Exception as e:
        print(f"Error loading CSV from GCS: {e}")
        df2 = pd.DataFrame()  # Fallback to an empty DataFrame.
else:
    print("BUCKET_NAME environment variable not set; skipping Cloud Storage demo.")
    df2 = pd.DataFrame()

# Define the app layout.
app.layout = html.Div(
    [
        html.Div("My First App with Data, Graph, and Controls"),
        html.Hr(),
        dcc.RadioItems(
            options=["pop", "lifeExp", "gdpPercap"],
            value="lifeExp",
            id="controls-and-radio-item",
        ),
        dash_table.DataTable(data=df.to_dict("records"), page_size=6),
        dcc.Graph(figure={}, id="controls-and-graph"),
        html.Hr(),
        # A second table for the Google Cloud Storage demo.
        dash_table.DataTable(data=df2.to_dict("records"), page_size=6),
    ]
)


# Callback to update the graph based on selected radio item.
@callback(
    Output("controls-and-graph", "figure"),
    Input("controls-and-radio-item", "value"),
)
def update_graph(col_chosen):
    fig = px.histogram(df, x="continent", y=col_chosen, histfunc="avg")
    return fig


# Run the app using the proper host and port configuration.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run_server(debug=True, host="0.0.0.0", port=port)
