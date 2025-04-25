# app.py
import os
import dash
from dash import Dash, html
import dash_bootstrap_components as dbc
from utils import get_csv_from_gcs
from Nav_bar import create_navbar

# Initialize with Bootstrap + custom assets/styles.css
app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server

app.layout = html.Div(className="dashboard-frame", children=[
    html.Header(
        html.H1("Smoke Signals: Time Series Forecasting of PM2.5 Amid California Wildfires"),
        className="header"
    ),

    # This will render into <nav>…</nav> if you provided assets/index.html,
    # or just sit here if not.
    create_navbar(
        project_name=""
    ),

    html.Main(
        html.Div(dash.page_container),
        className="main"
    ),

    html.Footer(
        html.P("© 2023 Smoke Signals. All rights reserved."),
        className="footer"
    ),
])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True)
