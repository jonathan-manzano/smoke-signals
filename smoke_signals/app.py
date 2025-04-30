# app.py
import os
import dash
from dash import Dash, html
import dash_bootstrap_components as dbc
from app.Nav_bar import create_navbar
# import sys
# from pathlib import Path


# sys.path.append(str(Path(__file__).resolve().parents[1]))
# Initialize the Dash app with Bootstrap and custom styles
app = Dash(
    __name__,
    use_pages=True,
    pages_folder="app/pages",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server

# Define the app layout
app.layout = html.Div(
    className="dashboard-frame",
    children=[
        # Navbar Section
        create_navbar(),  # Move the navbar to the top

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
