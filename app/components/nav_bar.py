# Nav_bar.py
from dash import html


def create_navbar(project_name="Research Dashboard"):
    return html.Nav(
        html.Div([
            # Brand text comes from project_name
            html.A(project_name, href="/", className="brand"),

            # Main links
            html.Ul([
                html.Li(html.A("Home", href="/")),
                html.Li(html.A("Objectives", href="/proposal")),
                html.Li(html.A("Findings", href="/analytics")),
            ], className="nav-links"),

        ], className="nav-inner container"),
        className="site-nav"
    )