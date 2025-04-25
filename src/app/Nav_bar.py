# Nav_bar.py
import dash_bootstrap_components as dbc
from dash import html

FOREST_GREEN = '#228B22'
WHITE = '#F7F7E6'

def create_navbar(project_name, background_fill=WHITE, text_color=FOREST_GREEN):
    return dbc.Navbar(
        dbc.Container(
            dbc.Row(
                [
                  dbc.Col(dbc.NavbarBrand("Smoke Signals", href="/", style={"color": text_color}), width="auto"),
                  dbc.Col(html.Div(project_name, className="navbar-title"), width="auto", className="mx-auto"),
                  dbc.Col(
                    dbc.DropdownMenu(
                      nav=True, in_navbar=True, label="Menu", children=[
                        dbc.DropdownMenuItem("Home", href="/"),
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem("Analytics", href="/analytics"),
                        dbc.DropdownMenuItem("Proposal", href="/proposal"),
                      ]
                    ),
                    width="auto"
                  )
                ], align="center", className="w-100"
            )
        ),
        color=background_fill,
        dark=True,
        className="nav"   # ← optional hook if you want to target it explicitly
    )
