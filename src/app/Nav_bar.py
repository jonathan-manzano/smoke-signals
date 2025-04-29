# Nav_bar.py
import dash_bootstrap_components as dbc
from dash import html

JUNIPER_GREEN = '#3A5311'  # Juniper Green color
WHITE = '#FFFFFF'

def create_navbar(project_name, background_fill=JUNIPER_GREEN, text_color=WHITE):
    return dbc.Navbar(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand(
                            "Smoke Signals", href="/", className="navbar-brand", style={"color": text_color, "fontSize": "24px"}
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.Div(
                            project_name, className="navbar-title", style={"color": text_color, "fontSize": "18px"}
                        ),
                        width="auto",
                        className="mx-auto",
                    ),
                    dbc.Col(
                        dbc.DropdownMenu(
                            nav=True,
                            in_navbar=True,
                            label="Menu",
                            children=[
                                dbc.DropdownMenuItem("Home", href="/"),
                                dbc.DropdownMenuItem(divider=True),
                                dbc.DropdownMenuItem("Analytics", href="/analytics"),
                                dbc.DropdownMenuItem("Proposal", href="/proposal"),
                            ],
                            style={ "backgroundColor": WHITE, "color": JUNIPER_GREEN},  # Menu button color
                        ),
                        width="auto",
                    ),
                ],
                align="center",
                className="w-100",
            )
        ),
        color=background_fill,  # Juniper Green for nav bar
        dark=False,  # Set to False for light text
        className="nav",
    )

