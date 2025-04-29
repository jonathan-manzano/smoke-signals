import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, name="Proposal", path="/proposal")

layout = dbc.Container(
    [
        # Title Section
        html.H1("Project Proposal", className="text-center my-4"),
        
        # Project Summary Section
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Project Summary", className="card-title"),
                    html.P(
                        "This project aims to create reliable time series forecasting models to accurately predict hourly PM2.5 levels during California's wildfire seasons. "
                        "By analyzing historical air quality data, meteorological conditions, and wildfire events, we aim to provide actionable insights and improve early warning systems.",
                        className="card-text",
                    ),
                ]
            ),
            className="mb-4 shadow-sm",
        ),

        # Objectives Section
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Objectives", className="card-title"),
                    dbc.ListGroup(
                        [
                            dbc.ListGroupItem("Analyze seasonal and temporal trends in PM2.5 concentrations."),
                            dbc.ListGroupItem("Identify key meteorological and wildfire-related drivers of PM2.5 levels."),
                            dbc.ListGroupItem("Develop advanced forecasting models for early warning systems."),
                        ]
                    ),
                ]
            ),
            className="mb-4 shadow-sm",
        ),

        # Broader Impacts Section
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Broader Impacts", className="card-title"),
                    dbc.ListGroup(
                        [
                            dbc.ListGroupItem("Provide accurate PM2.5 forecasts for sensitive communities."),
                            dbc.ListGroupItem("Influence policies on land management and fire prevention strategies."),
                            dbc.ListGroupItem("Highlight the impact of wildfires on air quality."),
                        ]
                    ),
                ]
            ),
            className="mb-4 shadow-sm",
        ),

        # Data Sources Section
        dbc.Card(
            dbc.CardBody(
                [
                    html.H2("Data Sources", className="card-title"),
                    dbc.ListGroup(
                        [
                            dbc.ListGroupItem("United States Environmental Protection Agency (EPA)"),
                            dbc.ListGroupItem("California Air Resources Board (AQMIS)"),
                            dbc.ListGroupItem("ERA5 Reanalysis (ECMWF)"),
                        ]
                    ),
                ]
            ),
            className="mb-4 shadow-sm",
        ),
    ],
    className="p-4",
)