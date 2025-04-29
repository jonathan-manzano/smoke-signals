import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/", name="Home")

layout = dbc.Container(
    [
        html.H1("Welcome to Smoke Signals", className="text-center my-4"),
        html.P(
            "This project explores PM2.5 data to understand the impact of wildfires on air quality and develop a time series forecasting model.",
            className="lead text-center",
        ),
        html.Hr(),

        # Section: What is PM2.5 and its health effects wrapped in a card
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    [
                        # Left Column: Description
                        dbc.Col(
                            [
                                html.H2("What is Particulate Matter 2.5?"),
                                html.P(
                                    "Particulate Matter 2.5 (PM2.5) refers to fine inhalable particles with diameters that are generally 2.5 micrometers and smaller. "
                                    "These particles are small enough to penetrate deep into the lungs and even enter the bloodstream."
                                ),
                                html.H3("Health Effects"),
                                html.P(
                                    "Exposure to PM2.5 can cause serious health problems, including respiratory and cardiovascular issues. "
                                    "It is particularly harmful to sensitive groups such as children, the elderly, and individuals with pre-existing health conditions."
                                ),
                            ],
                            width=6,
                        ),
                        # Right Column: Image
                        dbc.Col(
                            html.Img(
                                src="https://via.placeholder.com/400x300",  # Placeholder image
                                alt="Health effects of PM2.5",
                                className="img-fluid",
                            ),
                            width=6,
                        ),
                    ],
                    className="mt-4",
                )
            ),
            className="mb-4 shadow-sm",
        ),

        html.Hr(),

        # Section: Navigation Cards
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4("Explore Analytics", className="card-title"),
                                    html.P(
                                        "Dive into the data to uncover trends and insights about PM2.5 levels during wildfire seasons.",
                                        className="card-text",
                                    ),
                                    dbc.Button("Go to Analytics", href="/analytics", color="primary"),
                                ]
                            )
                        ],
                        className="mb-4",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4("View Proposal", className="card-title"),
                                    html.P(
                                        "Learn about the project's objectives, methods, and expected outcomes.",
                                        className="card-text",
                                    ),
                                    dbc.Button("View Proposal", href="/proposal", color="success"),
                                ]
                            )
                        ],
                        className="mb-4",
                    ),
                    width=6,
                ),
            ],
            className="mt-4",
        ),
    ],
    className="p-4",
)