# home.py
import dash
from dash import html
from Nav_bar import create_navbar

dash.register_page(__name__, path="/", name="Home")

layout = html.Div([

    # Hero / Intro
    html.Section(
      html.Div([
        html.H1("Smoke Signals: Time Series Forecasting of PM2.5 Amid California Wildfires"),
        html.P(
          "Using Historical PM2.5 and Meteorological data to predict PM2.5 levels during California's wildfire seasons. This project explores PM2.5 data to understand the effects on air quality.",
          className="lead"
        ),
      ], className="container"),
      className="hero hero--blue"  # Apply the new blue card style
    ),

    # Yellow Card Section
    html.Section(
      html.Div([
        html.Div([
          html.H3("What is PM2.5?"),
          html.P(
            "PM2.5 refers to fine particulate matter that is 2.5 micrometers or smaller in diameter. These particles can pose serious health risks when inhaled."
          ),
        ], className="card card--yellow"),
      ], className="container"),
    ),

    # PM2.5 Health Effects Section
    html.Section(
      html.Div([
        html.H2("Health Effects of PM2.5", style={"textAlign": "center"}),
        html.Div([
          html.Div([
            html.Img(src="/assets/pm25healtheffects.webp", alt="Health Effect 1"),
          ], className="card"),
        ], className="cards-grid"),
      ], className="container"),
    ),

    # Red Card Section
    html.Section(
      html.Div([
        html.Div([
          html.H3("What role do wildfires play in PM2.5 pollution?"),
          html.P(
            "Wildfires release large amounts of PM2.5 into the atmosphere, significantly impacting air quality and public health. " \
            "Preparing for and managing wildfire smoke is crucial for minimizing these effects."
          ),
        ], className="card card--green"),
      ], className="container"),
    ),

    # Wildfire PM2.5 Emissions Section
    html.Section(
        html.Div([
            html.H2("Estimates of Wildfire PM2.5 Emissions", style={"textAlign": "center"}),
            html.Img(
                src="/assets/wildfirepm2.5_emissions.jpg",
                alt="Wildfire PM2.5 Emissions",
                style={"display": "block", "margin": "0 auto", "width": "100%", "height": "auto"}
            ),
            html.P(
                "Graph provided by California Air Resources Board.",
                style={"textAlign": "center", "marginTop": "10px", "fontStyle": "italic"}
            ),
        ], className="container"),
    ),

    # Embedded Map
    html.Section(
      html.Div([
        html.H2("Interactive Map of California"),
        html.Iframe(
          src="/assets/location_map.html",
          style={"width": "100%", "height": "600px", "border": "none"}
        ),
      ], className="container"),
    ),

   
    # Final CTA
    html.Section(
      html.Div([
        html.H2("Ready to Explore the Data?"),
        html.A("View Findings →", href="/analytics", className="btn btn-primary"),
        html.A("Research Objectives →", href="/proposal", className="btn btn-secondary")
      ], className="container"),
      className="hero hero--inverse"
    )
])
