import dash
from dash import html

dash.register_page(__name__, path="/", name="Home")

layout = html.Div(
    [
        html.P(
            html.B(
                "Welcome to Smoke Signals. In this project, we aim to predict PM2.5 levels in California using time series forecasting techniques. "
                "Our goal is to explore the ways in how our climate and wildfire events affect each other."
            )
        ),
        html.H2("What is Particulate Matter 2.5?"),
        html.P(
            html.B(
                "Particulate Matter 2.5 (PM2.5) refers to fine inhalable particles with diameters that are generally 2.5 micrometers and smaller. "
                "These particles are small enough to penetrate deep into the lungs and even enter the bloodstream, posing significant health risks. "
                "After wildfires, PM2.5 levels can spike due to the release of smoke and other pollutants into the air. "
            )
        ),
        html.Div(
            html.Img(
                src="https://pub.mdpi-res.com/nanomaterials/nanomaterials-12-02656/article_deploy/html/images/nanomaterials-12-02656-g002.png?1659510056",  # Replace with the actual image URL
                alt="Particulate Matter 2.5 Illustration",
                style={"width": "40%", "margin": "20px auto", "display": "block"},
            )
        ),
        html.Iframe(
            srcDoc=open("location_map.html", "r").read(),  # Load the saved map HTML
            width="60%",
            height="500px",
        ),
    ]
)