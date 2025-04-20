import dash
from dash import html

dash.register_page(__name__, name="Proposal", path="/proposal")

layout = html.Div(
    [
        html.H1("Project Proposal"),
        html.H2("Project Summary"),
        html.P(
            "Creating reliable time series forecasting models to accurately anticipate hourly PM2.5 levels "
            "during California's wildfire seasons is the aim of this project. Our project will objectively "
            "investigate the ways in which certain climatic circumstances and wildfire events interact to affect "
            "air quality by combining meteorological data, records of wildfire incidents, and historical air quality measurements."
        ),
        html.H2("Our Method"),
        html.H3("Trend Analysis"),
        html.P(
            "During wildfire times, we will investigate seasonal trends, temporal trends, and anomalies in PM2.5 "
            "concentrations. This includes employing a two-step modeling procedure to differentiate between PM2.5 unique "
            "to wildfires and ambient pollution."
        ),
        html.H3("Finding the Main Drivers"),
        html.P(
            "We will ascertain how meteorological parameters, including wind speed, temperature inversions, humidity, "
            "and other atmospheric conditions, and wildfire occurrences contribute to the accumulation of PM2.5 by conducting a thorough analysis of these variables."
        ),
        html.H3("Predictive Modeling"),
        html.P(
            "In order to generate useful hourly PM2.5 forecasts for early warning systems, we will apply and evaluate forecasting models, "
            "such as multilayer perceptrons (MLP), graph neural networks (GNN), and long short-term memory (LSTM) networks. It is anticipated that these sophisticated "
            "spatiotemporal models will better represent intricate relationships than conventional methods."
        ),
        html.H2("Broader Impacts"),
        html.Ul(
            [
                html.Li("Providing an accurate forecast of PM2.5 level for sensitive communities"),
                html.Li("Use to influence local and national policies on tighter regulations on land management and fire prevention strategies"),
                html.Li("Showcasing how the increasing amount of wildfires is affecting our air quality"),
            ]
        ),
        html.H2("Data Sources"),
        html.Ul(
            [
                html.Li("United States Environmental Protection Agency (EPA)"),
                html.Li("California Air Resources Board - Air Quality and Meteorological Information System (AQMIS)"),
                html.Li("ERA5 Reanalysis (European Centre for Medium-Range Weather Forecasts)"),
            ]
        ),
        html.H2("Expected Major Findings"),
        html.Ul(
            [
                html.Li(
                    "Improved Predictive Precision using Spatio-Temporal Model Advancements: With the use of graph neural networks (GNNs) in conjunction with baseline models (such as LSTM and MLP), "
                    "our method should provide better hourly PM2.5 forecasts during wildfire seasons. We expect that the GNN's capacity to capture intricate spatial and temporal connections will enhance performance measures like MAE and RMSE, "
                    "especially for sharp increases in PM2.5 during major wildfire occurrences."
                ),
                html.Li(
                    "Urban versus Rural Impacts: Due to variations in baseline pollution levels and local topography influencing dispersion, wildfire-induced increases in PM2.5 are more noticeable in urban areas than in rural ones. "
                    "Urban settings may make wildfire smoke buildup worse due to their greater baseline pollution levels and structural effects on airflow."
                ),
                html.Li(
                    "Weather-Related Factors: Detailed meteorological data (e.g., wind speed and direction, temperature inversions, humidity) will reveal which climatic factors most strongly exacerbate PM2.5 buildup during wildfire events. "
                    "The goal is to demonstrate that certain weather conditions not only correlate but also amplify the impact of wildfire emissions on air quality, thereby refining forecast lead times and accuracy."
                ),
            ]
        ),
    ],
    style={"padding": "20px"}
)