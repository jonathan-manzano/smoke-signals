# analytics.py
import dash
from dash import html, dcc, callback, Input, Output
import plotly.express as px
from utils import get_csv_from_gcs
from Nav_bar import create_navbar

dash.register_page(__name__, path="/analytics", name="Methods")

# load your dataframe as before…
df = get_csv_from_gcs("smoke-signal-bucket", "gapminder2007.csv")

layout = html.Div([
    create_navbar(),

    # Hero
    html.Section(
      html.Div([
        html.H1("Analytics Dashboard"),
        html.P("Interactive charts to explore your data", className="lead")
      ], className="container"),
      className="hero"
    ),

    # Metric cards if any…
    html.Section(
      html.Div([
        # e.g. html.Div([...], className="metric"), …
      ], className="metrics container")
    ),

    # Plot + controls
    html.Section(
      html.Div([
        dcc.RadioItems(
          id="analytics-radio-item",
          options=[{"label": col, "value": col} for col in ["pop", "lifeExp", "gdpPercap"]],
          value="pop",
          inline=True
        ),
        dcc.Graph(id="analytics-graph")
      ], className="container"),
    ),

    # Final CTA
    html.Section(
      html.Div([
        html.A("View Findings →", href="/findings", className="btn btn-primary")
      ], className="container"),
      className="hero hero--inverse"
    ),
])

@callback(
    Output("analytics-graph", "figure"),
    Input("analytics-radio-item", "value"),
)
def update_graph(col_chosen):
    fig = px.histogram(df, x="continent", y=col_chosen,
                       histfunc="avg",
                       title=f"Average {col_chosen} by Continent")
    return fig
