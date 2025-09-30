import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import os

# Path to season tables folder
SEASON_PATH = "season_tables"

# Get list of CSV files
season_files = [f for f in os.listdir(SEASON_PATH) if f.endswith(".csv")]

# Extract season labels (e.g. "1995-1996") from filenames
season_options = [
    {"label": f.replace("season_", "").replace(".csv", "").replace("_", "-"),
     "value": f}
    for f in sorted(season_files)
]

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Layout
app.layout = dbc.Container([

    # ---------------- Header (Logo + Title) ----------------
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Img(
                    src="/assets/prem_logo.png",
                    style={"height": "60px", "marginRight": "15px"}
                ),
                html.H1(
                    "Premier League Dashboard",
                    style={
                        "fontFamily": "Radikal, sans-serif",
                        "fontWeight": "bold",
                        "color": "rgb(52, 0, 64)",
                        "margin": 0
                    }
                )
            ], style={
                "backgroundColor": "white",
                "padding": "10px",
                "borderRadius": "8px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "marginTop": "20px",
                "marginBottom": "40px"
            })
        )
    ]),

    # ---------------- Title + Dropdown + Table in Same Column ----------------
    dbc.Row([
        dbc.Col([

            # Title + Dropdown in one row
            dbc.Row([
                dbc.Col(
                    html.Div(
                        style={"position": "relative", "width": "100%"}
                    ),
                    width=3
                ),
                dbc.Col(
                    html.H2(
                        "Table",
                        style={
                            "fontFamily": "Radikal, sans-serif",
                            "fontSize": "22px",
                            "color": "white",
                            "margin": "0",
                            "textAlign": "center"
                        }
                    ),
                    width=6
                ),
                dbc.Col(
                    html.Div([
                        html.Label("Select Season:", style={"color": "white", "marginRight": "8px"}),
                        dcc.Dropdown(
                            id="season-dropdown",
                            options=season_options,
                            value=season_options[-1]["value"],
                            clearable=False,
                            style={"width": "200px", "color": "black"}
                        )
                    ], style={"display": "flex", "justifyContent": "flex-end", "alignItems": "center"}),
                    width=3
                )
            ], justify="between", className="mb-2"),

            # Standings table
            dash_table.DataTable(
                id="standings-table",
                columns=[
                    {"name": "Rank", "id": "Rank"},
                    {"name": "Team", "id": "Team"},
                    {"name": "Points", "id": "Points"},
                    {"name": "Wins", "id": "Wins"},
                    {"name": "Draws", "id": "Draws"},
                    {"name": "Losses", "id": "Losses"},
                    {"name": "GF", "id": "GF"},
                    {"name": "GA", "id": "GA"},
                    {"name": "GD", "id": "GD"}
                ],
                style_table={
                    "overflowX": "auto",
                    "border": "2px solid black",
                    "borderCollapse": "collapse"
                },
                style_cell={
                    "textAlign": "center",
                    "padding": "6px",
                    "borderLeft": "1px solid black",
                    "borderRight": "1px solid black",
                    "borderBottom": "1px solid black",
                    "backgroundColor": "white"
                },
                style_header={
                    "fontWeight": "bold",
                    "backgroundColor": "#f4f4f4",
                    "border": "2px solid black"
                },
                style_as_list_view=True
            ),

            # Fixtures section title
            html.H2(
                "Fixtures",
                style={
                    "fontFamily": "Radikal, sans-serif",
                    "fontSize": "22px",
                    "color": "white",
                    "margin": "0",
                    "textAlign": "center",
                    "marginTop": "40px",
                    "marginBottom": "20px"
                }
            ),

            # Month slider
            html.Div([
                html.Label("Select Month:", style={"color": "white"}),
                dcc.Slider(
                    id="month-slider",
                    min=0, max=0, step=1, value=0,
                    marks={}
                )
            ], style={"marginBottom": "20px"}),

            # Fixtures table
            html.Div(
                dash_table.DataTable(
                    id="fixtures-table",
                    columns=[
                        {"name": "", "id": "date"},
                        {"name": "", "id": "home_team"},
                        {"name": "", "id": "home_goals"},
                        {"name": "", "id": "away_goals"},
                        {"name": "", "id": "away_team"}
                    ],
                    style_table={
                        "borderCollapse": "collapse",
                        "margin": "0 auto",
                        "width": "100%",
                        "maxWidth": "900px"
                    },
                    style_cell={
                        "textAlign": "center",
                        "padding": "8px",
                        "borderLeft": "1px solid black",
                        "borderRight": "1px solid black",
                        "borderBottom": "1px solid black",
                        "backgroundColor": "white"
                    },
                    style_cell_conditional=[
                        {
                            "if": {"column_id": "date"},
                            "backgroundColor": "rgb(65, 0, 64)",
                            "color": "white",
                            "border": "none",
                            "fontSize": "12px",
                            "width": "15%"
                        },
                        {
                            "if": {"column_id": "home_team"},
                            "textAlign": "right",
                            "paddingRight": "15px",
                            "width": "30%"
                        },
                        {
                            "if": {"column_id": "home_goals"},
                            "width": "5%",
                            "paddingLeft": "8px",
                            "paddingRight": "8px"
                        },
                        {
                            "if": {"column_id": "away_goals"},
                            "width": "5%",
                            "paddingLeft": "8px",
                            "paddingRight": "8px"
                        },
                        {
                            "if": {"column_id": "away_team"},
                            "textAlign": "left",
                            "paddingLeft": "15px",
                            "width": "30%"
                        }
                    ],
                    style_header={
                        "display": "none"
                    },
                    style_as_list_view=True,
                    page_size=15
                ),
                style={"display": "flex", "justifyContent": "center"}
            )

        ], width=10)
    ], justify="center")

], fluid=True, style={
    "backgroundColor": "rgb(65, 0, 64)",
    "minHeight": "100vh"
})


# ---------------- Callbacks ----------------
@app.callback(
    dash.Output("standings-table", "data"),
    dash.Input("season-dropdown", "value")
)
def update_table(filename):
    df = pd.read_csv(os.path.join(SEASON_PATH, filename))
    df = df[["Rank", "Team", "Points", "Wins", "Draws", "Losses", "GF", "GA", "GD"]]
    return df.to_dict("records")


@app.callback(
    [dash.Output("month-slider", "marks"),
     dash.Output("month-slider", "min"),
     dash.Output("month-slider", "max"),
     dash.Output("month-slider", "value")],
    dash.Input("season-dropdown", "value")
)
def update_month_slider(filename):
    # Extract season from filename (e.g., "season_1995_1996.csv" -> "1995-96")
    season = filename.replace("season_", "").replace(".csv", "").replace("_", "-")
    
    # Load full results data
    df = pd.read_csv("prem_results.csv")
    
    # Filter for the selected season
    season_df = df[df["season"] == season]
    
    # Get unique months in chronological order for this season
    months = season_df["month_year"].unique().tolist()
    
    # Create marks for slider
    marks = {i: m for i, m in enumerate(months)}
    
    return marks, 0, len(months) - 1, 0


@app.callback(
    dash.Output("fixtures-table", "data"),
    [dash.Input("season-dropdown", "value"),
     dash.Input("month-slider", "value")]
)
def update_fixtures(filename, month_idx):
    # Extract season from filename
    season = filename.replace("season_", "").replace(".csv", "").replace("_", "-")
    
    # Load full results data
    df = pd.read_csv("prem_results.csv")
    
    # Filter for the selected season
    season_df = df[df["season"] == season]
    
    # Get unique months for this season
    months = season_df["month_year"].unique().tolist()
    
    if not months or month_idx >= len(months):
        return []
    
    # Get selected month
    selected_month = months[month_idx]
    
    # Filter fixtures for selected month
    month_fixtures = season_df[season_df["month_year"] == selected_month]
    
    # Select columns to display
    fixtures = month_fixtures[["date", "home_team", "away_team", "home_goals", "away_goals"]]
    
    return fixtures.to_dict("records")


if __name__ == "__main__":
    app.run(debug=True)