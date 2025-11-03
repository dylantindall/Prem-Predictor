import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
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

# Load predictions data
try:
    predictions_df = pd.read_csv("fixtures_with_accuracy.csv")
    has_accuracy = True
except FileNotFoundError:
    predictions_df = pd.read_csv("fixtures_2025_2026_simulated.csv")
    has_accuracy = False
    predictions_df['prediction_status'] = 'pending'
    predictions_df['prediction_correct'] = None
    predictions_df['predicted_result'] = predictions_df['result']

# Filter for predictions only (matchweek 11 onwards where goals are missing)
predictions_df = predictions_df[predictions_df['match_week'] >= 11].copy()

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
                "marginBottom": "20px"
            })
        )
    ]),

    # ---------------- Navigation Buttons ----------------
    dbc.Row([
        dbc.Col(
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button(
                        "Historic Data",
                        id="historic-btn",
                        color="light",
                        style={
                            "fontFamily": "Radikal, sans-serif",
                            "fontWeight": "bold",
                            "padding": "10px 30px",
                            "marginRight": "10px"
                        }
                    ),
                    dbc.Button(
                        "Model Predictions",
                        id="predictions-btn",
                        color="light",
                        style={
                            "fontFamily": "Radikal, sans-serif",
                            "fontWeight": "bold",
                            "padding": "10px 30px"
                        }
                    )
                ])
            ], style={
                "display": "flex",
                "justifyContent": "center",
                "marginBottom": "30px"
            })
        )
    ]),

    # ---------------- Historic Data Page ----------------
    html.Div(id="historic-page", children=[
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
    ], style={"display": "block"}),

    # ---------------- Predictions Page ----------------
    html.Div(id="predictions-page", children=[
        dbc.Row([
            dbc.Col([

                # Predictions Title
                html.H2(
                    "Model Predictions",
                    style={
                        "fontFamily": "Radikal, sans-serif",
                        "fontSize": "22px",
                        "color": "white",
                        "margin": "0",
                        "textAlign": "center",
                        "marginBottom": "20px"
                    }
                ),

                # Predicted Final Standings
                html.H3(
                    "Predicted Final Table",
                    style={
                        "fontFamily": "Radikal, sans-serif",
                        "fontSize": "18px",
                        "color": "white",
                        "margin": "0",
                        "textAlign": "center",
                        "marginBottom": "15px",
                        "marginTop": "20px"
                    }
                ),

                dash_table.DataTable(
                    id="predicted-standings-table",
                    columns=[
                        {"name": "Rank", "id": "Rank"},
                        {"name": "Team", "id": "Team"},
                        {"name": "Points", "id": "Points"},
                        {"name": "Wins", "id": "Wins"},
                        {"name": "Draws", "id": "Draws"},
                        {"name": "Losses", "id": "Losses"}
                    ],
                    style_table={
                        "overflowX": "auto",
                        "border": "2px solid black",
                        "borderCollapse": "collapse",
                        "marginBottom": "40px"
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

                # Fixtures Section Title
                html.H3(
                    "Matchweek Predictions",
                    style={
                        "fontFamily": "Radikal, sans-serif",
                        "fontSize": "18px",
                        "color": "white",
                        "margin": "0",
                        "textAlign": "center",
                        "marginBottom": "20px"
                    }
                ),

                # Matchweek slider
                html.Div([
                    html.Label("Select Matchweek:", style={"color": "white", "marginBottom": "10px"}),
                    dcc.Slider(
                        id="matchweek-slider",
                        min=predictions_df['match_week'].min(),
                        max=predictions_df['match_week'].max(),
                        step=1,
                        value=predictions_df['match_week'].min(),
                        marks={int(mw): str(int(mw)) for mw in predictions_df['match_week'].unique()},
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], style={"marginBottom": "30px"}),

                # Predictions table
                html.Div(
                    dash_table.DataTable(
                        id="predictions-table",
                        columns=[
                            {"name": "Date", "id": "date"},
                            {"name": "Home Team", "id": "home_team"},
                            {"name": "Prediction", "id": "prediction"},
                            {"name": "Away Team", "id": "away_team"}
                        ],
                        style_table={
                            "borderCollapse": "collapse",
                            "margin": "0 auto",
                            "width": "100%",
                            "maxWidth": "900px"
                        },
                        style_cell={
                            "textAlign": "center",
                            "padding": "10px",
                            "borderLeft": "1px solid black",
                            "borderRight": "1px solid black",
                            "borderBottom": "1px solid black",
                            "backgroundColor": "white",
                            "fontFamily": "Radikal, sans-serif"
                        },
                        style_cell_conditional=[
                            {
                                "if": {"column_id": "date"},
                                "backgroundColor": "rgb(65, 0, 64)",
                                "color": "white",
                                "border": "2px solid black",
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
                                "if": {"column_id": "prediction"},
                                "width": "15%",
                                "fontWeight": "bold",
                                "fontSize": "14px"
                            },
                            {
                                "if": {"column_id": "away_team"},
                                "textAlign": "left",
                                "paddingLeft": "15px",
                                "width": "30%"
                            }
                        ],
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f4f4f4",
                            "border": "2px solid black",
                            "fontFamily": "Radikal, sans-serif"
                        },
                        style_as_list_view=True
                    ),
                    style={"display": "flex", "justifyContent": "center", "marginBottom": "60px"}
                )

            ], width=10)
        ], justify="center")
    ], style={"display": "none"})

], fluid=True, style={
    "backgroundColor": "rgb(65, 0, 64)",
    "minHeight": "100vh"
})


# ---------------- Callbacks ----------------

# Toggle between pages
@app.callback(
    [Output("historic-page", "style"),
     Output("predictions-page", "style"),
     Output("historic-btn", "color"),
     Output("predictions-btn", "color"),
     Output("predicted-standings-table", "data")],
    [Input("historic-btn", "n_clicks"),
     Input("predictions-btn", "n_clicks")]
)
def toggle_pages(historic_clicks, predictions_clicks):
    ctx = dash.callback_context
    
    # Load predicted standings
    try:
        standings_df = pd.read_csv('predicted_final_standings_2025_2026.csv')
        standings_data = standings_df.to_dict("records")
    except FileNotFoundError:
        standings_data = []
    
    if not ctx.triggered:
        # Default to historic page
        return {"display": "block"}, {"display": "none"}, "primary", "light", standings_data
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "historic-btn":
        return {"display": "block"}, {"display": "none"}, "primary", "light", standings_data
    else:
        return {"display": "none"}, {"display": "block"}, "light", "primary", standings_data


# Update standings table
@app.callback(
    Output("standings-table", "data"),
    Input("season-dropdown", "value")
)
def update_table(filename):
    df = pd.read_csv(os.path.join(SEASON_PATH, filename))
    df = df[["Rank", "Team", "Points", "Wins", "Draws", "Losses", "GF", "GA", "GD"]]
    return df.to_dict("records")


# Update month slider
@app.callback(
    [Output("month-slider", "marks"),
     Output("month-slider", "min"),
     Output("month-slider", "max"),
     Output("month-slider", "value")],
    Input("season-dropdown", "value")
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


# Update fixtures table
@app.callback(
    Output("fixtures-table", "data"),
    [Input("season-dropdown", "value"),
     Input("month-slider", "value")]
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


# Update predictions table
@app.callback(
    Output("predictions-table", "data"),
    Input("matchweek-slider", "value")
)
def update_predictions(matchweek):
    # Filter for selected matchweek
    mw_predictions = predictions_df[predictions_df['match_week'] == matchweek].copy()
    
    # Create prediction string based on predicted_result column
    def format_prediction(row):
        pred = row.get('predicted_result', row.get('result', 'N/A'))
        if pred == 'H':
            return "Home Win"
        elif pred == 'A':
            return "Away Win"
        elif pred == 'D':
            return "Draw"
        else:
            return "N/A"
    
    def format_actual(row):
        # Check if match has been played
        if pd.notna(row.get('home_goals')):
            actual = row.get('result')
            goals_h = int(row['home_goals'])
            goals_a = int(row['away_goals'])
            if actual == 'H':
                return f"{goals_h}-{goals_a} (H)"
            elif actual == 'A':
                return f"{goals_h}-{goals_a} (A)"
            elif actual == 'D':
                return f"{goals_h}-{goals_a} (D)"
        return ""
    
    def format_accuracy(row):
        status = row.get('prediction_status', 'pending')
        if status == 'correct':
            return "✓"
        elif status == 'incorrect':
            return "✗"
        return ""
    
    mw_predictions['prediction'] = mw_predictions.apply(format_prediction, axis=1)
    mw_predictions['actual_result'] = mw_predictions.apply(format_actual, axis=1)
    mw_predictions['accuracy'] = mw_predictions.apply(format_accuracy, axis=1)
    
    # Select and order columns
    predictions_data = mw_predictions[['date', 'home_team', 'prediction', 'away_team', 'actual_result', 'accuracy']]
    
    return predictions_data.to_dict("records")




if __name__ == "__main__":
    app.run(debug=True)