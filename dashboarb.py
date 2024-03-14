import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from dash import dcc, html, Dash
import pandas as pd
import numpy as np
import json
import datetime
import dash_bootstrap_components as dbc
import dash.dependencies

# Ignore the SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
pd.options.mode.chained_assignment = None


class DashConfig:
    """
    Represents the configuration settings for the Dashboard application.
    """

    def __init__(self):
        self.masterFolder = None
        self.tradeFile = None
        self.pnlLookback = None

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        with open("dashboarb_config.json", "r") as f:
            config = json.load(f)

        self.masterFolder = config["appConfig"]["masterFolder"]
        self.tradeFile = config["appConfig"]["tradeFile"]
        self.pnlLookback = config["appConfig"]["pnlLookback"]

        self.help_text = """
        Changing lookback only affects the static chart. The tabs below will always display all historical data.
        The slider can be used to zoom in for the interactive chart.
        """


dc = DashConfig()


def handle_json_trade_output(filepath):
    """
    Process a newline-delimited JSON file containing trade data.

    Args:
        filepath (str): The path to the JSON file.

    Returns:
        tuple: A tuple containing three DataFrames:
            - df: The main DataFrame with all trade data.
            - buy_df: A DataFrame containing only the 'bought' trades.
            - sell_df: A DataFrame containing only the 'sold' trades.
    """

    # Load newline-delimited JSON file into a list of dictionaries
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f]

    # Convert the list of dictionaries into a list of dictionaries with 'date' and 'values' keys
    data = [
        {"date": list(item.keys())[0], "values": list(item.values())[0]}
        for item in data
    ]

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)

    # Set the date as the index
    df.set_index("date", inplace=True)

    # Split the values list into separate columns
    df = pd.DataFrame(df["values"].to_list(), index=df.index)

    # Drop 12th column because it is empty
    df.drop(df.columns[12], axis=1, inplace=True)

    # Get the previous and next values
    df["prev"] = df[11].shift(1)
    df["next"] = df[11].shift(-1)

    # Replace 0s with the average of the previous and next values, or either one if it exists
    df[11] = np.where(df[11] == 0, (df["prev"] + df["next"]) / 2, df[11])
    df[11] = df[11].ffill()
    df[11] = df[11].bfill()

    # Drop the temporary columns
    df = df.drop(columns=["prev", "next"])

    # Column names:
    bought_columns = [
        "side",
        "realBuySpreadPct",
        "buyRoe",
        "buyPrice",
        "sellPrice",
        "buyAmountR",
        "sellAmountR",
        "buyPNL",
        "investedBalance",
        "delim",
        "symbol",
        "balance",
        "mstime",
    ]
    sold_columns = [
        "side",
        "sellDeltaTaker",
        "realSellSpread",
        "sellPrice",
        "buyPrice",
        "amount",
        "sellroe",
        "totalSpreadSelllROE",
        "totalRoe",
        "None",
        "symbol",
        "balance",
        "mstime",
    ]

    # buy_df is where df[0] == 'bought'
    buy_df = df[df[0] == "bought"]
    sell_df = df[df[0] == "sold"]

    # Set column names
    buy_df.columns = bought_columns
    sell_df.columns = sold_columns

    # drop delim and None columns
    buy_df.drop(columns=["delim", "side"], inplace=True)
    sell_df.drop(columns=["None", "side"], inplace=True)

    buy_df["entryPriceSpreadUSDT"] = buy_df["buyPrice"] - buy_df["sellPrice"]
    cols = list(buy_df.columns)
    cols.insert(5, cols.pop(cols.index("entryPriceSpreadUSDT")))
    buy_df = buy_df[cols]

    buy_df["entryAmountSpreadUSDT"] = (
        buy_df["buyAmountR"] - buy_df["sellAmountR"]
    ) * buy_df["buyPrice"]
    cols = list(buy_df.columns)
    cols.insert(8, cols.pop(cols.index("entryAmountSpreadUSDT")))
    buy_df = buy_df[cols]

    return df, buy_df, sell_df


def plot_bar_chart(roe_dict, num):
    roe_dict = dict(sorted(roe_dict.items(), key=lambda item: item[1]))
    top_vals = dict(list(roe_dict.items())[-num:])
    bottom_vals = dict(list(roe_dict.items())[:num])

    # Create traces for top and bottom 5
    top_trace = go.Bar(
        x=list(top_vals.keys()),
        y=list(top_vals.values()),
        name="Top 5",
        showlegend=False,
    )
    bottom_trace = go.Bar(
        x=list(bottom_vals.keys()),
        y=list(bottom_vals.values()),
        name="Bottom 5",
        showlegend=False,
    )

    return top_trace, bottom_trace


def get_json_files_list(name):
    import os
    import glob

    json_files_list = []
    for root, dirs, files in os.walk(dc.masterFolder):
        for file in files:
            if file == name:
                json_files_list.append(os.path.join(root, file))
    return json_files_list


def get_totalRoe_dict(lookback=None):
    totalRoe_dict = {}
    for folder in get_json_files_list(name=dc.tradeFile):
        print("getting json file from folder:", folder)
        try:
            df, buy_df, sell_df = handle_json_trade_output(folder)
            sell_df["totalRoe"] = sell_df["totalRoe"].astype(float)
            if lookback:
                sell_df["datetime"] = pd.to_datetime(
                    sell_df.index, format="%d/%m/%Y-%H:%M:%S:%f"
                )
                sell_df = sell_df[
                    sell_df["datetime"]
                    > datetime.datetime.now() - datetime.timedelta(days=lookback)
                ]
            sell_df["tradeProfit"] = (
                (sell_df["sellPrice"] + sell_df["buyPrice"])
                / 2
                * sell_df["amount"]
                * sell_df["totalRoe"]
                / 100
            )
            # Adjust the below line depending on folder structure
            # print(folder)
            totalRoe_dict[folder.split("_")[1].split("/")[0]] = sell_df[
                "tradeProfit"
            ].sum()
        except Exception as e:
            print(
                "Error in get_totalRoe_dict, likely a problem with trade_operations of folder:",
                folder,
            )
            continue
    return totalRoe_dict


def handle_balance_csv():
    path = dc.masterFolder + "balance.csv"
    in_out_table = handle_in_out_csv()
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Drop all rows if they don't have "00"  in minute part - these are genereated from pm2 restarts
    df = df[df["datetime"].str.contains("00")]
    # Drop the rows where "23:59" is in the datetime column
    df = df[~df["datetime"].str.contains("23:59")]

    # Calculate rolling one day sharpe ratio in df from total_usdt
    df["hourly_return"] = df["total_usdt"].pct_change()

    # in_out_table['datetime'] = pd.to_datetime(in_out_table['datetime'], format='%Y/%m/%d %H:%M:%S')
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d/%m/%Y %H:%M:%S")

    in_out_table["datetime"] = in_out_table["datetime"].dt.ceil("h")
    df["datetime"] = df["datetime"].dt.floor("h")

    ## TODO - Need to have a solution for sharpe ratio when deposit and withdraws are made
    # Drop all rows that have a datetime in in_out_table
    # df = df[~df["datetime"].isin(in_out_table["datetime"])]

    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y/%m/%D %H:%M:%S")

    # If in_out_table has a datetime that is in df, change the df['hourly_return'] to the average of the prev and next value
    df["hourly_return"] = df["hourly_return"].where(
        ~df["datetime"].isin(in_out_table["datetime"]),
        (df["hourly_return"].shift(1) + df["hourly_return"].shift(-1)) / 2,
    )

    # Calculate rolling single day Sharpe ratio - 24 hours
    df["sharpe_ratio"] = (
        df["hourly_return"].rolling(window=24).mean()
        / df["hourly_return"].rolling(window=24).std()
    )
    # Annualize the sharpe ratio
    df["sharpe_ratio"] = df["sharpe_ratio"] * np.sqrt(24 * 365)
    # MA168 of sharpe_ratio
    df["sharpe_ratio_MA168"] = df["sharpe_ratio"].rolling(window=168).mean()

    # Free balance percent to plot alongside balance
    df["free_pct"] = (df["spot_margin"] + df["future_margin"]) / (
        df["spot_usdt"] + df["future_usdt"]
    )

    # Annualised vol of hourly_return with expanding
    df["vol"] = df["hourly_return"].expanding().std() * np.sqrt(24 * 365)

    return df


def handle_in_out_csv():
    path = dc.masterFolder + "in_out.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y/%m/%d %H:%M:%S")

    return df


def plot_balance(df, in_out):

    # Plot the balance
    balance = go.Figure()

    # Add total_usdt trace
    balance.add_trace(
        go.Scatter(x=df["datetime"], y=df["total_usdt"], mode="lines", name="USDT")
    )

    # TODO - This is too cluttered.  Unsure if we want it anywhere.  Maybe in another modal.
    # Add free_pct trace
    # balance.add_trace(
    #     go.Scatter(x=df["datetime"], y=df["free_pct"].round(2), mode="lines", name="Free %", yaxis="y2")
    # )
    # balance.data[1].update(opacity=0.1)

    # Update layout
    balance.update_layout(
        title="Balance",
        xaxis_title="Datetime",
        yaxis_title="USDT",
        xaxis=dict(
            rangeslider=dict(visible=True),
            showticklabels=False,
        ),
        yaxis=dict(fixedrange=False),
        height=400,
        margin=dict(l=20, r=20, t=40, b=40),
        template="plotly_dark",
    )

    # This makes the bullet show on the preceeding hour before the balance change
    in_out["datetime"] = in_out["datetime"].dt.floor("h")

    # Add red dots to the balance graph for every datetime in in_out
    for date in in_out["datetime"]:
        balance.add_trace(
            go.Scatter(
                x=[date],
                y=[df.loc[df["datetime"] == date, "total_usdt"].iloc[0]],
                mode="markers",
                marker=dict(color="red"),
                showlegend=False,
                name="IN/OUT",
            )
        )

    # Add legend for the IN/OUT red dots
    balance.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color="red"),
            showlegend=True,
            name="IN/OUT",
        )
    )

    # Move the key to the top of the graph under the title
    balance.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Plot the sharpe ratio
    sharpe = go.Figure()
    sharpe.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["sharpe_ratio"],
            mode="lines",
            name="Rolling 24h Sharpe Ratio",
        )
    )
    sharpe.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["sharpe_ratio_MA168"],
            mode="lines",
            name="7 Day MA (168 hours)",
        )
    )
    sharpe.update_layout(
        title="Annualized Sharpe",
        xaxis_title="Datetime",
        yaxis_title="Sharpe Ratio",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(rangeslider=dict(visible=True), showticklabels=False),
        yaxis=dict(fixedrange=False),
        height=400,
        margin=dict(l=20, r=20, t=40, b=40),
        template="plotly_dark",
    )

    # Deposit / Withdraw table
    in_out = ff.create_table(in_out)
    in_out.update_layout(
        template="plotly_dark",
    )

    return balance, sharpe, in_out


def run_dash():
    """
    Run the Dash app.
    """

    # Create a Dash app with dark theme
    app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

    # Add the dropdown for the lookback period
    lookback_input = dcc.Input(
        id="lookback-input",
        type="number",
        value=dc.pnlLookback,  # Default value
        step=1,
        style={"width": "100px"},  # Adjust width as needed
    )

    # Top figure for the static chart
    roe_dict = get_totalRoe_dict(
        lookback=dc.pnlLookback
    )  # Use default lookback initially
    top_trace, bottom_trace = plot_bar_chart(roe_dict, num=10)
    static_fig = make_subplots(rows=1, cols=2, subplot_titles=("Winners", "Losers"))

    # Add top and bottom traces to the subplots
    static_fig.add_trace(top_trace, row=1, col=1)
    static_fig.add_trace(bottom_trace, row=1, col=2)

    static_fig.update_layout(
        barmode="group",
        height=400,
        margin=dict(l=40, r=40, t=80, b=40),
        template="plotly_dark",
    )

    folders = [
        name
        for name in os.listdir(dc.masterFolder)
        if os.path.isdir(os.path.join(dc.masterFolder, name))
        and dc.tradeFile in os.listdir(os.path.join(dc.masterFolder, name))
    ]
    folder_dropdown = dcc.Dropdown(
        id="folder-selector",
        options=[{"label": folder, "value": folder} for folder in folders],
        value=folders[0] if folders else None,
        style={"color": "black", "backgroundColor": "#A9A9A9", "font-size": "20px"},
    )

    # Create the balance and sharpe ratio modal and plot
    df = handle_balance_csv()
    in_out = handle_in_out_csv()
    balance, sharpe, in_out_table = plot_balance(df, in_out)

    modal = dbc.Modal(
        [
            dbc.ModalHeader("Details"),
            dbc.ModalBody(
                [
                    html.Div(
                        [
                            dcc.Graph(id="balance-graph", style={"width": "90%"}),
                            dcc.Graph(id="sharpe-graph", style={"width": "90%"}),
                        ],
                        style={"margin-bottom": "20px"},
                    ),
                    # Add text here
                    html.Div(
                        [
                            html.P(
                                "Annualized vol, expanding, full time series: "
                                + str(round(df["vol"].iloc[-1], 10))
                            ),
                            html.P("Deposits and Withdrawals"),
                        ],
                        style={
                            "margin-bottom": "20px",
                            "color": "white",
                        },  # Adjust color as needed
                    ),
                    # Add the 'in_out' table
                    html.Div(
                        [
                            dcc.Graph(id="in_out-table", style={"width": "90%"}),
                        ],
                        style={"margin-bottom": "20px"},
                    ),
                ]
            ),
            dbc.ModalFooter(dbc.Button("Close", id="close-modal", className="ml-auto")),
        ],
        id="modal",
        size="xl",
    )

    ##########################
    # APP LAYOUT AND CALLBACKS
    ##########################
    # Define the callback to toggle the modal visibility
    @app.callback(
        dash.dependencies.Output("modal", "is_open"),
        [
            dash.dependencies.Input("open-modal", "n_clicks"),
            dash.dependencies.Input("close-modal", "n_clicks"),
        ],
        [dash.dependencies.State("modal", "is_open")],
    )
    def toggle_modal(open_clicks, close_clicks, is_open):
        if open_clicks or close_clicks:
            return not is_open
        return is_open

    # Define the callback to update the modal's content
    @app.callback(
        [
            dash.dependencies.Output("balance-graph", "figure"),
            dash.dependencies.Output("sharpe-graph", "figure"),
            dash.dependencies.Output("in_out-table", "figure"),
        ],
        [dash.dependencies.Input("modal", "is_open")],
    )
    def update_modal_content(is_open):
        if is_open:
            df = handle_balance_csv()
            in_out = handle_in_out_csv()
            balance, sharpe, in_out_table = plot_balance(df, in_out)
            return balance, sharpe, in_out_table
        return dash.no_update, dash.no_update, dash.no_update

    img = "assets/logo.png"
    app.layout = html.Div(
        [
            html.Div(dc.help_text),
            dbc.Row(
                [
                    dbc.Col(
                        html.Label("Lookback period (days):"),
                        width={"size": 2},
                    ),
                    dbc.Col(
                        lookback_input,
                        width={"size": 2},
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Balance History Details",
                            id="open-modal",
                            color="primary",
                            className="mr-1",
                        ),
                        width={"size": 2},
                    ),
                ],
                className="mb-3",
            ),
            html.Div(
                [
                    dcc.Graph(
                        id="static-graph",
                        figure=static_fig,
                        style={"width": "75%", "display": "inline-block"},
                    ),
                    html.Img(
                        src=img,
                        style={
                            "width": "25%",
                            "display": "inline-block",
                            "vertical-align": "top",
                        },
                    ),
                ]
            ),
            html.Label("Select a folder:"),
            folder_dropdown,
            dcc.Tabs(id="side-tabs"),
            modal,
        ]
    )

    # Folder selection callback
    @app.callback(
        [
            dash.dependencies.Output("static-graph", "figure"),
            dash.dependencies.Output("side-tabs", "children"),
        ],
        [
            dash.dependencies.Input("folder-selector", "value"),
            dash.dependencies.Input("lookback-input", "value"),
        ],  # Add input dependency
    )
    def update_dashboard(selected_folder, lookback):
        # Update the static graph
        roe_dict = get_totalRoe_dict(lookback=lookback)
        top_trace, bottom_trace = plot_bar_chart(roe_dict, num=10)
        static_fig = make_subplots(rows=1, cols=2, subplot_titles=("Winners", "Losers"))
        static_fig.add_trace(top_trace, row=1, col=1)
        static_fig.add_trace(bottom_trace, row=1, col=2)
        static_fig.update_layout(
            title="Top and Bottom 5 coins by total profit in USD",
            barmode="group",
            height=400,
            margin=dict(l=40, r=40, t=80, b=40),
            template="plotly_dark",
        )

        # Update the side tabs
        df, buy_df, sell_df = handle_json_trade_output(
            os.path.join(dc.masterFolder, selected_folder, dc.tradeFile)
        )
        side_tabs = []
        for df_name, df in [("Buy", buy_df), ("Sell", sell_df)]:
            tabs = []
            for column in df.columns:
                data = go.Scatter(x=df.index, y=df[column], mode="lines")
                dates = pd.to_datetime(df.index, format="%d/%m/%Y-%H:%M:%S:%f").date
                layout = go.Layout(
                    xaxis=dict(
                        rangeslider=dict(visible=True, bgcolor="#B0B0B0"),
                        tickvals=df.index,
                        ticktext=dates,
                    ),
                    height=600,
                    margin=dict(l=20, r=20, t=20, b=20),
                    template="plotly_dark",
                )
                fig = go.Figure(data=data, layout=layout)
                tabs.append(
                    dcc.Tab(
                        label=column,
                        children=[dcc.Graph(figure=fig)],
                        style={
                            "color": "black",
                            "backgroundColor": "#A9A9A9",
                            "textAlign": "left",
                            "padding-left": "2px",
                            "font-size": "15px",
                        },
                    )
                )
            side_tabs.append(
                dcc.Tab(
                    label=df_name,
                    children=[dcc.Tabs(tabs)],
                    style={
                        "color": "black",
                        "backgroundColor": "#A9A9A9",
                        "font-size": "20px",
                    },
                )
            )
        return static_fig, side_tabs

    return app


# Run the app
if __name__ == "__main__":
    print("Starting DashboarB 0.3.1")
    app = run_dash()
    app.run_server(debug=True, host="0.0.0.0")
