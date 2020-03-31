# Import required libraries
import copy
import pathlib
import dash
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from utils import *
from datetime import timedelta
import datetime
import json

import urllib.parse
import re


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)


server = app.server

# ================================================================================
# Keys
CTRY_K = "Country/Region"

# Data
df = pd.read_hdf("./data/covid19.h5", "covid19_data")

# Available data
ctry_arr = df.index.levels[0].values
dates_arr = pd.to_datetime(df.columns[3:].values)
sts_arr = np.unique(df.index.levels[1].values)

# Buttons options
ctry_opts = [{"label": ctry, "value": ctry} for ctry in ctry_arr]

dt_opt_sldr = np.linspace(0, dates_arr.shape[0] - 1, 10, dtype=int)
dt_sldr = {
    int(i): {
        "label": dates_arr[i].strftime("%d/%m/%y"),
        "style": {"transform": "rotate(45deg)"},
    }
    for i in dt_opt_sldr
}
dt_max_sldr = dates_arr.shape[0] - 1


# Graphs
pred_graph = [[]]
dpred_graph = [[]]
pred_lyt = []

grab_click = 0
clear_click = None

app.title = "COVID-19 - Status & Predictions"

mapbox_access_token = "pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w"

layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation="h"),
    title="Satellite Overview",
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(lon=-78.05, lat=42.54),
        zoom=7,
    ),
)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        # html.Img(
                        #     src=app.get_asset_url('dash-logo.png'),
                        #     id='plotly-image',
                        #     style={
                        #         'height': '60px',
                        #         'width': 'auto',
                        #         'margin-bottom': '25px',
                        #     },
                        # )
                    ],
                    className="one-third column",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "COVID-19", style={"margin-bottom": "0px"},),
                                html.H5(
                                    "Status & Predictions", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="one-half column",
                    id="title",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("Based on", id="learn-more-button"),
                            href="https://coronavirus.jhu.edu/map.html",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P("Temporal evolution:",
                               className="control_label"),
                        dcc.RadioItems(
                            id="temporal_status_selector",
                            options=[
                                {"label": "Confirmed ", "value": "confirmed"},
                                {"label": "Deaths ", "value": "deaths"},
                                {"label": "Recovered ", "value": "recovered"},
                            ],
                            value="confirmed",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="countries_slct",
                            options=ctry_opts,
                            multi=True,
                            value=["China", "Italy", "US", "Spain", "Germany"],
                            className="dcc_control",
                        ),
                        dcc.Markdown(
                            """
                            **About the model:**
                            The model used to predict is a sigmoid function adjusted using least squares method. It is not always possible to fit this function. Therefore, some cases do not work. This is a very simple model, so, take the result with a grain of salt."""
                        ),
                        html.P("Prediction:", className="control_label"),
                        dcc.RadioItems(
                            id="prediction_status_selector",
                            options=[
                                {"label": "Confirmed ", "value": "confirmed"},
                                {"label": "Deaths ", "value": "deaths"},
                                {"label": "Recovered ", "value": "recovered"},
                            ],
                            value="confirmed",
                            labelStyle={"display": "inline-block"},
                            className="dcc_control",
                        ),
                        dcc.Dropdown(
                            id="country_slct",
                            options=ctry_opts,
                            multi=False,
                            value="Italy",
                            className="dcc_control",
                        ),
                        html.Div(
                            [
                                dcc.Markdown(
                                    """The model is presented below:"""),
                                html.P(id="mdl_unc_txt"),
                                html.P(id="mdl_eq_txt"),
                            ],
                            id="mdl_cfg",
                            className="mini_container",
                        ),
                        html.Div(
                            [
                                dcc.Slider(
                                    id="date_slider_pred",
                                    step=1,
                                    min=0,
                                    max=dt_max_sldr,
                                    value=dt_max_sldr,
                                    marks=dt_sldr,
                                ),
                                html.Div(
                                    [
                                        html.Div(id="date_selected"),
                                        html.Button(
                                            "Clear",
                                            style={"margin-right": 10},
                                            id="clear_graph_bt",
                                        ),
                                        html.Button(
                                            "Grab",
                                            n_clicks=0,
                                            type="submit",
                                            id="grab_graph_bt",
                                        ),
                                    ],
                                    id="slider_buttons",
                                    style={"margin-top": 50},
                                ),
                            ],
                            id="date_slider",
                        ),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(id="cases_tot_txt"),
                                        html.P("No. of cases"),
                                    ],
                                    id="ncases_tot_txt",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [
                                        html.H6(id="deaths_tot_txt"),
                                        html.P("No. of deaths"),
                                    ],
                                    id="ndeaths_tot_txt",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [
                                        html.H6(id="recv_tot_txt"),
                                        html.P("No. of recovered"),
                                    ],
                                    id="nrecv_tot_txt",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [
                                        html.H6(id="act_tot_txt"),
                                        html.P("No. of active"),
                                    ],
                                    id="nact_tot_txt",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        html.Div(
                            [dcc.Graph(id="status_graph")],
                            id="countGraphContainer",
                            className="pretty_container",
                        ),
                    ],
                    id="right-column",
                    className="eight columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="pred_graph")],
                    className="pretty_container seven columns",
                ),
                html.Div(
                    [dcc.Graph(id="dpred_graph")],
                    className="pretty_container five columns",
                ),
            ],
            className="row flex-display",
        ),
        html.Div(id="new_graph_data", style={"display": "none"}),
        html.Div(id="curr_graph_data", style={"display": "none"}),
        html.Div(id="stored_graph_data", style={"display": "none"}),
        html.Div(id="clear_nclick_data", style={"display": "none"}),
        html.Div(id="grab_nclick_data", style={"display": "none"}),
        html.Div(id="model_data", style={"display": "none"}),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("status_graph", "figure")],
)


# Update status text
@app.callback(
    Output("cases_tot_txt", "children"),
    [Input("countries_slct", "value"), Input(
        "temporal_status_selector", "value"), ],
)
def update_cases_text(countries_slct, temporal_status_selector):
    # Slice data
    slc_date = slice("2020-01-22", None)
    df_view = df.loc[(countries_slct, "confirmed"),
                     slc_date].groupby(CTRY_K).sum().T
    # Get data
    return "{:,}".format(df_view.max().sum())


@app.callback(
    Output("deaths_tot_txt", "children"),
    [Input("countries_slct", "value"), Input(
        "temporal_status_selector", "value"), ],
)
def update_deaths_text(countries_slct, temporal_status_selector):
    # Slice data
    slc_date = slice("2020-01-22", None)
    df_view = df.loc[(countries_slct, "deaths"),
                     slc_date].groupby(CTRY_K).sum().T
    # Get data
    return "{:,}".format(df_view.max().sum())


@app.callback(
    Output("recv_tot_txt", "children"),
    [Input("countries_slct", "value"), Input(
        "temporal_status_selector", "value"), ],
)
def update_recovered_text(countries_slct, temporal_status_selector):
    # Slice data
    slc_date = slice("2020-01-22", None)
    df_view = df.loc[(countries_slct, "recovered"),
                     slc_date].groupby(CTRY_K).sum().T
    # Get data
    return "{:,}".format(df_view.max().sum())


@app.callback(
    Output("act_tot_txt", "children"),
    [Input("countries_slct", "value"), Input(
        "temporal_status_selector", "value"), ],
)
def update_active_text(countries_slct, temporal_status_selector):
    # Slice data
    slc_date = slice("2020-01-22", None)
    df_cview = df.loc[(countries_slct, "confirmed"),
                      slc_date].groupby(CTRY_K).sum().T
    df_dview = df.loc[(countries_slct, "deaths"),
                      slc_date].groupby(CTRY_K).sum().T
    df_rview = df.loc[(countries_slct, "recovered"),
                      slc_date].groupby(CTRY_K).sum().T
    # Get data
    act = "{:,}".format(
        df_cview.max().sum() - df_dview.max().sum() - df_rview.max().sum()
    )
    return act


@app.callback(
    Output("mdl_eq_txt", "children"), [Input("model_data", "children"), ],
)
def upd_equation(mdl):
    """
    Updates the equation.

    It does not use the input parameters, they are used just to update
    the equation when a change occurs on them.

    Parameters:
    ctry (str): Country to pull data
    sts (str): Confirmed cases or deaths or recovered
    sldr_idx (int): The date index
    clr (int): Number of clicks on clear button

    Returns:
    dict: A plotly markdown text block

    """
    # Get fit data
    def convert(text):

        def toimage(x):
            if x[1] and x[-2] == r"$":
                x = x[2:-2]
                img = "\n<img src='https://math.now.sh?from={}'>\n"
                return img.format(urllib.parse.quote_plus(x))
            else:
                x = x[1:-1]
                return r"![](https://math.now.sh?from={})".format(
                    urllib.parse.quote_plus(x)
                )

        return re.sub(r"\${2}([^$]+)\${2}|\$(.+?)\$", lambda x: toimage(x.group()), text)

    try:
        fit = json.loads(mdl)
        # $$y = \frac{a}{1 + e^{-b*(x - x_0)}}$$

        x0 = fit["x0"]
        a = fit["a"]
        b = fit["b"]

        txt = r"$$y = \frac{" + "{:d}".format(int(a))
        txt += r"}{1 + e^{-" + "{:.3e}".format(b)
        txt += r"\cdot(x - " + "{:.2f}".format(x0)
        txt += r")}}$$"

        txt = convert(txt)
        # md = dcc.Markdown(txt)
        md = dcc.Markdown(txt, dangerously_allow_html=True)
        return md
    except:
        pass


@app.callback(
    Output("date_selected", "children"), [
        Input("date_slider_pred", "value"), ],
)
def upd_slider_slected(sldr_idx):
    return "Date selected: " + dates_arr[sldr_idx].strftime("%d/%m/%y")


@app.callback(
    [
        Output("pred_graph", "figure"),
        Output("curr_graph_data", "children"),
        Output("stored_graph_data", "children"),
        Output("grab_nclick_data", "children"),
        Output("clear_nclick_data", "children"),
    ],
    [
        Input("new_graph_data", "children"),
        Input("grab_graph_bt", "n_clicks"),
        Input("clear_graph_bt", "n_clicks"),
    ],
    [
        State("stored_graph_data", "children"),
        State("curr_graph_data", "children"),
        State("grab_nclick_data", "children"),
        State("clear_nclick_data", "children"),
    ],
)
def grab_clear_graph(nw_graph, g_clk, c_clk, graphs, curr_graph, g_clkd, c_clkd):
    """
    Store, clear and update the prediction graph.

    It handle the prediction graph update, the grab and clear buttons.

    Parameters:
    nw_graph (str): New graph that is stored in the 'new_graph_data' html.Div
    g_clk (int): Value from the button
    c_clk (int): Value from the button
    graphs (str): JSON input that contains a list of plotly figures (dictionary)
    curr_graph (str): JSON that stores the current graph
    g_clkd (str): Last number of clicks on the grab button
    c_clkd (str): Last number of clicks on the clear button

    Returns:
    dict: Prediction graph
    str: The current prediction graph as a JSON string
    str: List of selected graphs as a JSON string
    str: Number of clicks on grab button as a string
    str: Number of clicks on clear button as a string

    """

    # JSON -> variables
    try:
        nw_graph = json.loads(nw_graph)  # figure dictionary
    except:
        nw_graph = None
    try:
        graphs = json.loads(graphs)  # list of figures
    except:
        graphs = [None]
    try:
        curr_graph = json.loads(curr_graph)  # figure
    except:
        curr_graph = dict(data=[])
    try:
        g_clkd = int(g_clkd)  # integer
    except:
        g_clkd = 0
    try:
        c_clkd = int(c_clkd)  # integer
    except:
        c_clkd = 0

    # Check if g_clk and c_clk were not initialized yet
    g_clk = g_clk if g_clk else 0
    c_clk = c_clk if c_clk else 0

    # Handle the events
    if g_clk > g_clkd:
        # Grab button pressed
        graphs += [nw_graph]
        out_graph = curr_graph
    elif c_clk > c_clkd:
        # Clear button pressed
        graphs = [None]
        out_graph = nw_graph
        curr_graph = out_graph
    else:
        # New graph generated
        graphs_tmp = graphs + [nw_graph]
        data = []
        layout = []
        for gr in graphs_tmp:
            if gr:
                for gr_dt in gr["data"]:
                    data += [gr_dt]
                layout = gr["layout"]
        out_graph = dict(data=data, layout=layout)
        curr_graph = out_graph

    # Store the graphs in a JSON string and then a html.Div
    curr_graph = json.dumps(curr_graph)
    graphs = json.dumps(graphs)

    return out_graph, curr_graph, graphs, str(g_clk), str(c_clk)


@app.callback(
    Output("status_graph", "figure"),
    [Input("countries_slct", "value"), Input(
        "temporal_status_selector", "value"), ],
)
def upd_status_graph(ctry, sts):
    """
    It updates the status graph.

    Parameters:
    ctry (str): Country to pull data
    sts (str): Confirmed cases or deaths or recovered

    Returns:
    dict: The plotly figure

    """

    # Get layout
    layout_individual = copy.deepcopy(layout)

    # Slice data
    slc_date = slice("2020-01-22", None)

    df_view = df.loc[(ctry, sts), slc_date].groupby(CTRY_K).sum().T

    data = []
    # Generate plots
    for col in df_view.columns:
        # Configure plot
        data += [
            dict(
                type="scatter",
                mode="lines+markers",
                name=col,
                x=df_view.index.values,
                y=df_view[col].values,
                line=dict(shape="spline", smoothing=2, width=1,),
                marker=dict(symbol="diamond-open"),
            )
        ]
    layout_individual["title"] = sts
    layout_individual["xaxis_title"] = "Cases"
    layout_individual["yaxis_title"] = "Days"

    figure = dict(data=data, layout=layout_individual)
    return figure


@app.callback(
    Output("dpred_graph", "figure"),
    [
        Input("country_slct", "value"),
        Input("prediction_status_selector", "value"),
        Input("date_slider_pred", "value"),
    ],
)
def upd_dpred_graph(ctry, sts, sldr_idx):
    """
    Updates the derivative of the prediction graph.

    Parameters:
    ctry (str): Country to pull data
    sts (str): Confirmed cases or deaths or recovered
    sldr_idx (int): The date index

    Returns:
    str: The plotly figure is parsed into a json string

    """

    # Get layout
    layout_individual = copy.deepcopy(layout)

    start_date = "2020-01-22"
    end_date = str(dates_arr[sldr_idx].date())
    n_days = 30

    # Slice the data
    slc_date = slice(start_date, end_date)
    df_view = df.loc[(ctry, sts), slc_date]

    day_start = np.where(df_view > 0)[1][0]
    day_start = day_start - 7 if day_start > 7 else day_start

    # Date operations
    last_day = pd.to_datetime(df_view.columns.values[-1])
    list_day = [last_day + timedelta(days=n) for n in range(1, n_days)]
    pred_day = [str(day.date()) for day in list_day]

    # Sigmoid fit
    x0, a, b, fit = fit_curve(df, ctry, sts, start_date, end_date=end_date)

    n_days += fit["y_data"].shape[0]
    lw_band, up_band, y_nom, y_std = dcases_ci(x0, a, b, n_days, fit["y_data"])

    fit["y_data"] = np.diff(fit["y_data"])

    days = df_view.columns.values[day_start:]
    # Generate plots
    data = [
        dict(
            type="scatter",
            mode="markers",
            name="Today",
            x=[str(last_day.date())],
            y=[fit["y_data"][-1]],
            marker=dict(size=12, color="red",),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="Real cases",
            x=days,
            y=fit["y_data"][day_start:],
            line=dict(
                shape="spline",
                smoothing=2,
                width=3,
                # color='#fac1b7'
            ),
            marker=dict(symbol="diamond-open"),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="Fitted model",
            x=np.hstack([days, pred_day]),
            y=y_nom[day_start:],
            line=dict(
                shape="spline",
                smoothing=2,
                width=2,
                # color='#a9bb95'
            ),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="95% Confidence region",
            x=np.hstack([days, pred_day]),
            y=y_nom[day_start:] - 1.96 * y_std[day_start:],
            line=dict(shape="spline", smoothing=2,
                      width=1, color="gray", dash="dash"),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="95% Confidence region",
            showlegend=False,
            x=np.hstack([days, pred_day]),
            y=y_nom[day_start:] + 1.96 * y_std[day_start:],
            line=dict(shape="spline", smoothing=2,
                      width=1, color="gray", dash="dash"),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="95% Prediction band",
            x=np.hstack([days, pred_day]),
            y=up_band[day_start:],
            line=dict(shape="spline", smoothing=2,
                      width=1, color="black", dash="dash"),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="95% Prediction band",
            showlegend=False,
            x=np.hstack([days, pred_day]),
            y=lw_band[day_start:],
            line=dict(shape="spline", smoothing=2,
                      width=1, color="black", dash="dash"),
        ),
    ]

    peak = max(up_band.max(), (y_nom + 1.96 * y_std).max())
    title = "{} - R²: {:.4f} - Peak: {:d}".format(ctry, fit["r2"], int(peak))
    layout_individual["title"] = title
    layout_individual["xaxis_title"] = "New cases"
    layout_individual["yaxis_title"] = "Days"

    figure = dict(data=data, layout=layout_individual)
    return figure


@app.callback(
    [Output("new_graph_data", "children"), Output("model_data", "children")],
    [
        Input("country_slct", "value"),
        Input("prediction_status_selector", "value"),
        Input("date_slider_pred", "value"),
    ],
)
def upd_pred_graph(ctry, sts, sldr_idx):
    """
    Updates the prediction graph.

    Parameters:
    ctry (str): Country to pull data
    sts (str): Confirmed cases or deaths or recovered
    sldr_idx (int): The date index

    Returns:
    str: The plotly figure is parsed into a json string

    """

    # Get layout
    layout_individual = copy.deepcopy(layout)

    start_date = "2020-01-22"
    end_date = str(dates_arr[sldr_idx].date())
    n_days = 30

    # Slice the data
    slc_date = slice(start_date, end_date)
    df_view = df.loc[(ctry, sts), slc_date]

    day_start = np.where(df_view > 0)[1][0]
    day_start = day_start - 7 if day_start > 7 else day_start

    # Date operations
    last_day = pd.to_datetime(df_view.columns.values[-1])
    list_day = [last_day + timedelta(days=n) for n in range(1, n_days)]
    pred_day = [str(day.date()) for day in list_day]

    # Sigmoid fit
    x0, a, b, fit = fit_curve(df, ctry, sts, start_date, end_date=end_date)

    n_days += fit["y_data"].shape[0]
    lw_band, up_band, y_nom, y_std = cases_ci(x0, a, b, n_days, fit["y_data"])

    days = df_view.columns.values[day_start:]
    # Generate plots
    data = [
        dict(
            type="scatter",
            mode="markers",
            name="Today",
            x=[str(last_day.date())],
            y=[fit["y_data"][-1]],
            marker=dict(size=12, color="red",),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="Real cases",
            x=days,
            y=fit["y_data"][day_start:],
            line=dict(
                shape="spline",
                smoothing=2,
                width=3,
                # color='#fac1b7'
            ),
            marker=dict(symbol="diamond-open"),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="Fitted model",
            x=np.hstack([days, pred_day]),
            y=y_nom[day_start:],
            line=dict(
                shape="spline",
                smoothing=2,
                width=2,
                # color='#a9bb95'
            ),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="95% Confidence region",
            x=np.hstack([days, pred_day]),
            y=y_nom[day_start:] - 1.96 * y_std[day_start:],
            line=dict(shape="spline", smoothing=2,
                      width=1, color="gray", dash="dash"),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="95% Confidence region",
            showlegend=False,
            x=np.hstack([days, pred_day]),
            y=y_nom[day_start:] + 1.96 * y_std[day_start:],
            line=dict(shape="spline", smoothing=2,
                      width=1, color="gray", dash="dash"),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="95% Prediction band",
            x=np.hstack([days, pred_day]),
            y=up_band[day_start:],
            line=dict(shape="spline", smoothing=2,
                      width=1, color="black", dash="dash"),
        ),
        dict(
            type="scatter",
            mode="lines",
            name="95% Prediction band",
            showlegend=False,
            x=np.hstack([days, pred_day]),
            y=lw_band[day_start:],
            line=dict(shape="spline", smoothing=2,
                      width=1, color="black", dash="dash"),
        ),
    ]

    # Generate the graph title
    title = "{} - R²: {:.4f} - Projection cases: {:d}".format(
        ctry, fit["r2"], int(fit["coef"][1])
    )

    layout_individual["title"] = title
    layout_individual["xaxis_title"] = "Cases"
    layout_individual["yaxis_title"] = "Days"
    figure = dict(data=data, layout=layout_individual)

    fit = {"x0": x0.nominal_value, "a": a.nominal_value, "b": b.nominal_value}
    return json.dumps(figure, cls=NpEncoder), json.dumps(fit)


# Main
if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0")
    # app.run_server(debug=True)
