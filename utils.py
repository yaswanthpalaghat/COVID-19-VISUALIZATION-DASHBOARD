import json
import numpy as np
import pandas as pd
from scipy import stats
import scipy.optimize as opt
import uncertainties as unc
import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
import sklearn.metrics as mt

# import seaborn as sns

# sns.set()

CTRY_K = "Country/Region"


def sig(x, x0, a, b):
    # Sigmoid with freedom on amplitude, X-axis and inclination
    return a / (1 + np.exp(-b * (x - x0))) + 0


def sig_unc(x, x0, a, b):
    # Same sigmoid but with uncertainty package
    return a / (1 + unp.exp(-b * (x - x0))) + 0


def dsig_unc(x, x0, a, b):
    # Derivative of the sigmoid function with uncertainty package
    return (a * b * unp.exp(-b * (x - x0))) / ((1 + unp.exp(-b * (x - x0))) ** 2)


# TODO: put credits
# TODO: rewrite it to understand better
def pred_band(x, xd, yd, p, func, conf=0.95):
    """
    Summary line.

    Extended description of function.

    https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics

    Parameters:
    arg1 (int): Description of arg1

    Returns:
    int: Description of return value

    """

    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf  # significance
    N = xd.size  # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Std of an individual measurement
    se = np.sqrt(1.0 / (N - var_n) * np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0 + (1.0 / N) + (sx / sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb


# Get the Curves
def fit_curve(df, country, status, start_date, end_date=None):
    """
    Summary line.

    Extended description of function.

    Parameters:
    arg1 (int): Description of arg1

    Returns:
    int: Description of return value

    """

    # Select the data
    slc_date = slice(start_date, end_date)
    y_data = df.loc[(country, status), slc_date].groupby(
        CTRY_K).sum().values[0]

    # Generate a dummy x_data
    x_data = np.arange(0, y_data.shape[0])

    # Set initial guesses for the curve fit
    x0_0 = x_data[np.where(y_data > 0)[0][0]]  # Day of the first case
    a_0 = y_data.max()  # Current number of cases
    b_0 = 0.1  # Arbitrary
    p0 = [x0_0, a_0, b_0]
    # Fit the curve
    popt, pcov = opt.curve_fit(sig, x_data, y_data, p0=p0)

    # Evaluate the curve fit to calculate the R²
    y_fit = sig(x_data, *popt)
    r2 = mt.r2_score(y_data, y_fit)

    # Estimate the uncertainty of the obtained coefficients
    x0, a, b, = unc.correlated_values(popt, pcov)
    # Store the fit information
    fit = {
        "r2": r2,
        "x0": x0,
        "a": a,
        "b": b,
        "coef": popt,
        "coef_cov": pcov,
        "y_data": y_data,
        "x_data": slc_date,
    }
    return x0, a, b, fit


def cases_ci(x0, a, b, n_days, y_data, conf=0.95):
    """
    Summary line.

    Extended description of function.

    Parameters:
    arg1 (int): Description of arg1

    Returns:
    int: Description of return value

    """
    # Generate an array of days
    xp = np.arange(0, n_days)
    # Propagate the uncertainty from the coefficients to the prediction
    y_unc = sig_unc(xp, x0, a, b)

    y_nom = unp.nominal_values(y_unc)
    y_std = unp.std_devs(y_unc)

    # Parameters nominal value
    p = unp.nominal_values([x0, a, b])

    x_data = np.arange(y_data.shape[0])
    lw_band, up_band = pred_band(xp, x_data, y_data, p, sig)
    return lw_band, up_band, y_nom, y_std


def dcases_ci(x0, a, b, n_days, y_data, conf=0.95):
    """
    Summary line.

    Extended description of function.

    Parameters:
    arg1 (int): Description of arg1

    Returns:
    int: Description of return value

    """
    # Generate an array of days
    xp = np.arange(0, n_days)
    # Propagate the uncertainty from the coefficients to the prediction
    y_unc = sig_unc(xp, x0, a, b)

    # Calulate the derivatives
    dy_data = np.diff(y_data)
    dy_unc = np.diff(y_unc)

    dy_nom = unp.nominal_values(dy_unc)
    dy_std = unp.std_devs(dy_unc)

    # Parameters nominal value
    p = unp.nominal_values([x0, a, b])
    x_data = np.arange(y_data.shape[0] - 1)
    dlw_band, dup_band = pred_band(xp, x_data, dy_data, p, dsig_unc)
    return dlw_band, dup_band, dy_nom, dy_std


class NpEncoder(json.JSONEncoder):
    # https: // stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


# def gen_graph(graphs):
#     """
#     List of plotly figures -> plotly figure

#     Convert a list of plotly figures (dictionaries) to a single plotly figure.

#     Parameters:
#     graph (list): list of plotly figures

#     Returns:
#     dict: A plotly figure

#     """
#     # if type(graphs) == list:
#     #     graphs = [i for i in graphs if i]
#     #     if len(graphs) > 0:
#     data = []
#     layout = []
#     for gr in graphs:
#         if gr:
#             for gr_dt in gr["data"]:
#                 data += [gr_dt]
#             layout = gr["layout"]

#     return dict(data=data, layout=layout)


# def convert(text):
#     def toimage(x):
#         if x[1] and x[-2] == r"$":
#             x = x[2:-2]
#             img = "\n<img src='https://math.now.sh?from={}'>\n"
#             return img.format(urllib.parse.quote_plus(x))
#         else:
#             x = x[1:-1]
#             return r"![](https://math.now.sh?from={})".format(
#                 urllib.parse.quote_plus(x)
#             )

#     return re.sub(r"\${2}([^$]+)\${2}|\$(.+?)\$", lambda x: toimage(x.group()), text)
