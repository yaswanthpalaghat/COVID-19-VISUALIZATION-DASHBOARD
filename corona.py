from scipy import stats
import uncertainties.unumpy as unp
import uncertainties as unc
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
import datetime
import sklearn.metrics as mt
from utils import *

# Confidence region and prediction band: https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics
# Confidence region: https://en.wikipedia.org/wiki/1.96

sns.set()

# Load the cases
df = pd.read_hdf("./data/covid19.h5", "covid19_data")

# Helper of all data slice
slca = slice(None)

# %%
# Create plot
plt.tight_layout()
countries = ["China", "US", "Japan", "Italy", "Brazil", "Germany", "Spain"]
date = "2020-01-22"
status = "confirmed"

# Cases
df.loc[(countries, status), date:].groupby(cnt_rgn).sum().T.plot()
# Derivative of the cases
df.loc[(countries, status), date:].groupby(cnt_rgn).sum().diff(axis=1).T.plot()


# %%
countries = ["Spain"]

y_data = df.loc[(countries, status), date:].groupby(cnt_rgn).sum().values[0]
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
x0, a, b, = unc.correlated_values(popt, pcov)

print("R²: {:.5f}".format(r2))
print("Coefficients: x0: {}  a: {: 2f}  b: {}".format(x0, a, b))

# Regression confidence interval
# Generate an array of days
xp = np.arange(0, 100)
# Propagate the uncertainty from the coefficients to the prediction
y_unc = sig_unc(xp, x0, a, b)

y_nom = unp.nominal_values(y_unc)
y_std = unp.std_devs(y_unc)

# Parameters nominal value
p = unp.nominal_values([x0, a, b])

x_data = np.arange(y_data.shape[0])
lw_band, up_band = pred_band(xp, x_data, y_data, p, sig)
dlw_band, dup_band = predband(xp, x_data[0:-1], np.diff(y_data), p, dsig_unc)

# Initialise plot
fig, ax = plt.subplots(1, 2, num=2)
plt.tight_layout()

# Fit quality
ax[0].cla()
ax[0].plot(xp, nom + 1.96 * std, "b--", lw=1, alpha=0.7, label="95% Confidence region ")
ax[0].plot(xp, nom - 1.96 * std, "b--", lw=1, alpha=0.7)
ax[0].plot(xp, lw_band, "k--", lw=1, alpha=0.7, label="95% Prediction band")
ax[0].plot(xp, up_band, "k--", lw=1, alpha=0.7)
ax[0].plot(xp, nom, label="Adjusted")
ax[0].plot(x_data, y_data, label="Real")
ax[0].legend()
title = "{} - Fit quality: {:.4f} - Projection cases: {:d}".format(
    countries[0], r2_train, int(popt[1])
)
ax[0].set_title(title)
ax[0].set_xlabel("Days")
ax[0].set_ylabel(status)

ax[1].cla()
ax[1].plot(
    xp[0:-1], dnom + 1.96 * dstd, "b--", lw=1, alpha=0.7, label="95% Confidence region "
)
ax[1].plot(xp[0:-1], dnom - 1.96 * dstd, "b--", lw=1, alpha=0.7)
ax[1].plot(xp[0:-1], dnom, "k", label="Prediction")
ax[1].plot(x_data[0:-1], np.diff(y_data), label="Real")
ax[1].plot(xp, dlw_band, "k--", lw=1, alpha=0.7, label="95% Prediction band")
ax[1].plot(xp, dup_band, "k--", lw=1, alpha=0.7)
ax[1].legend()
ax[1].set_title("Increase of cases per day")
ax[1].set_ylabel("Cases")
ax[1].set_xlabel("Days")


# %%
# Prediction on past basis
day = -2
future = 120
y_train = y_data[0::] if day == -1 else y_data[0:day]
x_train = np.arange(0, y_train.shape[0])

popt, pcov = opt.curve_fit(sig, x_train, y_train)
y_fit = sig(x_data, *popt)
r2_train = mt.r2_score(y_data, y_fit)

# Model from trained data
y_trained = tan(x_train, *popt)
# Model used to predict
proj_day = x_train[day] if day < 0 else day
x_proj = np.arange(proj_day, future)
y_fit = sig(x_proj, *popt)

# Plotting
plt.cla()
plt.plot(x_proj, y_fit, linestyle="--", label="Projection")
plt.plot(x_train, y_trained, label="Adjusted")
plt.plot(x_data, y_data, alpha=0.7, label="Real")
plt.legend()
title = "{} - Fit quality: {:.4f} - Projection cases: {:d}".format(
    # countries[0], r2_train, int(popt[-1])*2)
    countries[0],
    r2_train,
    int(popt[1]) * 2,
)
plt.title(title)

plt.cla()
plt.plot(np.diff(y_trained))
plt.plot(np.diff(y_data))


# %matplotlib qt5
