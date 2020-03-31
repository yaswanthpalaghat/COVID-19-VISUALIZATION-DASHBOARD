import numpy as np
import pandas as pd

data_path = "./data/csse_covid_19_data/csse_covid_19_time_series/"

cv_cfm = "time_series_covid19_confirmed_global.csv"
cv_dth = "time_series_covid19_deaths_global.csv"
cv_rec = "time_series_covid19_recovered_global.csv"

# Read .csv files
df_cfm = pd.read_csv(data_path + cv_cfm)
df_dth = pd.read_csv(data_path + cv_dth)
df_rec = pd.read_csv(data_path + cv_rec)

# Set index as Country/Region
cnt_rgn = "Country/Region"
df_cfm = df_cfm.set_index(cnt_rgn)
df_dth = df_dth.set_index(cnt_rgn)
df_rec = df_rec.set_index(cnt_rgn)

# Join the datasets
df = pd.concat([df_cfm, df_dth, df_rec], keys=["confirmed", "deaths", "recovered"])
df = df.swaplevel(1, 0)
df = df.sort_index()


date_col = zip(pd.to_datetime(df.columns[3:]), df.columns[3:])
df = df.rename(columns={ufmt: str(fmt.date()) for fmt, ufmt in date_col})
df.to_hdf("./data/covid19.h5", "covid19_data")
