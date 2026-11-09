from src.data_loading import load_raw_data
from src.labeling import add_failure_label
import pandas as pd

data = load_raw_data()
df = add_failure_label(data["telemetry"], data["failures"])

print("rows:", len(df))
print("positive rows:", df["label"].sum())
print("positive rate:{:.2%}".format(df["label"].mean()))
print("machine with any positive label:", df.loc[df["label"]==1, "machineID"].nunique())


# Eyeball one failure : rows just before it
first_fail = data["failures"].iloc[0]
m,t = first_fail["machineID"], first_fail["datetime"]
window = df.loc[(df["machineID"] == m)&
                (df["datetime"] >=t - pd.Timedelta(hours=26))&
                (df["datetime"] <=t + pd.Timedelta(hours=2))]
print(window[["datetime", "label", "hours_to_failure"]].to_string())