from src.data_loading import load_raw_data
from src.labeling import add_failure_label
from src.split import time_split
from src.features import add_rolling_features, get_feature_columns
import pandas as pd

data = load_raw_data()
df = add_failure_label(data["telemetry"], data["failures"])
df = add_rolling_features(df)

feature_cols = get_feature_columns(df)
print("number of features:", len(feature_cols))
print("first 8:", feature_cols[:8])

train, val, test = time_split(df)
print("train/val/test rows:", len(train), len(val), len(test))
print("\n NAN count in train features (top 5):")
print(train[feature_cols].isna().sum().sort_values(ascending=False).head(5))

# Sanity check: Does vibration look different before failure?
print("\nMean vibration_mean_24h by label (train):")
print(train.groupby("label")["vibration_mean_24h"].mean())