import pandas as pd

SENSORS = ["volt", "rotate", "pressure", "vibration"]
WINDOWS = [3, 12, 24, 48, 72]

# Columns that must never be model inputs
NON_FEATURES = ["datetime", "machineID", "label", "next_failure_time", "hours_to_failure"]


def add_rolling_features(df: pd.DataFrame,
                         sensors = SENSORS,
                         windows = WINDOWS) -> pd.DataFrame:
    """Add rolling mean/std/min/max per machine, using past hours only."""
    df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    g=df.groupby("machineID")

    new_cols={}

    for s in sensors:
        for w in windows:
            roll = g[s].rolling(w, min_periods=w)
            new_cols[f"{s}_mean_{w}h"] = roll.mean().reset_index(level=0, drop=True)
            new_cols[f"{s}_std_{w}h"] = roll.std().reset_index(level=0, drop=True)
            new_cols[f"{s}_min_{w}h"] = roll.min().reset_index(level=0, drop=True)
            new_cols[f"{s}_max_{w}h"] = roll.max().reset_index(level=0, drop=True)

    return pd.concat([df, pd.DataFrame(new_cols)], axis=1)

def get_feature_columns(df: pd.DataFrame) -> list:
    """Everything except identifiers, the label, and leak columns."""
    return [c for c in df.columns if c not in NON_FEATURES]
