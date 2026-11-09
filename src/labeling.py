import pandas as pd

from src.data_loading import load_raw_data


def add_failure_label(telemetry: pd.DataFrame, 
                      failures: pd.DataFrame, 
                      horizon_hours: int =24) -> pd.DataFrame:
    """Label = 1 if the machine fails within the next `horizon_hours`."""
    fail_times = (
        failures[["machineID", "datetime"]].drop_duplicates().rename(columns={"datetime":"next_failure_time"}).sort_values("next_failure_time")
    )

    tel = telemetry.sort_values("datetime")

    out = pd.merge_asof(
        tel,
        fail_times,
        left_on="datetime",
        right_on="next_failure_time",
        by="machineID",
        direction="forward",
        allow_exact_matches=False
    )

    out["hours_to_failure"] = (
        (out["next_failure_time"] - out["datetime"]).dt.total_seconds() / 3600
    )
    out["label"] = (out["hours_to_failure"] <= horizon_hours).astype(int)

    return out.sort_values(["machineID", "datetime"]).reset_index(drop=True)