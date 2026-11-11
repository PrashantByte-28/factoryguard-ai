from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent /"data"/"raw"


def load_raw_data(raw_dir: Path = RAW_DIR) -> dict:
    """Load the five Azure PdM files, parse datetimes, and sort them.""" 
    telemetry = pd.read_csv(raw_dir / "PdM_telemetry.csv",parse_dates=["datetime"])
    failures = pd.read_csv(raw_dir / "PdM_failures.csv",parse_dates=["datetime"])
    errors = pd.read_csv(raw_dir / "PdM_errors.csv",parse_dates=["datetime"])
    maint = pd.read_csv(raw_dir / "PdM_maint.csv",parse_dates=["datetime"])
    machines = pd.read_csv(raw_dir / "PdM_machines.csv")

    telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    failures = failures.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    return {
        "telemetry": telemetry,
        "failures": failures,
        "errors": errors,
        "maint": maint,
        "machines": machines,   

    }