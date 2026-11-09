import pandas as pd

def time_split(df: pd.DataFrame,
               train_end: str = "2015-09-01",
               val_end: str = "2015-11-01",
               gap_hours: int = 24):
   
   
    """Train: before train_end. Validation: train_end to val_end. Test: val_end onwards.
    A gap is cut off the end of train and validation so labels never peek into the next period."""
    gap = pd.Timedelta(hours=gap_hours)
    train_end, val_end = pd.Timestamp(train_end), pd.Timestamp(val_end)

    train = df[df["datetime"]<train_end - gap]
    val = df[(df["datetime"] >= train_end) & (df["datetime"] < val_end - gap)]
    test = df[df["datetime"] >= val_end]
    return train ,val, test 