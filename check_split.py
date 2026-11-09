from src.data_loading import load_raw_data
from src.labeling import add_failure_label
from src.split import time_split

data =load_raw_data()
df = add_failure_label(data["telemetry"], data["failures"])
train, val, test = time_split(df)

for name, part in [("train", train), ("val",val), ("test", test)]:
    print(f"{name:5s} rows={len(part):7d} "
          f"{part['datetime'].min()} -> {part['datetime'].max()}  "
          f"positive rate={part['label'].mean():.2%}")