from src.data_loading import load_raw_data

data = load_raw_data()

for name, df in data.items():
    print(f"\n==== {name} ====")
    print("shape:", df.shape)
    print(df.dtypes.to_string())
    if "datetime" in df.columns:
        print("datetime range:", df["datetime"].min(), "->", df["datetime"].max())
    if "machineID" in df.columns:
        print("unique machines:", df["machineID"].nunique())
    print(df.head(3).to_string())

print(f"Failures per machine:")
print(data["failures"].groupby("machineID").size().describe())