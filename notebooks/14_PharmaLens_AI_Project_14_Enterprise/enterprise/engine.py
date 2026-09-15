from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

def load_processed_data():
    parquet_files = list(PROCESSED_DATA.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No processed parquet file found in {PROCESSED_DATA}")
    return pd.read_parquet(parquet_files[0])

def load_private_dataset(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Unsupported dataset format.")

def call_existing(function, df, *args, **kwargs):
    return function(df, *args, **kwargs)
