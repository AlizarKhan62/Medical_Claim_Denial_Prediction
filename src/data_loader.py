# src/data_loader.py
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from given CSV path."""
    try:
        df = pd.read_csv(path)
        print(f"Data loaded successfully from {path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def save_data(df: pd.DataFrame, path: str):
    """Save processed dataset to given path."""
    try:
        df.to_csv(path, index=False)
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error saving data: {e}")
