import pandas as pd
from src.utils import print_header

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from a CSV file.
    Args:
        path (str): Path to the CSV file.
    """
    print_header("LOADING DATASET")
    df = pd.read_csv(path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_dataset("data/phising.csv")
    print(df.head())