from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

def download_dataset(path: str) -> pd.DataFrame:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        print("Downloading Wine Quality dataset...")
        urlretrieve(DATA_URL, destination)

    return pd.read_csv(destination, sep=";")

if __name__ == "__main__":
    df = download_dataset("data/raw/winequality-red.csv")
    print(f"Loaded {len(df)} records.")
