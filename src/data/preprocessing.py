from pathlib import Path
import pandas as pd

TARGET = "quality"

def validate_dataframe(df: pd.DataFrame) -> None:
    if TARGET not in df.columns:
        raise ValueError("Target column 'quality' is missing.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    numeric_columns = df.columns
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in numeric_columns):
        raise ValueError("All dataset columns must be numeric.")

def create_quality_class(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def classify(value):
        if value <= 4:
            return 0
        if value <= 6:
            return 1
        return 2

    out["quality_class"] = out[TARGET].apply(classify)
    return out

def preprocess_and_save(df: pd.DataFrame, path: str) -> pd.DataFrame:
    validate_dataframe(df)

    out = df.drop_duplicates().reset_index(drop=True)
    out = create_quality_class(out)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    return out
