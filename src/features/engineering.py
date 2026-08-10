import numpy as np
import pandas as pd

BASE_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

MODEL_FEATURES = BASE_FEATURES + [
    "acidity_ratio",
    "sulfur_dioxide_ratio",
    "sugar_density_ratio",
    "sulphate_chloride_ratio",
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    eps = 1e-8

    out["acidity_ratio"] = (
        out["fixed acidity"] /
        (out["volatile acidity"] + eps)
    )

    out["sulfur_dioxide_ratio"] = (
        out["free sulfur dioxide"] /
        (out["total sulfur dioxide"] + eps)
    )

    out["sugar_density_ratio"] = (
        out["residual sugar"] /
        (out["density"] + eps)
    )

    out["sulphate_chloride_ratio"] = (
        out["sulphates"] /
        (out["chlorides"] + eps)
    )

    out = out.replace([np.inf, -np.inf], np.nan)

    return out
