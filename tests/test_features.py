import pandas as pd
from src.features.engineering import engineer_features

def test_feature_engineering():
    df = pd.DataFrame({
        "fixed acidity": [7.4],
        "volatile acidity": [0.7],
        "citric acid": [0.0],
        "residual sugar": [1.9],
        "chlorides": [0.076],
        "free sulfur dioxide": [11],
        "total sulfur dioxide": [34],
        "density": [0.9978],
        "pH": [3.51],
        "sulphates": [0.56],
        "alcohol": [9.4],
    })

    result = engineer_features(df)

    assert "acidity_ratio" in result.columns
    assert "sulfur_dioxide_ratio" in result.columns
    assert "sugar_density_ratio" in result.columns
    assert "sulphate_chloride_ratio" in result.columns
