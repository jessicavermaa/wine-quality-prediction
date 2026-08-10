from pathlib import Path
import joblib
import pandas as pd

from src.features.engineering import engineer_features

CLASS_NAMES = {
    0: "Poor",
    1: "Average",
    2: "Good",
}

RAW_FEATURES = [
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

def load_model(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run python run_pipeline.py first."
        )
    return joblib.load(path)

def prepare_input(values):
    row = dict(zip(RAW_FEATURES, values))
    df = pd.DataFrame([row])
    return engineer_features(df)

def predict(model, values):
    sample = prepare_input(values)

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]

    return {
        "quality": CLASS_NAMES[prediction],
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(probabilities))
        },
    }
