import pandas as pd
from src.data.preprocessing import validate_dataframe, create_quality_class

def test_dataset_validation():
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
        "quality": [5],
    })
    validate_dataframe(df)

def test_quality_classification():
    df = pd.DataFrame({"quality": [4, 5, 6, 7, 8]})
    result = create_quality_class(df)
    assert result["quality_class"].tolist() == [0, 1, 1, 2, 2]
