import streamlit as st

from src.models.predict import load_model, predict, RAW_FEATURES
from app.components.prediction import render_prediction
from app.components.charts import render_input_summary

MODEL_PATH = "models/wine_quality_best_model.joblib"

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="",
    layout="wide",
)

st.title("Wine Quality Prediction")
st.caption("Machine-learning classification dashboard")

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

defaults = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}

ranges = {
    "fixed acidity": (0.0, 20.0, 0.1),
    "volatile acidity": (0.0, 2.0, 0.01),
    "citric acid": (0.0, 2.0, 0.01),
    "residual sugar": (0.0, 30.0, 0.1),
    "chlorides": (0.0, 1.0, 0.001),
    "free sulfur dioxide": (0.0, 100.0, 1.0),
    "total sulfur dioxide": (0.0, 300.0, 1.0),
    "density": (0.9, 1.1, 0.0001),
    "pH": (2.0, 5.0, 0.01),
    "sulphates": (0.0, 2.0, 0.01),
    "alcohol": (5.0, 20.0, 0.1),
}

st.subheader("Wine measurements")

columns = st.columns(3)
values = []

for index, feature in enumerate(RAW_FEATURES):
    minimum, maximum, step = ranges[feature]

    with columns[index % 3]:
        value = st.number_input(
            feature.title(),
            min_value=minimum,
            max_value=maximum,
            value=defaults[feature],
            step=step,
        )
        values.append(value)

if st.button("Predict Wine Quality", type="primary", use_container_width=True):
    result = predict(model, values)

    left, right = st.columns([1, 1])

    with left:
        render_prediction(result)

    with right:
        st.subheader("Input summary")
        render_input_summary(values, [f.title() for f in RAW_FEATURES])
