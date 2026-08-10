from src.models.predict import prepare_input, RAW_FEATURES

def test_prediction_input_shape():
    values = [
        7.4, 0.70, 0.00, 1.9, 0.076,
        11.0, 34.0, 0.9978, 3.51, 0.56, 9.4
    ]

    sample = prepare_input(values)

    assert sample.shape[0] == 1
    assert all(feature in sample.columns for feature in RAW_FEATURES)
