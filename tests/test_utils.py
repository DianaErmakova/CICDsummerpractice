from src.predict_utils import predict_from_dict


def test_predict_from_dict_valid_input():
    sample_input = {
        "age": 63,
        "sex": 1,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "ca": 0,
        "cp": 3,
        "restecg": 0,
        "slope": 0,
        "thal": 1
    }

    prediction = predict_from_dict(sample_input)
    assert isinstance(prediction, bool)
