import pandas as pd
from src.data_loader import preprocess_data


def test_preprocess_data():
    # Минимальный валидный набор признаков из heart dataset
    data = {
        'age': [63, 45],
        'sex': [1, 0],
        'trestbps': [145, 130],
        'chol': [233, 250],
        'fbs': [1, 0],
        'thalach': [150, 160],
        'exang': [0, 1],
        'oldpeak': [2.3, 1.4],
        'ca': [0, 1],
        'cp': [3, 2],
        'restecg': [0, 1],
        'slope': [0, 2],
        'thal': [1, 2],
        'condition': [1, 0]
    }

    df = pd.DataFrame(data)
    X, y = preprocess_data(df)

    # Проверим форму
    assert X.shape[0] == 2
    assert y.tolist() == [1, 0]
    assert any(
        col.startswith('cp_') for col in X.columns
    )
    assert any(
        col.startswith('thal_') for col in X.columns
    )
