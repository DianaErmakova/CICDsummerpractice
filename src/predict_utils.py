import os
import joblib
import pandas as pd
from src.data_loader import preprocess_data

# Пути
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.joblib')
FEATURES_PATH = os.path.join(BASE_DIR, '..', 'models', 'features.joblib')

# Загрузка
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)


def predict_from_dict(data: dict):
    df = pd.DataFrame([data])
    X, _ = preprocess_data(df)
    X = X.reindex(columns=features, fill_value=0)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return bool(pred)
