import os
import joblib
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_and_preprocess


# Пути
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.joblib')
METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.json')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'heart_cleveland_upload.csv')


def train_and_evaluate_model(data_path=DATA_PATH):
    # Загрузка и подготовка
    X, y = load_and_preprocess(data_path)
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4)
    }

    return model, scaler, feature_names, metrics


if __name__ == "__main__":
    model, scaler, feature_names, metrics = train_and_evaluate_model()

    print("📊 Метрики обучения:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Сохраняем артефакты
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names.tolist(), FEATURES_PATH)

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "features": feature_names.tolist(),
        **metrics
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✅ Модель и объекты сохранены.")
