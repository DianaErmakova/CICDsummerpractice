import os
import joblib
import pandas as pd
from datetime import datetime
from src.data_loader import preprocess_data

# Пути
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'model.joblib')
SCALER_PATH = os.path.join(BASE_DIR, '..', 'models', 'scaler.joblib')
FEATURES_PATH = os.path.join(BASE_DIR, '..', 'models', 'features.joblib')
CSV_PATH = os.path.join(BASE_DIR, '..', 'data', 'heart_cleveland_upload.csv')
PRED_PATH = os.path.join(BASE_DIR, '..', 'predictions.csv')
REPORT_PATH = os.path.join(BASE_DIR, '..', 'report.html')

# Проверка наличия файлов
for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Отсутствует файл: {path}")

# Загрузка
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

# Данные
raw_data = pd.read_csv(CSV_PATH).head(5)
X, _ = preprocess_data(raw_data)
X = X.reindex(columns=feature_names, fill_value=0)
X_scaled = scaler.transform(X)

# Предсказания
preds = model.predict(X_scaled)
raw_data['Condition_Predicted'] = preds
raw_data.to_csv(PRED_PATH, index=False)
print(f"✅ Предсказания сохранены в {PRED_PATH}")

date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# HTML-отчёт
html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Отчёт о предсказаниях (Heart Dataset)</title>
</head>
<body>
    <h1>🫀 Heart Disease Inference Report</h1>
    <p><strong>Дата:</strong> {date_str}</p>
    <p><strong>Количество образцов:</strong> {len(preds)}</p>
    <p><strong>Предсказания:</strong></p>
    {raw_data[['Condition_Predicted']].to_html(index=False)}
</body>
</html>
"""

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"📄 Отчёт сохранён в {REPORT_PATH}")
