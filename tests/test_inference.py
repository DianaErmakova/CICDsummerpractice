import os
import pandas as pd
from datetime import datetime
from src.predict_utils import model, scaler, features, preprocess_data


def test_inference_prediction_and_report():
    BASE_DIR = os.path.dirname(__file__)
    CSV_PATH = os.path.join(
        BASE_DIR,
        '..',
        'data',
        'heart_cleveland_upload.csv'
    )
    PRED_PATH = os.path.join(BASE_DIR, '..', 'predictions.csv')
    REPORT_PATH = os.path.join(BASE_DIR, '..', 'report.html')

    raw_data = pd.read_csv(CSV_PATH).head(5)
    X, _ = preprocess_data(raw_data)
    X = X.reindex(columns=features, fill_value=0)
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    raw_data['Condition_Predicted'] = preds
    raw_data.to_csv(PRED_PATH, index=False)

    # Проверка, что предсказания - массив нужной длины
    assert len(preds) == 5

    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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

    # Проверяем, что файл отчёта создался
    assert os.path.exists(REPORT_PATH)
