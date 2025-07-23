# 🫀 Heart Disease Predictor

Простой ML-проект для предсказания наличия сердечного заболевания на основе медицинских показателей.

## 📦 Структура проекта

```

.
├── data/                   # Исходные данные (CSV)
├── models/                 # Сохранённая модель, scaler, метаданные
├── src/
│   ├── app.py              # FastAPI приложение
│   ├── data\_loader.py      # Загрузка и предобработка данных
│   ├── inference.py        # Скрипт предсказания и HTML-отчёт
│   ├── model.py            # Определение модели
│   ├── predict_utils.py    # Предсказание через загруженные артефакты
│   └── train.py            # Обучение модели
├── tests/                  # Тесты
├── .github/workflows/ci.yml # CI/CD пайплайн
├── report.html             # HTML-отчёт (автоматически создаётся)
└── requirements.txt

````

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
````

### 2. Обучение модели

```bash
python src/train.py
```

### 3. Предсказания и отчёт

```bash
python src/inference.py
```

### 4. Запуск FastAPI-сервера

```bash
uvicorn src.app:app --reload
```

После запуска открой:

* 🔗 Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* 📄 HTML-отчёт: [http://127.0.0.1:8000/report](http://127.0.0.1:8000/report)

## 🧪 Тестирование

```bash
pytest
```

## ⚙️ CI

Проект содержит GitHub Actions workflow, который:

* устанавливает зависимости
* запускает тесты
* обучает модель
* делает предсказания

Учебный проект, 2025