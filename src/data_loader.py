import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str):
    """Загружает данные из CSV и возвращает DataFrame."""
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    """Обработка датасета болезней сердца."""
    df = df.copy()

    # Заполняем пропуски, если есть
    df.fillna(df.median(numeric_only=True), inplace=True)

    # One-hot encoding для категориальных переменных
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Разделение признаков и цели
    if 'condition' in df.columns:
        y = df['condition']
        X = df.drop(columns=['condition'])
    else:
        X = df
        y = None

    return X, y


def load_and_preprocess(path: str):
    df = load_data(path)
    return preprocess_data(df)


def load_sample_data(path: str, test_size=0.2, random_state=42):
    """Загружает и делит данные на обучающую и тестовую выборки."""
    X, y = load_and_preprocess(path)

    # Сохраняем имена признаков
    feature_names = X.columns

    # Масштабируем признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_names
