import pandas as pd
import numpy as np
from pmdarima import auto_arima


def processing(df):
    print("\n" + "=" * 50)
    print("ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 50)
    # Проверка пропущенных значений
    missing_values = df.isnull().sum()
    print("Пропущенные значения:")
    print(missing_values[missing_values > 0])

    # Если есть пропущенные значения, заполним их
    if df.isnull().sum().sum() > 0:
        # Для числовых колонок - медианой
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Для категориальных - модой
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')


    print(f"Дубликатов до очистки: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"Дубликатов после очистки: {df.duplicated().sum()}")

    # Обработка выбросов для числовых колонок
    numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    for col in numeric_columns:
        if col in df.columns:
            df = handle_outliers_iqr(df, col)

    # Проверяем категориальные колонки
    categorical_cols = df.select_dtypes(include=['object']).columns
    print("Категориальные колонки:", categorical_cols.tolist())

    # One-Hot Encoding для категориальных переменных с небольшим количеством уникальных значений
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} уникальных значений")

        if unique_count <= 10:  # One-Hot Encoding для колонок с малым количеством категорий
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
        else:  # Label Encoding для колонок с большим количеством категорий
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    df = optimize_types(df)
    print("Типы данных после оптимизации:")
    print(df.dtypes)

    return df


def handle_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Ограничение выбросов и преобразование типа
    data.loc[:, column] = data[column].clip(lower=lower_bound, upper=upper_bound).astype('int64')

    return data

# Оптимизация типов данных для экономии памяти
def optimize_types(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object':
            if df[col].nunique() / len(df) < 0.5:  # Если мало уникальных значений
                df[col] = df[col].astype('category')
    return df


def processing_for_arima(df, target_column='price'):
    print("\n" + "=" * 50)
    print("ПРЕДОБРАБОТКА ДАННЫХ ДЛЯ ARIMA")
    print("=" * 50)

    # Ваша существующая логика обработки
    missing_values = df.isnull().sum()
    print("Пропущенные значения:")
    print(missing_values[missing_values > 0])

    if df.isnull().sum().sum() > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    print(f"Дубликатов до очистки: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print(f"Дубликатов после очистки: {df.duplicated().sum()}")

    # Обработка выбросов
    numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    for col in numeric_columns:
        if col in df.columns:
            df = handle_outliers_iqr(df, col)

    # Кодирование категориальных переменных
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= 10:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Создание искусственного временного индекса
    df = df.sort_values(by=target_column).reset_index(drop=True)
    df['time_index'] = range(1, len(df) + 1)

    df = optimize_types(df)
    print("Типы данных после оптимизации:")
    print(df.dtypes)

    return df
