import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


def check_stationarity(timeseries):
    """
    Проверка стационарности временного ряда с помощью теста Дики-Фуллера
    """
    result = adfuller(timeseries.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

    if result[1] <= 0.05:
        print("Ряд стационарен (отвергаем H0)")
        return True
    else:
        print("Ряд нестационарен (не отвергаем H0)")
        return False


def find_best_sarima_params(timeseries, seasonal_period=52):
    """
    Поиск оптимальных параметров SARIMA с помощью грубой силы
    """
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None

    # Ограниченный набор параметров для демонстрации
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = SARIMAX(timeseries,
                                    order=(p, d, q),
                                    seasonal_order=(p, d, q, seasonal_period),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    results = model.fit(disp=False)

                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                        best_seasonal_order = (p, d, q, seasonal_period)

                except:
                    continue

    return best_order, best_seasonal_order, best_aic


def sarima_segmentation(timeseries, forecast_periods=12):
    """
    Сегментация временного ряда с помощью SARIMA
    """
    print("\n" + "=" * 50)
    print("СЕГМЕНТАЦИЯ ВРЕМЕННОГО РЯДА С ПОМОЩЬЮ SARIMA")
    print("=" * 50)

    # Визуализация исходного ряда
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(timeseries)
    plt.title('Исходный временной ряд (средняя цена бриллиантов)')
    plt.xlabel('Дата')
    plt.ylabel('Средняя цена')
    plt.xticks(rotation=45)

    # Декомпозиция временного ряда
    decomposition = seasonal_decompose(timeseries.dropna(), model='additive', period=52)

    plt.subplot(3, 2, 2)
    plt.plot(decomposition.trend)
    plt.title('Тренд')
    plt.xlabel('Дата')
    plt.ylabel('Тренд')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 3)
    plt.plot(decomposition.seasonal)
    plt.title('Сезонность')
    plt.xlabel('Дата')
    plt.ylabel('Сезонность')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 4)
    plt.plot(decomposition.resid)
    plt.title('Остатки')
    plt.xlabel('Дата')
    plt.ylabel('Остатки')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Проверка стационарности
    print("\nПроверка стационарности исходного ряда:")
    is_stationary = check_stationarity(timeseries)

    # Если ряд нестационарен, применяем дифференцирование
    if not is_stationary:
        print("\nПрименяем дифференцирование...")
        timeseries_diff = timeseries.diff().dropna()
        print("Проверка стационарности после дифференцирования:")
        check_stationarity(timeseries_diff)

    # Поиск оптимальных параметров SARIMA
    print("\nПоиск оптимальных параметров SARIMA...")
    best_order, best_seasonal_order, best_aic = find_best_sarima_params(timeseries)

    print(f"Лучшие параметры: SARIMA{best_order}x{best_seasonal_order}")
    print(f"Лучший AIC: {best_aic:.2f}")

    # Обучение финальной модели
    print("\nОбучение финальной модели SARIMA...")
    model = SARIMAX(timeseries,
                    order=best_order,
                    seasonal_order=best_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    fitted_model = model.fit(disp=False)
    print(fitted_model.summary())

    # Прогнозирование
    forecast = fitted_model.get_forecast(steps=forecast_periods)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Визуализация результатов
    plt.figure(figsize=(15, 8))

    # Исторические данные
    plt.plot(timeseries.index, timeseries, label='Исторические данные', color='blue')

    # Прогноз
    plt.plot(forecast_mean.index, forecast_mean, label='Прогноз', color='red')

    # Доверительный интервал
    plt.fill_between(forecast_ci.index,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1],
                     color='pink', alpha=0.3, label='95% Доверительный интервал')

    plt.title('Прогноз средней цены бриллиантов с помощью SARIMA')
    plt.xlabel('Дата')
    plt.ylabel('Средняя цена')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Анализ остатков
    residuals = fitted_model.resid

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(residuals)
    plt.title('Остатки модели')
    plt.xlabel('Дата')
    plt.ylabel('Остатки')

    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.title('Гистограмма остатков')
    plt.xlabel('Остатки')
    plt.ylabel('Частота')

    plt.subplot(2, 2, 3)
    from scipy import stats
    stats.probplot(residuals.dropna(), dist="norm", plot=plt)
    plt.title('Q-Q plot остатков')

    plt.subplot(2, 2, 4)
    pd.plotting.autocorrelation_plot(residuals.dropna())
    plt.title('Автокорреляция остатков')

    plt.tight_layout()
    plt.show()

    # Сегментация на основе прогноза
    print("\n" + "=" * 50)
    print("СЕГМЕНТАЦИЯ РЫНКА НА ОСНОВЕ ПРОГНОЗА")
    print("=" * 50)

    last_historical = timeseries.iloc[-1]
    forecast_values = forecast_mean.values

    # Определяем сегменты на основе динамики цен
    segments = []
    for i, value in enumerate(forecast_values):
        change_percent = ((value - last_historical) / last_historical) * 100

        if change_percent > 5:
            segment = "РАСТУЩИЙ"
        elif change_percent < -5:
            segment = "ПАДАЮЩИЙ"
        else:
            segment = "СТАБИЛЬНЫЙ"

        segments.append({
            'period': i + 1,
            'date': forecast_mean.index[i],
            'price': value,
            'change_percent': change_percent,
            'segment': segment
        })

    segments_df = pd.DataFrame(segments)
    print("Сегментация прогнозируемых периодов:")
    print(segments_df.to_string(index=False))

    return fitted_model, forecast_mean, segments_df


# Дополнительный анализ: сегментация по разным категориям бриллиантов
def segment_by_categories(df):
    """
    Сегментация по разным категориям бриллиантов
    """
    print("\n" + "=" * 50)
    print("СЕГМЕНТАЦИЯ ПО КАТЕГОРИЯМ БРИЛЛИАНТОВ")
    print("=" * 50)

    # Анализ по cut (качество огранки)
    if 'cut' in df.columns or any('cut_' in col for col in df.columns):
        cut_columns = [col for col in df.columns if 'cut_' in col]
        if cut_columns:
            cut_prices = {}
            for col in cut_columns:
                mask = df[col] == 1
                if mask.any():
                    cut_prices[col] = df.loc[mask, 'price'].mean()

            print("Средние цены по качеству огранки:")
            for cut, price in sorted(cut_prices.items(), key=lambda x: x[1]):
                print(f"  {cut}: ${price:.2f}")

    # Анализ по color (цвет)
    if 'color' in df.columns:
        color_prices = df.groupby('color')['price'].mean().sort_values()
        print("\nСредние цены по цвету:")
        for color, price in color_prices.items():
            print(f"  Цвет {color}: ${price:.2f}")

    # Анализ по carat (вес)
    df['carat_category'] = pd.cut(df['carat'], bins=[0, 0.5, 1, 1.5, 2, 10],
                                  labels=['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2+'])
    carat_prices = df.groupby('carat_category')['price'].mean()
    print("\nСредние цены по весу (каратам):")
    for category, price in carat_prices.items():
        print(f"  {category} карат: ${price:.2f}")


# Функции для работы с временными рядами
def create_time(df, target_column='price'):
    date_column = None
    # Если нет временной метки, создаем искусственную
    if date_column is None or date_column not in df.columns:
        # Создаем искусственный временной индекс (предполагаем, что данные собраны за 3 года)
        dates = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        df_with_dates = df.copy()
        df_with_dates['date'] = dates
    else:
        df_with_dates = df.copy()
        df_with_dates['date'] = pd.to_datetime(df_with_dates[date_column])

    # Группируем по дате и агрегируем целевую переменную
    time_series = df_with_dates.groupby('date')[target_column].mean()

    # Ресемплируем до недельной частоты для лучшей стабильности
    time_series_weekly = time_series.resample('W').mean()

    return time_series_weekly


def processing(df):
    print("\n" + "=" * 50)
    print("ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 50)

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

    numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    for col in numeric_columns:
        if col in df.columns:
            df = handle_outliers_iqr(df, col)

    categorical_cols = df.select_dtypes(include=['object']).columns
    print("Категориальные колонки:", categorical_cols.tolist())

    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} уникальных значений")

        if unique_count <= 10:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
        else:
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

    data.loc[:, column] = data[column].clip(lower=lower_bound, upper=upper_bound).astype('int64')
    return data


def optimize_types(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object':
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    return df
