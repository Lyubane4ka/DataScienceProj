import warnings
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from pmdarima import auto_arima
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from Hw6.data_load import *

warnings.filterwarnings('ignore')

def arima_start(df_processed):



    print("\n" + "=" * 50)
    print("АНАЛИЗ И СЕГМЕНТАЦИЯ ДАННЫХ")
    print("=" * 50)

    # 1. Анализ временных рядов для цены
    print("1. Анализ временных рядов для цены алмазов...")

    # Создаем искусственный временной ряд на основе индекса
    df_sorted = df_processed.sort_values('price').reset_index(drop=True)

    # Разделяем данные на обучающую и тестовую выборки
    train_size = int(0.8 * len(df_sorted))
    train_data = df_sorted['price'][:train_size]
    test_data = df_sorted['price'][train_size:]

    print(f"Размер обучающей выборки: {len(train_data)}")
    print(f"Размер тестовой выборки: {len(test_data)}")

    # Построение модели ARIMA с помощью auto_arima
    print("\n2. Подбор параметров ARIMA с помощью auto_arima...")

    # Переменные для хранения метрик
    arima_metrics = {}

    try:
        model = auto_arima(train_data,
                           seasonal=False,  # Не используем сезонность
                           stepwise=True,
                           suppress_warnings=True,
                           error_action='ignore',
                           trace=True)

        print(f"Лучшая модель: {model}")

        # Прогнозирование
        forecast = model.predict(n_periods=len(test_data))

        # Расчет метрик качества для ARIMA
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = sqrt(mse)
        mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
        r2 = r2_score(test_data, forecast)

        arima_metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }

        print("\nМетрики качества прогноза ARIMA:")
        print(f"MAE (Средняя абсолютная ошибка): {mae:.2f}")
        print(f"MSE (Средняя квадратичная ошибка): {mse:.2f}")
        print(f"RMSE (Среднеквадратичная ошибка): {rmse:.2f}")
        print(f"MAPE (Средняя абсолютная процентная ошибка): {mape:.2f}%")
        print(f"R² (Коэффициент детерминации): {r2:.4f}")

        # Визуализация результатов
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.plot(train_data.index, train_data.values, label='Обучающая выборка', color='blue')
        plt.plot(test_data.index, test_data.values, label='Тестовая выборка', color='green')
        plt.plot(test_data.index, forecast, label='Прогноз ARIMA', color='red', linestyle='--')
        plt.title('Прогнозирование цены алмазов с помощью ARIMA')
        plt.xlabel('Индекс')
        plt.ylabel('Цена')
        plt.legend()
        plt.grid(True)

    except Exception as e:
        print(f"Ошибка при построении ARIMA: {e}")
        # Используем простую модель в случае ошибки
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(train_data, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test_data))

    # 3. Кластеризация для сегментации алмазов
    print("\n3. Кластеризация алмазов...")

    # Выбираем признаки для кластеризации
    features_for_clustering = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    clustering_data = df_processed[features_for_clustering].copy()

    # Стандартизация данных
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Определение оптимального числа кластеров
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    k_range = range(2, 8)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)

        # Метрики качества кластеризации
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        calinski_avg = calinski_harabasz_score(scaled_data, cluster_labels)
        davies_avg = davies_bouldin_score(scaled_data, cluster_labels)

        silhouette_scores.append(silhouette_avg)
        calinski_scores.append(calinski_avg)
        davies_scores.append(davies_avg)

    # Выбор оптимального числа кластеров
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    optimal_k_calinski = k_range[np.argmax(calinski_scores)]
    optimal_k_davies = k_range[np.argmin(davies_scores)]

    print(f"\nОптимальное число кластеров по разным метрикам:")
    print(f"По силуэтному коэффициенту: {optimal_k_silhouette}")
    print(f"По индексу Calinski-Harabasz: {optimal_k_calinski}")
    print(f"По индексу Davies-Bouldin: {optimal_k_davies}")

    # Используем силуэтный коэффициент как основной критерий
    optimal_k = optimal_k_silhouette
    print(f"\nВыбранное оптимальное число кластеров: {optimal_k}")

    # Кластеризация с оптимальным K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    df_processed['cluster'] = cluster_labels

    # Метрики качества для финальной кластеризации
    final_silhouette = silhouette_score(scaled_data, cluster_labels)
    final_calinski = calinski_harabasz_score(scaled_data, cluster_labels)
    final_davies = davies_bouldin_score(scaled_data, cluster_labels)

    print(f"\nМетрики качества кластеризации (K={optimal_k}):")
    print(f"Силуэтный коэффициент: {final_silhouette:.4f}")
    print(f"Индекс Calinski-Harabasz: {final_calinski:.2f}")
    print(f"Индекс Davies-Bouldin: {final_davies:.4f}")

    # Интерпретация силуэтного коэффициента
    if final_silhouette > 0.7:
        silhouette_interpretation = "Сильная структура"
    elif final_silhouette > 0.5:
        silhouette_interpretation = "Разумная структура"
    elif final_silhouette > 0.25:
        silhouette_interpretation = "Слабая структура"
    else:
        silhouette_interpretation = "Нет существенной структуры"

    print(f"Интерпретация силуэтного коэффициента: {silhouette_interpretation}")

    # 4. Анализ кластеров
    print("\n4. Анализ кластеров:")

    # Статистика по кластерам
    cluster_stats = df_processed.groupby('cluster')[features_for_clustering].mean()
    print("\nСредние значения по кластерам:")
    print(cluster_stats)

    # Размеры кластеров
    cluster_sizes = df_processed['cluster'].value_counts().sort_index()
    print("\nРазмеры кластеров:")
    print(cluster_sizes)

    # Внутрикластерная дисперсия
    within_cluster_variance = kmeans.inertia_
    print(f"\nВнутрикластерная дисперсия (inertia): {within_cluster_variance:.2f}")

    # 5. Визуализация результатов
    plt.subplot(2, 2, 2)
    plt.bar(cluster_sizes.index, cluster_sizes.values)
    plt.title('Размеры кластеров')
    plt.xlabel('Кластер')
    plt.ylabel('Количество алмазов')
    plt.grid(True, alpha=0.3)

    # Визуализация кластеров в 2D (PCA)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    plt.subplot(2, 2, 3)
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                          c=df_processed['cluster'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('Визуализация кластеров (PCA)')
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')

    # Визуализация метрик кластеризации
    plt.subplot(2, 2, 4)
    plt.plot(k_range, silhouette_scores, 'bo-', label='Силуэтный коэффициент')
    plt.plot(k_range, np.array(calinski_scores) / max(calinski_scores), 'ro-', label='Calinski-Harabasz (норм.)')
    plt.plot(k_range, 1 - np.array(davies_scores) / max(davies_scores), 'go-', label='Davies-Bouldin (инверс.)')
    plt.xlabel('Число кластеров')
    plt.ylabel('Нормализованные метрики')
    plt.title('Метрики качества кластеризации')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 6. Детальный анализ характеристик кластеров
    print("\n" + "=" * 50)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ КЛАСТЕРОВ")
    print("=" * 50)

    # Создаем описание кластеров на основе их характеристик
    for cluster_id in range(optimal_k):
        cluster_data = df_processed[df_processed['cluster'] == cluster_id]

        print(f"\n--- Кластер {cluster_id} (n={len(cluster_data)}) ---")
        print(f"Средняя цена: ${cluster_data['price'].mean():.2f}")
        print(f"Средний карат: {cluster_data['carat'].mean():.3f}")
        print(
            f"Средние размеры: x={cluster_data['x'].mean():.2f}, y={cluster_data['y'].mean():.2f}, z={cluster_data['z'].mean():.2f}")
        print(f"Диапазон цен: ${cluster_data['price'].min():.2f} - ${cluster_data['price'].max():.2f}")

    # 7. Прогнозирование временных рядов для каждого кластера
    print("\n" + "=" * 50)
    print("ПРОГНОЗИРОВАНИЕ ДЛЯ КАЖДОГО КЛАСТЕРА")
    print("=" * 50)

    plt.figure(figsize=(15, 10))
    cluster_metrics = {}

    for cluster_id in range(optimal_k):
        cluster_prices = df_processed[df_processed['cluster'] == cluster_id]['price'].sort_values().values

        if len(cluster_prices) > 100:  # Только для кластеров с достаточным количеством данных
            # Создаем временной ряд для кластера
            time_series = pd.Series(cluster_prices)

            # Разделяем на train/test
            cluster_train_size = int(0.8 * len(time_series))
            cluster_train = time_series[:cluster_train_size]
            cluster_test = time_series[cluster_train_size:]

            try:
                # Строим модель ARIMA для кластера
                cluster_model = auto_arima(cluster_train,
                                           seasonal=False,
                                           stepwise=True,
                                           suppress_warnings=True,
                                           error_action='ignore')

                cluster_forecast = cluster_model.predict(n_periods=len(cluster_test))

                # Расчет метрик для кластера
                cluster_mae = mean_absolute_error(cluster_test, cluster_forecast)
                cluster_rmse = sqrt(mean_squared_error(cluster_test, cluster_forecast))
                cluster_mape = np.mean(np.abs((cluster_test - cluster_forecast) / cluster_test)) * 100

                cluster_metrics[cluster_id] = {
                    'MAE': cluster_mae,
                    'RMSE': cluster_rmse,
                    'MAPE': cluster_mape,
                    'Model': f"ARIMA{cluster_model.order}"
                }

                # Визуализация прогноза для кластера
                plt.subplot(2, 3, cluster_id + 1)
                plt.plot(range(len(cluster_train)), cluster_train.values, label='Train', color='blue')
                plt.plot(range(len(cluster_train), len(cluster_train) + len(cluster_test)),
                         cluster_test.values, label='Test', color='green')
                plt.plot(range(len(cluster_train), len(cluster_train) + len(cluster_test)),
                         cluster_forecast, label='Forecast', color='red', linestyle='--')
                plt.title(f'Кластер {cluster_id} - Прогноз цены\nMAPE: {cluster_mape:.1f}%')
                plt.xlabel('Время')
                plt.ylabel('Цена')
                plt.legend()
                plt.grid(True)

                print(f"\nКластер {cluster_id}: {cluster_model}")
                print(f"  MAE: ${cluster_mae:.2f}, RMSE: ${cluster_rmse:.2f}, MAPE: {cluster_mape:.1f}%")

            except Exception as e:
                print(f"Ошибка для кластера {cluster_id}: {e}")

    plt.tight_layout()
    plt.show()

    # 8. Сводная таблица метрик
    print("\n" + "=" * 50)
    print("СВОДНАЯ ТАБЛИЦА МЕТРИК КАЧЕСТВА")
    print("=" * 50)

    print("\nМЕТРИКИ КЛАСТЕРИЗАЦИИ:")
    print(f"{'Метрика':<25} {'Значение':<15} {'Интерпретация':<30}")
    print("-" * 70)
    print(f"{'Силуэтный коэффициент':<25} {final_silhouette:<15.4f} {silhouette_interpretation:<30}")
    print(f"{'Calinski-Harabasz':<25} {final_calinski:<15.2f} {'Чем выше, тем лучше':<30}")
    print(f"{'Davies-Bouldin':<25} {final_davies:<15.4f} {'Чем ниже, тем лучше':<30}")
    print(f"{'Within-cluster SSE':<25} {within_cluster_variance:<15.2f} {'Чем ниже, тем лучше':<30}")

    if arima_metrics:
        print("\nМЕТРИКИ ПРОГНОЗИРОВАНИЯ (Общая модель ARIMA):")
        print(f"{'Метрика':<15} {'Значение':<15}")
        print("-" * 30)
        for metric, value in arima_metrics.items():
            if metric == 'MAPE':
                print(f"{metric:<15} {value:<15.1f}%")
            elif metric == 'R2':
                print(f"{metric:<15} {value:<15.4f}")
            else:
                print(f"{metric:<15} {value:<15.2f}")

    if cluster_metrics:
        print("\nМЕТРИКИ ПРОГНОЗИРОВАНИЯ ПО КЛАСТЕРАМ:")
        print(f"{'Кластер':<10} {'Модель':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 55)
        for cluster_id, metrics in cluster_metrics.items():
            print(
                f"{cluster_id:<10} {metrics['Model']:<15} ${metrics['MAE']:<9.2f} ${metrics['RMSE']:<9.2f} {metrics['MAPE']:<9.1f}%")
