import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# Подготовка данных для LSTM
def prepare_lstm_data(df, sequence_length=5):
    """Подготовка данных в формате временных рядов для LSTM"""
    features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Стандартизация
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    # Создание последовательностей
    sequences = []
    for i in range(len(scaled_data) - sequence_length + 1):
        sequences.append(scaled_data[i:i + sequence_length])

    return np.array(sequences), scaler, features


# Автоэнкодер на LSTM для сегментации
def create_lstm_autoencoder(input_shape, encoding_dim=32):
    """Создание LSTM автоэнкодера для обучения представлений"""
    # Энкодер
    inputs = Input(shape=input_shape)
    encoded = LSTM(64, activation='relu', return_sequences=True)(inputs)
    encoded = LSTM(32, activation='relu', return_sequences=False)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    # Декодер
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(decoded)
    decoded = Dense(input_shape[1], activation='linear')(decoded)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    autoencoder.compile(optimizer=Adam(learning_rate=0.001),
                        loss='mse')

    return autoencoder, encoder


# Кластеризация на основе LSTM представлений
def lstm_clustering(df, n_clusters=4, sequence_length=5):
    """Сегментация с использованием LSTM автоэнкодера"""

    # Подготовка данных
    sequences, scaler, features = prepare_lstm_data(df, sequence_length)

    if len(sequences) == 0:
        print("Недостаточно данных для создания последовательностей")
        return None, None, None, None

    print(f"Размер последовательностей: {sequences.shape}")

    # Создание и обучение автоэнкодера
    input_shape = (sequence_length, sequences.shape[2])
    autoencoder, encoder = create_lstm_autoencoder(input_shape)

    print("Обучение LSTM автоэнкодера...")
    history = autoencoder.fit(sequences, sequences,
                              epochs=50,
                              batch_size=128,
                              validation_split=0.2,
                              verbose=0)

    # Извлечение признаков
    encoded_features = encoder.predict(sequences, verbose=0)

    # K-means кластеризация на извлеченных признаках
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(encoded_features)

    return clusters, encoded_features, history, sequences


# Альтернативный подход: прямая кластеризация
def traditional_clustering(df, n_clusters=4):
    """Традиционная кластеризация для сравнения"""
    features = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    return clusters, scaled_data


# Визуализация результатов для LSTM
def plot_lstm_clustering_results(df, clusters, sequences, method_name):
    """Визуализация результатов LSTM кластеризации с учетом размера последовательностей"""
    # Берем только те строки, которые были использованы в LSTM
    df_lstm = df.iloc[:len(clusters)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Распределение по кластерам
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    axes[0, 0].bar(cluster_counts.index, cluster_counts.values)
    axes[0, 0].set_title(f'{method_name}: Распределение по кластерам')
    axes[0, 0].set_xlabel('Кластер')
    axes[0, 0].set_ylabel('Количество образцов')

    # Цена vs карат по кластерам
    scatter = axes[0, 1].scatter(df_lstm['carat'], df_lstm['price'], c=clusters,
                                 cmap='viridis', alpha=0.6)
    axes[0, 1].set_title(f'{method_name}: Цена vs Карат')
    axes[0, 1].set_xlabel('Карат')
    axes[0, 1].set_ylabel('Цена')
    plt.colorbar(scatter, ax=axes[0, 1])

    # Размеры по кластерам
    if all(col in df_lstm.columns for col in ['x', 'y', 'z']):
        scatter = axes[1, 0].scatter(df_lstm['x'], df_lstm['y'], c=clusters,
                                     cmap='viridis', alpha=0.6)
        axes[1, 0].set_title(f'{method_name}: Размеры X vs Y')
        axes[1, 0].set_xlabel('X (длина)')
        axes[1, 0].set_ylabel('Y (ширина)')
        plt.colorbar(scatter, ax=axes[1, 0])

    # Качество огранки по кластерам (если есть)
    cut_columns = [col for col in df_lstm.columns if 'cut' in col.lower()]
    if cut_columns:
        cut_data = df_lstm[cut_columns].idxmax(axis=1)
        cluster_cut = pd.crosstab(clusters, cut_data)
        cluster_cut.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title(f'{method_name}: Качество огранки по кластерам')
        axes[1, 1].set_xlabel('Кластер')
        axes[1, 1].set_ylabel('Количество')
        axes[1, 1].legend(title='Качество огранки')
    else:
        # Альтернативная визуализация, если нет cut колонок
        if 'price' in df_lstm.columns:
            box_data = []
            for cluster_id in np.unique(clusters):
                box_data.append(df_lstm[clusters == cluster_id]['price'])

            axes[1, 1].boxplot(box_data, labels=np.unique(clusters))
            axes[1, 1].set_title(f'{method_name}: Распределение цен по кластерам')
            axes[1, 1].set_xlabel('Кластер')
            axes[1, 1].set_ylabel('Цена')

    plt.tight_layout()
    plt.show()

    return df_lstm


# Визуализация для традиционной кластеризации
def plot_traditional_clustering_results(df, clusters, method_name):
    """Визуализация результатов традиционной кластеризации"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Распределение по кластерам
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    axes[0, 0].bar(cluster_counts.index, cluster_counts.values)
    axes[0, 0].set_title(f'{method_name}: Распределение по кластерам')
    axes[0, 0].set_xlabel('Кластер')
    axes[0, 0].set_ylabel('Количество образцов')

    # Цена vs карат по кластерам
    scatter = axes[0, 1].scatter(df['carat'], df['price'], c=clusters,
                                 cmap='viridis', alpha=0.6)
    axes[0, 1].set_title(f'{method_name}: Цена vs Карат')
    axes[0, 1].set_xlabel('Карат')
    axes[0, 1].set_ylabel('Цена')
    plt.colorbar(scatter, ax=axes[0, 1])

    # Размеры по кластерам
    if all(col in df.columns for col in ['x', 'y', 'z']):
        scatter = axes[1, 0].scatter(df['x'], df['y'], c=clusters,
                                     cmap='viridis', alpha=0.6)
        axes[1, 0].set_title(f'{method_name}: Размеры X vs Y')
        axes[1, 0].set_xlabel('X (длина)')
        axes[1, 0].set_ylabel('Y (ширина)')
        plt.colorbar(scatter, ax=axes[1, 0])

    # Качество огранки по кластерам (если есть)
    cut_columns = [col for col in df.columns if 'cut' in col.lower()]
    if cut_columns:
        cut_data = df[cut_columns].idxmax(axis=1)
        cluster_cut = pd.crosstab(clusters, cut_data)
        cluster_cut.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title(f'{method_name}: Качество огранки по кластерам')
        axes[1, 1].set_xlabel('Кластер')
        axes[1, 1].set_ylabel('Количество')
        axes[1, 1].legend(title='Качество огранки')

    plt.tight_layout()
    plt.show()


# Анализ характеристик кластеров
def analyze_clusters(df, clusters, method_name):
    """Анализ характеристик каждого кластера"""
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters

    print(f"\n{method_name} - АНАЛИЗ КЛАСТЕРОВ")
    print("=" * 50)

    numeric_cols = df_clustered.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop('cluster') if 'cluster' in numeric_cols else numeric_cols

    cluster_stats = df_clustered.groupby('cluster')[numeric_cols].mean()
    print("Средние значения по кластерам:")
    print(cluster_stats)

    # Дополнительная статистика
    print(f"\nРазмеры кластеров:")
    for cluster_id in np.unique(clusters):
        cluster_size = (clusters == cluster_id).sum()
        print(f"Кластер {cluster_id}: {cluster_size} образцов ({cluster_size / len(clusters) * 100:.1f}%)")

    return df_clustered
