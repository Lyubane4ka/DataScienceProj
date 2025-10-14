import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Настройка стиля графиков
plt.style.use('default')
rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")


def create_visualisation(df):
    print(f"Размер датасета: {df.shape}")
    print("\nПервые 5 записей:")
    print(df.head())

    # 1. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЙ ЧИСЛОВЫХ ПЕРЕМЕННЫХ
    print("\n" + "=" * 50)
    print("1. РАСПРЕДЕЛЕНИЯ ЧИСЛОВЫХ ПЕРЕМЕННЫХ")
    print("=" * 50)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Распределения числовых характеристик бриллиантов', fontsize=16)

    # Carat распределение
    axes[0, 0].hist(df['carat'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Распределение веса (карат)')
    axes[0, 0].set_xlabel('Караты')
    axes[0, 0].set_ylabel('Частота')

    # Price распределение
    axes[0, 1].hist(df['price'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Распределение цен')
    axes[0, 1].set_xlabel('Цена ($)')
    axes[0, 1].set_ylabel('Частота')

    # Depth распределение
    axes[0, 2].hist(df['depth'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title('Распределение глубины (%)')
    axes[0, 2].set_xlabel('Глубина (%)')
    axes[0, 2].set_ylabel('Частота')

    # Table распределение
    axes[1, 0].hist(df['table'], bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[1, 0].set_title('Распределение table (%)')
    axes[1, 0].set_xlabel('Table (%)')
    axes[1, 0].set_ylabel('Частота')

    # Размеры x, y, z
    axes[1, 1].hist(df['x'], bins=50, alpha=0.5, label='Длина (x)', color='red')
    axes[1, 1].hist(df['y'], bins=50, alpha=0.5, label='Ширина (y)', color='blue')
    axes[1, 1].hist(df['z'], bins=50, alpha=0.5, label='Высота (z)', color='green')
    axes[1, 1].set_title('Распределение размеров (мм)')
    axes[1, 1].set_xlabel('Размер (мм)')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].legend()

    # Boxplot цен по категориям огранки
    df_boxplot = df[df['price'] <= 10000]  # Фильтруем для лучшей визуализации
    sns.boxplot(data=df_boxplot, x='cut', y='price', ax=axes[1, 2])
    axes[1, 2].set_title('Распределение цен по качеству огранки')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # 2. КАТЕГОРИАЛЬНЫЕ ПЕРЕМЕННЫЕ
    print("\n" + "=" * 50)
    print("2. КАТЕГОРИАЛЬНЫЕ ПЕРЕМЕННЫЕ")
    print("=" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Качество огранки
    cut_counts = df['cut'].value_counts()
    axes[0, 0].pie(cut_counts.values, labels=cut_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Распределение качества огранки')

    # Цвет
    color_counts = df['color'].value_counts().sort_index()
    axes[0, 1].bar(color_counts.index, color_counts.values, color='lightblue', edgecolor='black')
    axes[0, 1].set_title('Распределение цвета')
    axes[0, 1].set_xlabel('Цвет (D - лучший)')
    axes[0, 1].set_ylabel('Количество')

    # Чистота
    clarity_counts = df['clarity'].value_counts()
    axes[1, 0].barh(list(clarity_counts.index), clarity_counts.values, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('Распределение чистоты')
    axes[1, 0].set_xlabel('Количество')

    # Средняя цена по качеству огранки
    avg_price_by_cut = df.groupby('cut')['price'].mean().sort_values()
    axes[1, 1].bar(avg_price_by_cut.index, avg_price_by_cut.values, color='salmon', edgecolor='black')
    axes[1, 1].set_title('Средняя цена по качеству огранки')
    axes[1, 1].set_xlabel('Качество огранки')
    axes[1, 1].set_ylabel('Средняя цена ($)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # 3. КОРРЕЛЯЦИИ И СВЯЗИ МЕЖДУ ПЕРЕМЕННЫМИ
    print("\n" + "=" * 50)
    print("3. КОРРЕЛЯЦИИ И СВЯЗИ")
    print("=" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Матрица корреляций
    numeric_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0, 0])
    axes[0, 0].set_title('Матрица корреляций')

    # Связь каратов и цены
    sns.scatterplot(data=df.sample(1000), x='carat', y='price', hue='cut', alpha=0.7, ax=axes[0, 1])
    axes[0, 1].set_title('Зависимость цены от веса (карат)')
    axes[0, 1].set_xlabel('Караты')
    axes[0, 1].set_ylabel('Цена ($)')

    # Цена по цвету и чистоте
    pivot_table = df.pivot_table(values='price', index='color', columns='clarity', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1, 0])
    axes[1, 0].set_title('Средняя цена по цвету и чистоте')

    # Размеры vs цена
    sample_df = df.sample(500)
    scatter = axes[1, 1].scatter(sample_df['x'], sample_df['y'], c=sample_df['price'],
                                 cmap='viridis', alpha=0.6, s=sample_df['carat'] * 50)
    axes[1, 1].set_xlabel('Длина (x)')
    axes[1, 1].set_ylabel('Ширина (y)')
    axes[1, 1].set_title('Размеры vs Цена (размер точки = караты)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Цена ($)')

    plt.tight_layout()
    plt.show()

    # 4. ДОПОЛНИТЕЛЬНЫЕ АНАЛИТИЧЕСКИЕ ГРАФИКИ
    print("\n" + "=" * 50)
    print("4. ДЕТАЛЬНЫЙ АНАЛИЗ")
    print("=" * 50)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Распределение каратов по качеству огранки
    sns.violinplot(data=df, x='cut', y='carat', ax=axes[0, 0])
    axes[0, 0].set_title('Распределение веса по качеству огранки')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Кумулятивное распределение цен
    df_sorted = df.sort_values('price')
    df_sorted['cumulative_prob'] = range(1, len(df_sorted) + 1)
    df_sorted['cumulative_prob'] = df_sorted['cumulative_prob'] / len(df_sorted)
    axes[0, 1].plot(df_sorted['price'], df_sorted['cumulative_prob'], linewidth=2)
    axes[0, 1].set_title('Кумулятивное распределение цен')
    axes[0, 1].set_xlabel('Цена ($)')
    axes[0, 1].set_ylabel('Процент бриллиантов (%)')
    axes[0, 1].grid(True, alpha=0.3)

    # Соотношение глубины и table
    sns.scatterplot(data=df.sample(1000), x='depth', y='table', hue='cut', alpha=0.6, ax=axes[1, 0])
    axes[1, 0].set_title('Соотношение глубины и table')
    axes[1, 0].set_xlabel('Глубина (%)')
    axes[1, 0].set_ylabel('Table (%)')

    # Топ 10 самых дорогих бриллиантов по каратам
    top_diamonds = df.nlargest(10, 'price')[['carat', 'price', 'cut', 'color', 'clarity']]
    table_data = []
    for i, row in top_diamonds.iterrows():
        table_data.append([row['carat'], f"${row['price']:,}", row['cut'], row['color'], row['clarity']])

    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=table_data,
                             colLabels=['Караты', 'Цена', 'Огранка', 'Цвет', 'Чистота'],
                             loc='center',
                             cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    axes[1, 1].set_title('Топ 10 самых дорогих бриллиантов')

    plt.tight_layout()
    plt.show()

    # Вывод статистики
    print("\n" + "=" * 50)
    print("ОСНОВНАЯ СТАТИСТИКА:")
    print("=" * 50)
    print(f"Всего записей: {len(df):,}")
    print(f"Средняя цена: ${df['price'].mean():.2f}")
    print(f"Средний вес: {df['carat'].mean():.2f} карат")
    print(f"Самый дорогой бриллиант: ${df['price'].max():,}")
    print(f"Самый большой бриллиант: {df['carat'].max():.2f} карат")
    print(f"\nСамое частое качество огранки: {df['cut'].mode().values[0]}")
    print(f"Самый частый цвет: {df['color'].mode().values[0]}")
