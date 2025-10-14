import pandas as pd
import sqlite3


def create_table_in_db():
    conn = sqlite3.connect('diamondInfo.db')
    cursor = conn.cursor()

    # Создаем таблицу
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS diamonds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            carat REAL,
            cut TEXT,
            color TEXT,
            clarity TEXT,
            depth REAL,
            "table" REAL,
            price INTEGER,
            x REAL,
            y REAL,
            z REAL
        )
    ''')
    df = pd.read_csv('../Hw6/diamonds.csv')  # Уточните имя файла
    df.to_sql('diamonds', conn, if_exists='replace', index=False)

    # Проверка первых 5 записей
    query = "SELECT * FROM diamonds LIMIT 10;"
    print(pd.read_sql_query(query, conn))
    print('\nDataFrame Info:')
    print(df.info())

    # Quick summary statistics of numeric columns
    print('\nSummary statistics:')
    print(df.describe())
    # Сохраните изменения и закройте соединение
    conn.commit()
    conn.close()

# Загрузка данных из SQLite
def load_data_from_db():
    conn = sqlite3.connect('diamondInfo.db')
    df = pd.read_sql('SELECT * FROM diamonds', conn)
    conn.close()
    return df


def describe_diamonds():
    # Подключение к базе данных
    conn = sqlite3.connect('diamondInfo.db')

    # Проверка структуры таблицы
    query_structure = "PRAGMA table_info(diamonds);"
    structure = pd.read_sql_query(query_structure, conn)
    print("Структура таблицы diamonds:")
    print(structure)

    # Проверка первых 10 записей
    query_data = "SELECT * FROM diamonds LIMIT 10;"
    data_sample = pd.read_sql_query(query_data, conn)
    print("\nПервые 10 записей:")
    print(data_sample)

    # Статистика по числовым колонкам
    query_stats = """
    SELECT 
        COUNT(*) as total_records,
        AVG(carat) as avg_carat,
        MIN(carat) as min_carat,
        MAX(carat) as max_carat,
        AVG(price) as avg_price,
        MIN(price) as min_price,
        MAX(price) as max_price
    FROM diamonds;
    """
    stats = pd.read_sql_query(query_stats, conn)
    print("\nБазовая статистика:")
    print(stats)

    # Уникальные значения категориальных колонок
    print("\nУникальные значения cut:")
    print(pd.read_sql_query("SELECT DISTINCT cut FROM diamonds;", conn))

    print("\nУникальные значения color:")
    print(pd.read_sql_query("SELECT DISTINCT color FROM diamonds;", conn))

    print("\nУникальные значения clarity:")
    print(pd.read_sql_query("SELECT DISTINCT clarity FROM diamonds;", conn))

    conn.close()
