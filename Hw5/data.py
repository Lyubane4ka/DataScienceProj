import pandas as pd
import sqlite3

# Загрузка данных из SQLite
def load_data_from_db(db_path='bmws.db'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql('SELECT * FROM bmw', conn)
    conn.close()
    return df