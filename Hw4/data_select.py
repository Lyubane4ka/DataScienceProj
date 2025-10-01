import sqlite3
from typing import List, Dict, Any


class TweetsDatabase:
    """Класс для работы с базой данных пассажиров Титаника"""

    def __init__(self, db_path: str = 'tweets_db.db'):
        self.db_path = db_path
        self.connection = None

    def connect(self) -> None:
        """Установка соединения с базой данных"""
        self.connection = sqlite3.connect(self.db_path)

    def disconnect(self) -> None:
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Выполнение SQL-запроса и возврат результатов"""
        if not self.connection:
            self.connect()

        cursor = self.connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        # Получаем названия колонок
        columns = [description[0] for description in cursor.description]

        # Преобразуем результаты в список словарей
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))

        return results

    def get_number_of_tweets_by_target(self) -> List[Dict[str, Any]]:
        """Количество твитов по категориям настроений"""
        query = "SELECT target, COUNT(*) as count FROM tweets GROUP BY target;"
        return self.execute_query(query)

    def get_first_hundred_tweets(self) -> List[Dict[str, Any]]:
        """Случайная выборка из 100 твитов"""
        query = "SELECT * FROM tweets ORDER BY RANDOM() LIMIT 100;"
        return self.execute_query(query)

    def get_user_with_negative_tweets(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Пользователи с наибольшим количеством негативных твитов"""
        query = ("SELECT user, COUNT(*) as negative_count FROM tweets WHERE target = 0 "
                 "GROUP BY user ORDER BY negative_count DESC LIMIT ?;")
        return self.execute_query(query, (limit,))
