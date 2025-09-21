import sqlite3
from typing import List, Dict, Any


class TitanicDatabase:
    """Класс для работы с базой данных пассажиров Титаника"""

    def __init__(self, db_path: str = 'titanic.db'):
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

    def get_survivors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить выживших пассажиров"""
        query = "SELECT * FROM passengers WHERE Survived = 1 LIMIT ?"
        return self.execute_query(query, (limit,))

    def get_first_class_passengers(self) -> List[Dict[str, Any]]:
        """Получить пассажиров первого класса"""
        query = "SELECT Name, Sex, Age, Fare FROM passengers WHERE Pclass = 1 ORDER BY Fare DESC"
        return self.execute_query(query)

    def get_children(self, max_age: int = 12) -> List[Dict[str, Any]]:
        """Получить детей до указанного возраста"""
        query = "SELECT Name, Age, Sex, Survived FROM passengers WHERE Age < ? ORDER BY Age"
        return self.execute_query(query, (max_age,))

    def get_survival_rate_by_class(self) -> List[Dict[str, Any]]:
        """Получить процент выживаемости по классам"""
        query = """
        SELECT 
            Pclass,
            COUNT(*) as Total,
            SUM(Survived) as Survived,
            ROUND(SUM(Survived) * 100.0 / COUNT(*), 2) as SurvivalRate
        FROM passengers 
        GROUP BY Pclass 
        ORDER BY Pclass
        """
        return self.execute_query(query)

    def get_survival_rate_by_gender_class(self) -> List[Dict[str, Any]]:
        """Получить выживаемость по полу и классу"""
        query = """
        SELECT 
            Sex,
            Pclass,
            COUNT(*) as Total,
            SUM(Survived) as Survived,
            ROUND(SUM(Survived) * 100.0 / COUNT(*), 2) as SurvivalRate
        FROM passengers 
        GROUP BY Sex, Pclass 
        ORDER BY Sex, Pclass
        """
        return self.execute_query(query)

    def get_age_statistics_by_survival(self) -> List[Dict[str, Any]]:
        """Получить статистику возраста по выживаемости"""
        query = """
        SELECT 
            Survived,
            COUNT(*) as Count,
            ROUND(AVG(Age), 2) as AverageAge,
            MIN(Age) as MinAge,
            MAX(Age) as MaxAge
        FROM passengers 
        WHERE Age IS NOT NULL
        GROUP BY Survived
        """
        return self.execute_query(query)

    def get_survived_women_children_first_class(self) -> List[Dict[str, Any]]:
        """Получить выживших женщин и детей из первого класса"""
        query = """
        SELECT Name, Age, Sex, Pclass, Fare 
        FROM passengers 
        WHERE Survived = 1 
          AND (Sex = 'female' OR Age < 16)
          AND Pclass = 1
        ORDER BY Age
        """
        return self.execute_query(query)

    def get_high_fare_passengers(self, min_fare: float = 100.0) -> List[Dict[str, Any]]:
        """Получить пассажиров с высокой стоимостью билета"""
        query = "SELECT Name, Pclass, Fare, Survived FROM passengers WHERE Fare > ? ORDER BY Fare DESC"
        return self.execute_query(query, (min_fare,))

    def get_passengers_without_age(self) -> List[Dict[str, Any]]:
        """Получить пассажиров без указанного возраста"""
        query = "SELECT Name, Sex, Pclass, Survived FROM passengers WHERE Age IS NULL"
        return self.execute_query(query)

    def get_families(self) -> List[Dict[str, Any]]:
        """Получить пассажиров с родственниками"""
        query = """
        SELECT Name, Age, Sex, SibSp, Parch, Survived 
        FROM passengers 
        WHERE SibSp > 0 OR Parch > 0 
        ORDER BY SibSp + Parch DESC
        """
        return self.execute_query(query)

    def get_embarkation_stats(self) -> List[Dict[str, Any]]:
        """Получить статистику по портам посадки"""
        query = """
        SELECT 
            Embarked,
            COUNT(*) as Passengers,
            SUM(Survived) as Survived,
            ROUND(AVG(Fare), 2) as AvgFare
        FROM passengers 
        WHERE Embarked IS NOT NULL
        GROUP BY Embarked
        """
        return self.execute_query(query)

    def get_top_expensive_tickets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить топ самых дорогих билетов"""
        query = "SELECT Name, Pclass, Fare, Survived FROM passengers ORDER BY Fare DESC LIMIT ?"
        return self.execute_query(query, (limit,))

    def get_age_group_analysis(self) -> List[Dict[str, Any]]:
        """Получить анализ по возрастным группам"""
        query = """
        SELECT 
            CASE 
                WHEN Age < 18 THEN 'Child'
                WHEN Age BETWEEN 18 AND 35 THEN 'Young Adult'
                WHEN Age BETWEEN 36 AND 55 THEN 'Adult'
                WHEN Age > 55 THEN 'Senior'
                ELSE 'Unknown'
            END as AgeGroup,
            COUNT(*) as Total,
            SUM(Survived) as Survived,
            ROUND(SUM(Survived) * 100.0 / COUNT(*), 2) as SurvivalRate
        FROM passengers 
        GROUP BY AgeGroup 
        ORDER BY SurvivalRate DESC
        """
        return self.execute_query(query)

    def get_passenger_count(self) -> int:
        """Получить общее количество пассажиров"""
        query = "SELECT COUNT(*) as count FROM passengers"
        result = self.execute_query(query)
        return result[0]['count'] if result else 0
