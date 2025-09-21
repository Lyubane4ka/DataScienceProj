import sqlite3
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("pastel")


class TitanicVisualizer:
    """Класс для визуализации данных Titanic"""

    def __init__(self, db_path: str = 'titanic.db'):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)

    def __del__(self):
        """Закрытие соединения при удалении объекта"""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()

    def execute_query(self, query: str) -> pd.DataFrame:
        """Выполнение SQL-запроса и возврат DataFrame"""
        return pd.read_sql_query(query, self.connection)

    def plot_survival_by_class(self, figsize: tuple = (10, 6)):
        """Визуализация выживаемости по классам"""
        query = """
        SELECT 
            Pclass,
            Survived,
            COUNT(*) as Count
        FROM passengers 
        GROUP BY Pclass, Survived
        """

        df = self.execute_query(query)
        pivot_df = df.pivot(index='Pclass', columns='Survived', values='Count')
        pivot_df.columns = ['Погиб', 'Выжил']

        fig, ax = plt.subplots(figsize=figsize)
        pivot_df.plot(kind='bar', ax=ax, color=['#ff6b6b', '#51cf66'])

        ax.set_title('Выживаемость пассажиров по классам', fontsize=16, fontweight='bold')
        ax.set_xlabel('Класс', fontsize=12)
        ax.set_ylabel('Количество пассажиров', fontsize=12)
        ax.legend(title='Статус')
        ax.grid(axis='y', alpha=0.3)

        # Добавляем проценты на столбцы
        for i, (idx, row) in enumerate(pivot_df.iterrows()):
            total = row.sum()
            for j, value in enumerate(row):
                percentage = (value / total) * 100
                ax.text(i + j * 0.3 - 0.15, value + 5, f'{percentage:.1f}%',
                        ha='center', fontweight='bold')

        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_survival_rate_by_gender_class(self, figsize: tuple = (10, 6)):
        """Визуализация процента выживаемости по полу и классу"""
        query = """
        SELECT 
            Sex,
            Pclass,
            ROUND(SUM(Survived) * 100.0 / COUNT(*), 2) as SurvivalRate
        FROM passengers 
        GROUP BY Sex, Pclass 
        ORDER BY Sex, Pclass
        """

        df = self.execute_query(query)

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(df['Pclass'].unique()))
        width = 0.35

        men = df[df['Sex'] == 'male']
        women = df[df['Sex'] == 'female']

        bars1 = ax.bar(x - width / 2, men['SurvivalRate'], width, label='Мужчины', alpha=0.8)
        bars2 = ax.bar(x + width / 2, women['SurvivalRate'], width, label='Женщины', alpha=0.8)

        ax.set_title('Процент выживаемости по полу и классу', fontsize=16, fontweight='bold')
        ax.set_xlabel('Класс', fontsize=12)
        ax.set_ylabel('Процент выживших (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(['1 класс', '2 класс', '3 класс'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Добавляем значения на столбцы
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def plot_age_distribution(self, figsize: tuple = (12, 6)):
        """Распределение возраста выживших и погибших"""
        query = """
        SELECT Age, Survived 
        FROM passengers 
        WHERE Age IS NOT NULL
        """

        df = self.execute_query(query)
        df['Survived'] = df['Survived'].map({0: 'Погиб', 1: 'Выжил'})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Гистограмма
        for survival_status in ['Выжил', 'Погиб']:
            data = df[df['Survived'] == survival_status]['Age']
            ax1.hist(data, bins=30, alpha=0.7, label=survival_status, density=True)

        ax1.set_title('Распределение возраста', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Возраст', fontsize=12)
        ax1.set_ylabel('Плотность', fontsize=12)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Boxplot
        sns.boxplot(data=df, x='Survived', y='Age', ax=ax2)
        ax2.set_title('Возрастное распределение по статусу', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Статус', fontsize=12)
        ax2.set_ylabel('Возраст', fontsize=12)

        plt.tight_layout()
        plt.show()

    def plot_fare_distribution(self, figsize: tuple = (12, 5)):
        """Распределение стоимости билетов"""
        query = "SELECT Fare, Pclass FROM passengers WHERE Fare > 0"

        df = self.execute_query(query)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Гистограмма
        ax1.hist(df['Fare'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Распределение стоимости билетов', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Стоимость билета ($)', fontsize=12)
        ax1.set_ylabel('Частота', fontsize=12)
        ax1.grid(alpha=0.3)

        # Boxplot по классам
        sns.boxplot(data=df, x='Pclass', y='Fare', ax=ax2)
        ax2.set_title('Стоимость билетов по классам', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Класс', fontsize=12)
        ax2.set_ylabel('Стоимость билета ($)', fontsize=12)
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, figsize: tuple = (10, 8)):
        """Тепловая карта корреляций"""
        query = "SELECT Survived, Pclass, Age, Fare FROM passengers"

        df = self.execute_query(query)
        correlation_matrix = df.corr()

        fig, ax = plt.subplots(figsize=figsize)

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, mask=mask, ax=ax, fmt='.2f')

        ax.set_title('Матрица корреляций признаков', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def create_dashboard(self):
        """Создание полной dashboard с всеми графиками"""
        print("🚢 Анализ данных Titanic - Dashboard\n")
        print("=" * 50)

        # Общая статистика
        query_total = "SELECT COUNT(*) as total FROM passengers"
        total_passengers = self.execute_query(query_total).iloc[0]['total']

        query_survived = "SELECT SUM(Survived) as survived FROM passengers"
        survived = self.execute_query(query_survived).iloc[0]['survived']

        print(f"Всего пассажиров: {total_passengers}")
        print(f"Выжило: {survived} ({survived / total_passengers * 100:.1f}%)")
        print("=" * 50)

        # Создаем все графики
        self.plot_survival_by_class()
        self.plot_survival_rate_by_gender_class()
        self.plot_age_distribution()
        self.plot_fare_distribution()
        self.plot_correlation_heatmap()
