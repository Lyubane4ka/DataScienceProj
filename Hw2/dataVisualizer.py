from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DataVisualizer:
    """Специализированный класс для визуализации данных кофейни"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.figures = {}
        self.set_style()

    def set_style(self, style: str = 'whitegrid') -> None:
        """Установка стиля графиков"""
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14

    # ГИСТОГРАММЫ
    def add_sales_by_hour_histogram(self, figure_name: str = 'sales_by_hour') -> None:
        """Гистограмма продаж по часам дня"""
        fig, ax = plt.subplots()

        self.df['hour_of_day'].hist(bins=24, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_title('Распределение продаж по часам дня')
        ax.set_xlabel('Час дня')
        ax.set_ylabel('Количество продаж')
        ax.grid(True, alpha=0.3)

        self.figures[figure_name] = fig

    def add_coffee_sales_histogram(self, figure_name: str = 'coffee_sales') -> None:
        """Гистограмма продаж по видам кофе"""
        fig, ax = plt.subplots()

        coffee_counts = self.df['coffee_name'].value_counts()
        coffee_counts.plot(kind='bar', ax=ax, color='sienna', alpha=0.7)

        ax.set_title('Продажи по видам кофе')
        ax.set_xlabel('Вид кофе')
        ax.set_ylabel('Количество продаж')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        self.figures[figure_name] = fig

    def add_payment_method_histogram(self, figure_name: str = 'payment_methods') -> None:
        """Гистограмма методов оплаты"""
        fig, ax = plt.subplots()

        payment_counts = self.df['cash_type'].value_counts()
        payment_counts.plot(kind='bar', ax=ax, color='green', alpha=0.7)

        ax.set_title('Распределение методов оплаты')
        ax.set_xlabel('Метод оплаты')
        ax.set_ylabel('Количество транзакций')
        ax.grid(True, alpha=0.3)

        self.figures[figure_name] = fig

    # ЛИНЕЙНЫЕ ГРАФИКИ
    def add_daily_sales_trend(self, figure_name: str = 'daily_sales_trend') -> None:
        """Линейный график продаж по дням"""
        fig, ax = plt.subplots()

        daily_sales = self.df.groupby('Date')['money'].sum()
        daily_sales.plot(ax=ax, marker='o', linestyle='-', color='red')

        ax.set_title('Динамика ежедневных продаж')
        ax.set_xlabel('Дата')
        ax.set_ylabel('Сумма продаж ($)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        self.figures[figure_name] = fig

    def add_hourly_sales_trend(self, figure_name: str = 'hourly_sales_trend') -> None:
        """Линейный график продаж по часам"""
        fig, ax = plt.subplots()

        hourly_sales = self.df.groupby('hour_of_day')['money'].mean()
        hourly_sales.plot(ax=ax, marker='s', linestyle='-', color='blue')

        ax.set_title('Средние продажи по часам дня')
        ax.set_xlabel('Час дня')
        ax.set_ylabel('Средняя сумма продаж ($)')
        ax.grid(True, alpha=0.3)

        self.figures[figure_name] = fig

    def add_weekly_sales_trend(self, figure_name: str = 'weekly_sales_trend') -> None:
        """Линейный график продаж по дням недели"""
        fig, ax = plt.subplots()

        # Сортируем по правильному порядку дней недели
        weekday_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        weekly_sales = self.df.groupby('Weekday')['money'].sum().reindex(weekday_order)
        weekly_sales.plot(ax=ax, marker='^', linestyle='-', color='purple')

        ax.set_title('Продажи по дням недели')
        ax.set_xlabel('День недели')
        ax.set_ylabel('Сумма продаж ($)')
        ax.grid(True, alpha=0.3)

        self.figures[figure_name] = fig

    # ДИАГРАММЫ РАССЕЯНИЯ
    def add_hour_vs_sales_scatter(self, figure_name: str = 'hour_vs_sales_scatter') -> None:
        """Диаграмма рассеяния: час дня vs сумма продажи"""
        fig, ax = plt.subplots()

        sns.scatterplot(data=self.df, x='hour_of_day', y='money',
                        hue='cash_type', ax=ax, palette='viridis', alpha=0.6)

        ax.set_title('Зависимость суммы продажи от времени дня')
        ax.set_xlabel('Час дня')
        ax.set_ylabel('Сумма продажи ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Метод оплаты')

        self.figures[figure_name] = fig

    def add_coffee_price_scatter(self, figure_name: str = 'coffee_price_scatter') -> None:
        """Диаграмма рассеяния: вид кофе vs цена"""
        fig, ax = plt.subplots(figsize=(14, 8))

        sns.scatterplot(data=self.df, x='coffee_name', y='money',
                        hue='Time_of_Day', ax=ax, palette='coolwarm', alpha=0.7)

        ax.set_title('Цены на разные виды кофе по времени дня')
        ax.set_xlabel('Вид кофе')
        ax.set_ylabel('Цена ($)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Время дня')

        self.figures[figure_name] = fig

    def add_weekday_sales_scatter(self, figure_name: str = 'weekday_sales_scatter') -> None:
        """Диаграмма рассеяния: день недели vs продажи"""
        fig, ax = plt.subplots()

        # Конвертируем дни недели в числовые значения для scatter plot
        weekday_mapping = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}
        df_with_numeric_weekday = self.df.copy()
        df_with_numeric_weekday['weekday_numeric'] = df_with_numeric_weekday['Weekday'].map(weekday_mapping)

        sns.scatterplot(data=df_with_numeric_weekday, x='weekday_numeric', y='money',
                        hue='coffee_name', ax=ax, palette='tab10', alpha=0.6)

        ax.set_title('Продажи по дням недели и видам кофе')
        ax.set_xlabel('День недели (1=Пн, 7=Вс)')
        ax.set_ylabel('Сумма продажи ($)')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Вид кофе', bbox_to_anchor=(1.05, 1), loc='upper left')

        self.figures[figure_name] = fig

    # ДОПОЛНИТЕЛЬНЫЕ ВИЗУАЛИЗАЦИИ
    def add_coffee_sales_pie(self, figure_name: str = 'coffee_sales_pie') -> None:
        """Круговая диаграмма продаж кофе"""
        fig, ax = plt.subplots()

        coffee_counts = self.df['coffee_name'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(coffee_counts)))

        wedges, texts, autotexts = ax.pie(coffee_counts.values, labels=coffee_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)

        ax.set_title('Доля продаж по видам кофе')
        plt.setp(autotexts, size=10, weight="bold")

        self.figures[figure_name] = fig

    def add_time_of_day_sales(self, figure_name: str = 'time_of_day_sales') -> None:
        """Гистограмма продаж по времени дня"""
        fig, ax = plt.subplots()

        time_sales = self.df.groupby('Time_of_Day')['money'].sum()
        time_sales.plot(kind='bar', ax=ax, color=['lightblue', 'lightcoral', 'lightgreen'])

        ax.set_title('Продажи по времени дня')
        ax.set_xlabel('Время дня')
        ax.set_ylabel('Сумма продаж ($)')
        ax.grid(True, alpha=0.3)

        self.figures[figure_name] = fig

    # МЕТОДЫ УПРАВЛЕНИЯ ГРАФИКАМИ
    def remove_figure(self, figure_name: str) -> None:
        """Удаление графика"""
        if figure_name in self.figures:
            plt.close(self.figures[figure_name])
            del self.figures[figure_name]
            print(f"График '{figure_name}' удален")
        else:
            print(f"График '{figure_name}' не найден")

    def show_figure(self, figure_name: str) -> None:
        """Показать конкретный график"""
        if figure_name in self.figures:
            plt.figure(self.figures[figure_name].number)
            plt.show()
        else:
            print(f"График '{figure_name}' не найден")

    def show_all_figures(self) -> None:
        """Показать все графики"""
        for name, fig in self.figures.items():
            print(f"Показ графика: {name}")
            plt.figure(fig.number)
            plt.show()

    def list_figures(self) -> List[str]:
        """Список всех созданных графиков"""
        return list(self.figures.keys())

    def save_figure(self, figure_name: str, filename: str,
                    dpi: int = 300, format: str = 'png') -> None:
        """Сохранение графика"""
        if figure_name in self.figures:
            self.figures[figure_name].savefig(
                filename, dpi=dpi, format=format,
                bbox_inches='tight', facecolor='white'
            )
            print(f"График сохранен как {filename}")
        else:
            print(f"График '{figure_name}' не найден")
