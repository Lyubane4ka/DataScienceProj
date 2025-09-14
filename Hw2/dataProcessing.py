from typing import List, Dict, Optional, Union
import numpy as np
import pandas as pd


class DataProcessing:
    """Класс для предобработки и обработки данных"""

    def __init__(self, df: pd.DataFrame = None):
        """Конструктор класса"""
        if df is not None:
            self.df = df.copy()
        else:
            self.df = pd.DataFrame()

    def set_data(self, df: pd.DataFrame) -> None:
        """Установка данных"""
        self.df = df.copy()

    def get_empty_values(self) -> pd.DataFrame:
        """Пропущенные значения"""
        total = self.df.isnull().sum()
        percent = (self.df.isnull().sum() / self.df.shape[0]) * 100
        types = self.df.dtypes

        missing_values = pd.DataFrame({
            'total': total,
            'percent': percent.round(2),
            'types': types
        })
        return missing_values.sort_values('percent', ascending=False)

    def remove_duplicates(self) -> None:
        """Удаление дубликатов"""
        self.df = self.df.drop_duplicates()

    def fill_missing_values(self, strategy: str = 'mean',
                            columns: Optional[List[str]] = None,
                            fill_value: Optional[Union[int, float, str]] = None) -> None:
        """Заполнение пропущенных значений"""
        if columns is None:
            columns = self.df.columns

        for column in columns:
            if column not in self.df.columns:
                continue

            if self.df[column].isnull().sum() > 0:
                if strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[column]):
                    self.df[column].fillna(self.df[column].mean(), inplace=True)
                elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[column]):
                    self.df[column].fillna(self.df[column].median(), inplace=True)
                elif strategy == 'mode':
                    mode_value = self.df[column].mode()[0] if not self.df[column].mode().empty else fill_value
                    self.df[column].fillna(mode_value, inplace=True)
                elif strategy == 'constant':
                    if fill_value is not None:
                        self.df[column].fillna(fill_value, inplace=True)
                    else:
                        raise ValueError('Для стратегии constant необходимо указать fill_value')

    def get_data_info(self) -> Dict:
        """Получение общей информации о данных"""
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum()
        }

    def get_numeric_columns(self) -> List[str]:
        """Получение списка числовых колонок"""
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def get_categorical_columns(self) -> List[str]:
        """Получение списка категориальных колонок"""
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def get_processed_data(self) -> pd.DataFrame:
        """Получение обработанных данных"""
        return self.df.copy()

    def show_head(self, n: int = 5) -> pd.DataFrame:
        """Показать первые n строк"""
        return self.df.head(n)
