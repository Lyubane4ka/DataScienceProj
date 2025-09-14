import pandas as pd


class DataLoader:
    """Класс для загрузки данных из csv file"""

    @staticmethod
    def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {file_path} не найден")
        except Exception as e:
            raise Exception(f"Ошибка при загрузке CSV: {str(e)}")
