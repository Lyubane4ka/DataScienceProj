import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_car_data(df):
    df.dropna(inplace=True)

    df['year'] = df['year'].astype(int)
    df['price'] = df['price'].astype(int)
    df['mileage'] = df['mileage'].astype(int)
    df['tax'] = df['tax'].astype(int)
    df['mpg'] = df['mpg'].astype(float)
    df['engineSize'] = df['engineSize'].astype(float)

    print('\nData types after conversion:')
    print(df.dtypes)
    return df
