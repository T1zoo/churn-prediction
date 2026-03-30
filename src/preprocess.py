# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_and_clean(filepath: str) -> pd.DataFrame:
    """Загрузка и базовая очистка данных"""
    df = pd.read_csv(filepath)
    

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    return df

def encode_target(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """Кодирование целевой переменной: Yes/No -> 1/0"""
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    return df

def encode_categorical(df: pd.DataFrame, cat_cols: list) -> tuple:
    """Label Encoding категориальных признаков"""
    encoders = {}
    df_encoded = df.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        # Заполняем пропуски модой перед кодированием
        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mode()[0])
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders

def prepare_features(df: pd.DataFrame, target_col: str = 'Churn'):
    """Разделение на признаки и целевую переменную, масштабирование"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler