"""
Скрипт обучения моделей для прогнозирования оттока клиентов.
Сравнивает несколько алгоритмов, выбирает лучший и сохраняет артефакты.
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from src.preprocess import (
    load_and_clean, 
    encode_target, 
    encode_categorical, 
    prepare_features
)

DATA_PATH = Path(__file__).parent.parent / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
MODELS_PATH = Path(__file__).parent.parent / 'models'
RANDOM_STATE = 42
TEST_SIZE = 0.2

def train_models(X_train, y_train, X_test, y_test):
    """
    Обучает несколько моделей и возвращает результаты сравнения.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    }
    
    results = []
    trained_models = {}
    
    print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛЕЙ")
    for name, model in models.items():
        print(f"\nОбучение модели: {name}")
        
        # Обучение
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Предсказание
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Метрики
        roc_auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"ROC-AUC:  {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        results.append({
            'model_name': name,
            'model': model,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return results, trained_models


def save_artifacts(best_model, scaler, encoders, models_path: Path):
    """
    Сохраняет лучшую модель и артефакты (скалер, энкодеры).
    """
    models_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_model, models_path / 'best_model.pkl')
    joblib.dump(scaler, models_path / 'scaler.pkl')
    joblib.dump(encoders, models_path / 'encoders.pkl')
    
    print("\nАртефакты сохранены:")
    print(f"{models_path / 'best_model.pkl'}")
    print(f"{models_path / 'scaler.pkl'}")
    print(f"{models_path / 'encoders.pkl'}")


def print_final_report(results):
    """
    Выводит итоговый рейтинг моделей.
    """
    print("\n")
    # Сортировка по убыванию ROC-AUC
    results_sorted = sorted(results, key=lambda x: x['roc_auc'], reverse=True)
    
    for i, res in enumerate(results_sorted, 1):
        print(f"{i}. {res['model_name']}: ROC-AUC = {res['roc_auc']:.4f}")
    
    return results_sorted[0]  # Возвращаем лучшую модель


def main():
    """
    Основная функция запуска обучения.
    """
    #Загрузка данных
    print("\nЗагрузка данных...")
    df = load_and_clean(str(DATA_PATH))
    print(f"Загружено строк: {len(df)}")
    
    #Кодирование целевой переменной
    print("\nКодирование целевой переменной...")
    df = encode_target(df)
    
    #Кодирование категориальных признаков
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Безопасное удаление таргета (если он вдруг ещё в object)
    if 'Churn' in cat_cols:
        cat_cols.remove('Churn')

    df_encoded, encoders = encode_categorical(df, cat_cols)
    
    #Подготовка признаков
    print("\nПодготовка признаков (масштабирование)...")
    X, y, scaler = prepare_features(df_encoded)
    
    #Разбиение на train/test
    print("\nРазбиение на обучающую и тестовую выборки...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y  # Важно для несбалансированных классов!
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    #Обучение моделей
    results, trained_models = train_models(X_train, y_train, X_test, y_test)
    
    #Выбор лучшей модели
    best = print_final_report(results)
    print(f"\nЛучшая модель: {best['model_name']}")
    
    #Сохранение артефактов
    print("\nСохранение модели и артефактов...")
    save_artifacts(best['model'], scaler, encoders, MODELS_PATH)
    
    #Отчёт по лучшей модели
    print("\n")
    y_pred_best = best['model'].predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=['No Churn', 'Churn']))
    
    print("\nОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")


if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nОШИБКА: Файл данных не найден!")
        print(f"Проверьте путь: {DATA_PATH}")
        print(f"Скачайте датасет с Kaggle и положите в data/raw/")
        sys.exit(1)
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        sys.exit(1)