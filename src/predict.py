import pandas as pd
import joblib
import json
import sys
from pathlib import Path

# Добавляем путь к src, если запускаем извне
sys.path.append(str(Path(__file__).parent))
from preprocess import load_and_clean, encode_target, encode_categorical, prepare_features

def predict_client(input_json: str, model_path: str, encoders_path: str, scaler_path: str) -> dict:
    """
    Предсказание оттока для одного клиента.
    
    Args:
        input_json: JSON-строка с данными клиента
        model_path: путь к сохранённой модели (.pkl)
        encoders_path: путь к словарю энкодеров
        scaler_path: путь к скалеру
    
    Returns:
        dict с предсказанием и вероятностью
    """
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)
    
    # Парсинг входных данных
    client_data = json.loads(input_json)
    df_client = pd.DataFrame([client_data])
    
    X_client = scaler.transform(df_client)
    prediction = model.predict(X_client)[0]
    probability = model.predict_proba(X_client)[0][1]
    
    return {
        'churn': bool(prediction),
        'probability': float(probability),
        'risk_level': 'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.4 else 'LOW'
    }

if __name__ == '__main__':
    # Пример использования из командной строки
    if len(sys.argv) < 2:
        print("Использование: python predict.py '{\"MonthlyCharges\": 70, \"tenure\": 12, ...}'")
        sys.exit(1)
    
    input_data = sys.argv[1]
    result = predict_client(
        input_data,
        model_path='../models/best_model.pkl',
        encoders_path='../models/encoders.pkl',
        scaler_path='../models/scaler.pkl'
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False))