# Churn Prediction Model (Прогнозирование оттока клиентов)

## Бизнес-задача

Снижение показателя **Churn Rate** (отток клиентов) за счёт:
- Раннего выявления клиентов с высоким риском ухода
- Сегментации клиентов по уровню риска
- Разработки персонализированных предложений удержания

## Стек технологий

- **Язык**  Python 3.9+ 
- **Анализ данных**  Pandas, NumPy 
- **Визуализация**  Matplotlib, Seaborn 
- **Машинное обучение**  Scikit-learn
- **Среда разработки**  Jupyter Notebook, VS Code 
- **Версионирование**  Git, GitHub 

## Быстрый старт

### 1. Клонирование репозитория
```bash
git clone https://github.com/T1zoo/churn-prediction.git
cd churn-prediction
```

## 2. Установка зависимостей
pip install -r requirements.txt

## 3. Подготовка данных
- Скачайте датасет Telco Customer Churn по ссылке https://www.kaggle.com/datasets/blastchar/telco-customer-churn?spm=a2ty_o01.29997173.0.0.72685171jp6ghC
- Положите файл WA_Fn-UseC_-Telco-Customer-Churn.csv в папку data/raw/

## 4. Запустите ноутбук
notebooks/analysis.ipynb

## 5. Предсказание для нового клиента
python src/predict.py '{"MonthlyCharges": 79.85, "tenure": 1, "Contract": 0, "PaymentMethod": 3}'

Пример вывода:
(json)
{
  "churn": true,
  "probability": 0.87,
  "risk_level": "HIGH"
}

***Результаты***
Были протестированы 3 модели классификации:
1) Logistic Regression 
2) Random Forest
3) Gradient Boosting

Оценка метрик моделей:

**Logistic Regression**
- ROC-AUC:  0.8402
- Precision: 0.6426
- Recall:    0.5481
- F1-Score:  0.5916

**Random Forest**
- ROC-AUC:  0.8218
- Precision: 0.6388
- Recall:    0.5107
- F1-Score:  0.5676

**Gradient Boosting**
- ROC-AUC:  0.8449
- Precision: 0.6655
- Recall:    0.5053
- F1-Score:  0.5745

Лучшей моделью оказалась: **Gradient Boosting**

Почему ROC-AUC?
- В задаче оттока классы несбалансированы (73% не уходят, 27% уходят). 
- Accuracy вводит в заблуждение - модель, всегда предсказывающая «не уйдет», получит 73% accuracy, но бесполезна.
- ROC-AUC оценивает качественнее в задачах с дисбалансом классов.

**Проект машинного обучения для портфолио**  
Студент 3 курса БФУ им. И. Канта, направление «Искусственный интеллект и анализ данных»

Проект по прогнозированию оттока клиентов телеком-оператора. 

Цель — выявить клиентов с высоким риском ухода для предотвращения потери выручки и разработки таргетированных стратегий удержания.

***Автор***
- *Ганцевский Денис Николаевич*
- *Email: denimax2006@gmail.com*
- *Telegram: @T1zoo*
