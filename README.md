# FinancialFraudDetection

## Цель проекта
Цель проекта заключается в применении разных ML-алгоритмов для решения задачи детектирования признаков мошенничества (фрода) в финансовых данных с целью сравнения качества обнаружения и определения субоптимального алгоритма и влияния на качество различных техник работы с датасетом

## Используемые ML-алгоритмы
- XGBoost (eXtreme Gradient Boosting)
- Random Forest (Случайный лес)
- Linear Regression (Линейная регрессия)

## Технологический стэк проекта
- **Язык программирования:** Python
- **Библиотеки:** numpy; pandas; sklearn: preprocessing, linear_model, ensemble, model_selection, xgboost, metrics; imblearn.over_sampling

## Источники наборов данных для исследования
- Financial Transactions Dataset: Analytics - https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets
- (Планируется к исследованию позже) Synthetic Financial Datasets For Fraud Detection (PaySim) - https://www.kaggle.com/datasets/ealaxi/paysim1
- (Планируется к исследованию позже) Libra Bank transaction graph over 3 months - https://graphomaly.upb.ro/#datasets

## Структура проекта
+ Financial Transaction Dataset Analytics - директория для хранения файлов по работе с датасетом https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets:
    - **archive** - директория для хранения неактульных файлов
    - **results** - директория для хранения актуальных результатов
    - **Подбор порога** - директория для хранения прогонов, где пытался выполнять подбор порога классификации
    - **config.py** - файл с путями расположения файлов датасета и директорий для локального сохранения результатов
    - **default_preprocess_data.py** - файл с логикой обработки исходных файлов, в том числе удаление некоторых фич и мердж всех файлов в один общий (без генерации новых признаков)
    - **eda.py** - файл с логикой разведочного анализа EDA
    - **fe_preprocess_data.py** - файл с логикой обработки исходных файлов, в том числе удаление некоторых фич и мердж всех файлов в один общий (с генерацией новых признаков)
    - **train_models_1.py** - файл с экспериментом 1 (без подбора порога, без FE, без SMOTE)
    - **train_models_2.py** - файл с экспериментом 2 (без подбора порога, c FE, без SMOTE)
    - **train_models_3.py** - файл с экспериментом 3 (без подбора порога, без FE, c SMOTE)
    - **train_models_4.py** - файл с экспериментом 4 (без подбора порога, c FE, c SMOTE)
    - **README.md** - описание проекта
