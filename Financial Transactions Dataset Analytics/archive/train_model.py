"""
Обучение и оценка ML-моделей для задачи обнаружения фрода.
Датасет: Financial Transactions Dataset: Analytics
Источник: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets

Модели:
    - Isolation Forest  (unsupervised)
    - Random Forest     (supervised)
    - Logistic Regression (supervised)

Метрики: Accuracy, Precision, Recall, F1-score, AUC
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# Импорт функций предобработки из соседнего файла
from preprocess_data import load_data, preprocess
from config import TRANSACTIONS_PATH, LABELS_PATH

# =============================================================================
# 1. КОНФИГУРАЦИЯ
# =============================================================================

TRANSACTIONS_PATH = TRANSACTIONS_PATH
LABELS_PATH = LABELS_PATH

TEST_SIZE = 0.2  # доля тестовой выборки
RANDOM_STATE = 42  # фиксируем seed для воспроизводимости результатов
CV_FOLDS = 5  # количество фолдов для кросс-валидации


# =============================================================================
# 2. РАЗДЕЛЕНИЕ НА ПРИЗНАКИ И ЦЕЛЕВУЮ ПЕРЕМЕННУЮ
# =============================================================================

def split_features_target(df: pd.DataFrame):
    """
    Разделяет датафрейм на матрицу признаков X и вектор меток y.

    Args:
        df: предобработанный DataFrame

    Returns:
        X: DataFrame с признаками
        y: Series с целевой переменной (isFraud)
    """
    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]
    return X, y


# =============================================================================
# 3. ВЫЧИСЛЕНИЕ МЕТРИК
# =============================================================================

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    model_name: str) -> dict:
    """
    Вычисляет и выводит метрики качества модели.

    Args:
        y_true:     истинные метки
        y_pred:     предсказанные метки (0 / 1)
        y_prob:     вероятности класса 1 (для AUC)
        model_name: название модели (для вывода)

    Returns:
        Словарь с метриками
    """
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
    }

    print(f"\n{'=' * 55}")
    print(f"  {model_name}")
    print(f"{'=' * 55}")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"  {metric:<12}: {value:.4f}")

    # Confusion matrix (строки = Predicted, столбцы = Actual)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"  {'':25} Actual 1 (Fraud)  Actual 0 (No Fraud)")
    print(f"  {'Predicted 1 (Fraud)':<25} {tp:<18}  {fp}")
    print(f"  {'Predicted 0 (No Fraud)':<25} {fn:<18}  {tn}")

    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

    return metrics


# =============================================================================
# 4. МОДЕЛИ
# =============================================================================

def run_isolation_forest(X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         y_test: np.ndarray) -> dict:
    """
    Isolation Forest — unsupervised модель обнаружения аномалий.

    Особенности:
        - Обучается без меток (на всём X_train)
        - Предсказывает -1 (аномалия) / 1 (норма) -> переводим в 1 / 0
        - Балансировка через contamination (ожидаемая доля аномалий)
        - Кросс-валидация не применяется (unsupervised алгоритм)
    """
    # contamination — ожидаемая доля мошеннических транзакций в данных.
    # Задаём вручную на основе знания о датасете; можно подобрать по бизнес-логике.
    contamination = float(y_test.mean())

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE
    )

    # Isolation Forest обучается без меток
    model.fit(X_train)

    # Предсказания: -1 -> фрод (1), 1 -> норма (0)
    raw_pred = model.predict(X_test)
    y_pred = np.where(raw_pred == -1, 1, 0)

    # Вероятности аномальности: инвертируем score (чем ниже score, тем аномальнее)
    raw_scores = model.decision_function(X_test)
    y_prob = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

    return compute_metrics(y_test, y_pred, y_prob, "Isolation Forest")


def run_random_forest(X_train: pd.DataFrame,
                      X_test: pd.DataFrame,
                      y_train: np.ndarray,
                      y_test: np.ndarray) -> dict:
    """
    Random Forest — supervised классификатор.

    Особенности:
        - class_weight='balanced' для борьбы с дисбалансом классов
        - Кросс-валидация (StratifiedKFold) для надёжной оценки на train
        - Финальная оценка на отложенной тестовой выборке
    """
    model = RandomForestClassifier(
        n_estimators=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=4
    )

    # Кросс-валидация на обучающей выборке
    print("\n[INFO] Random Forest: кросс-валидация...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_probs = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=4, verbose=2)
    cv_preds = np.argmax(cv_probs, axis=1)
    cv_f1 = f1_score(y_train, cv_preds, zero_division=0)
    print(f"[INFO] CV F1-score (среднее по {CV_FOLDS} фолдам): {cv_f1:.4f}")

    # Обучение на полном train и финальная оценка на test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob, "Random Forest")


def run_logistic_regression(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: np.ndarray,
                            y_test: np.ndarray) -> dict:
    """
    Logistic Regression — supervised классификатор (baseline).

    Особенности:
        - Требует масштабирования признаков (StandardScaler)
        - class_weight='balanced' для борьбы с дисбалансом классов
        - Кросс-валидация (StratifiedKFold) для надёжной оценки на train
        - Финальная оценка на отложенной тестовой выборке

    Важно: Scaler обучается ТОЛЬКО на train, затем применяется к test.
    Это предотвращает утечку данных (data leakage).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    # Кросс-валидация на обучающей выборке
    print("\n[INFO] Logistic Regression: кросс-валидация...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_probs = cross_val_predict(model, X_train_scaled, y_train, cv=cv, method="predict_proba")
    cv_preds = np.argmax(cv_probs, axis=1)
    cv_f1 = f1_score(y_train, cv_preds, zero_division=0)
    print(f"[INFO] CV F1-score (среднее по {CV_FOLDS} фолдам): {cv_f1:.4f}")

    # Обучение на полном train и финальная оценка на test
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob, "Logistic Regression")


# =============================================================================
# 5. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
# =============================================================================

def print_summary(results: list[dict]) -> pd.DataFrame:
    """
    Формирует и выводит сводную таблицу метрик всех моделей.

    Args:
        results: список словарей с метриками каждой модели

    Returns:
        DataFrame со сводной таблицей
    """
    df_results = pd.DataFrame(results).set_index("Model")
    df_results = df_results.round(4)

    print("\n" + "=" * 55)
    print("  СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 55)
    print(df_results.to_string())
    print("=" * 55)

    return df_results


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    # --- Шаг 1: Загрузка и предобработка ---
    print("\n[STEP 1] Загрузка и предобработка данных...")
    df_merged = load_data(TRANSACTIONS_PATH, LABELS_PATH)
    df_processed = preprocess(df_merged)

    # --- Шаг 2: Разделение на признаки и целевую переменную ---
    print("\n[STEP 2] Разделение на X и y...")
    X, y = split_features_target(df_processed)

    # --- Шаг 3: Разделение на train / test ---
    # stratify=y гарантирует, что доля фрода одинакова в обеих выборках
    print("\n[STEP 3] Разделение на train / test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} записей | Test: {len(X_test)} записей")
    print(f"[INFO] Доля фрода в train: {y_train.mean():.4f}")
    print(f"[INFO] Доля фрода в test:  {y_test.mean():.4f}")

    # --- Шаг 4: Обучение и оценка моделей ---
    print("\n[STEP 4] Обучение и оценка моделей...")
    results = []
    results.append(run_isolation_forest(X_train, X_test, y_test))
    results.append(run_random_forest(X_train, X_test, y_train, y_test))
    results.append(run_logistic_regression(X_train, X_test, y_train, y_test))

    # --- Шаг 5: Сводная таблица ---
    print("\n[STEP 5] Сводная таблица результатов...")
    df_summary = print_summary(results)
