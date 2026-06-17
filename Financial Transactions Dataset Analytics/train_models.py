"""
train_stage3.py
===============
Этап 3: обучение и оценка ML-моделей на «чистом» датасете.
Без feature engineering, без SMOTE.

Модели:
    - Logistic Regression  (линейная baseline-модель)
    - Random Forest        (ансамбль на бэггинге)
    - XGBoost              (ансамбль на бустинге)

Метрики:
    - Precision
    - Recall
    - F1-score
    - PR-AUC  (Average Precision)

Стратегия валидации:
    - Временно́е разбиение train/test (по дате, не случайное)
    - TimeSeriesSplit кросс-валидация на train для подбора гиперпараметров
      и оценки стабильности модели во времени

Почему PR-AUC, а не ROC-AUC:
    При дисбалансе классов ~668:1 ROC-AUC завышает реальное качество модели,
    т.к. учитывает TN, которых очень много. PR-AUC (Average Precision)
    фокусируется только на классе фрода и более информативен.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

from default_preprocess_data import load_and_merge, preprocess, time_split


# =============================================================================
# 1. КОНФИГУРАЦИЯ
# =============================================================================

RANDOM_STATE   = 42    # фиксируем seed для воспроизводимости
CV_FOLDS       = 5     # количество фолдов TimeSeriesSplit
TEST_CUTOFF_YEAR = 2019  # граница train/test по времени


# =============================================================================
# 2. ВЫЧИСЛЕНИЕ МЕТРИК
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict:
    """
    Вычисляет Precision, Recall, F1, PR-AUC и выводит отчёт.

    Args:
        y_true     : истинные метки
        y_pred     : предсказанные бинарные метки (0 / 1)
        y_prob     : вероятности класса 1 (нужны для PR-AUC)
        model_name : название модели

    Returns:
        Словарь с метриками (для сводной таблицы).
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    pr_auc    = average_precision_score(y_true, y_prob)

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  PR-AUC    : {pr_auc:.4f}")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n  Confusion Matrix:")
    print(f"  {'':28} Predicted Fraud  Predicted Legit")
    print(f"  {'Actual Fraud':<28} {tp:<17}  {fn}")
    print(f"  {'Actual Legit':<28} {fp:<17}  {tn}")

    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

    return {
        "Model":     model_name,
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1-score":  round(f1, 4),
        "PR-AUC":    round(pr_auc, 4),
    }


# =============================================================================
# 3. КРОСС-ВАЛИДАЦИЯ НА TRAIN (TimeSeriesSplit)
# =============================================================================

def cv_report(model, X_train: np.ndarray, y_train: np.ndarray, model_name: str) -> None:
    """
    Запускает TimeSeriesSplit кросс-валидацию на обучающей выборке
    и выводит средние Precision / Recall / F1 / PR-AUC по фолдам.

    TimeSeriesSplit не перемешивает данные: каждый следующий фолд
    использует бо́льшую обучающую часть и валидирует на более поздних данных.
    Это соответствует логике «модель обучается на прошлом, предсказывает будущее».

    Args:
        model      : sklearn-совместимый классификатор (уже инициализированный)
        X_train    : матрица признаков обучающей выборки
        y_train    : метки обучающей выборки
        model_name : название модели для вывода
    """
    print(f"\n[CV] {model_name}: TimeSeriesSplit ({CV_FOLDS} фолдов)...")

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    precisions, recalls, f1s, pr_aucs = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # Пропускаем фолды, где в val нет фрода (PR-AUC не определён)
        if y_val.sum() == 0:
            print(f"  Фолд {fold}: нет фрода в val — пропущен")
            continue

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        pr_aucs.append(average_precision_score(y_val, y_prob))

        print(f"  Фолд {fold}: Precision={precisions[-1]:.4f} "
              f"Recall={recalls[-1]:.4f} F1={f1s[-1]:.4f} "
              f"PR-AUC={pr_aucs[-1]:.4f}")

    if f1s:
        print(f"  Среднее : Precision={np.mean(precisions):.4f} "
              f"Recall={np.mean(recalls):.4f} F1={np.mean(f1s):.4f} "
              f"PR-AUC={np.mean(pr_aucs):.4f}")


# =============================================================================
# 4. МОДЕЛИ
# =============================================================================

def run_logistic_regression(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> dict:
    """
    Logistic Regression — линейная baseline-модель.

    Особенности:
        - Требует масштабирования: StandardScaler обучается ТОЛЬКО на train,
          затем применяется к test (предотвращает data leakage).
        - class_weight='balanced': автоматически взвешивает классы обратно
          пропорционально их частоте, компенсируя дисбаланс без SMOTE.
        - solver='saga': эффективен на больших датасетах, поддерживает L1/L2.
    """
    print("\n[MODEL] Logistic Regression")

    # Масштабирование — обязательно для LR
    scaler      = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver="lbfgs",  # быстрее saga на больших данных
        max_iter=1000,
        tol=1e-3,  # менее строгий критерий сходимости — быстрее
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # Кросс-валидация на train
    cv_report(model, X_tr_scaled, y_train, "Logistic Regression")

    # Финальное обучение на полном train -> оценка на test
    model.fit(X_tr_scaled, y_train)
    y_pred = model.predict(X_te_scaled)
    y_prob = model.predict_proba(X_te_scaled)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob, "Logistic Regression")


def run_random_forest(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> dict:
    """
    Random Forest — ансамбль на бэггинге.

    Особенности:
        - class_weight='balanced_subsample': взвешивание классов
          вычисляется отдельно для каждого дерева на его бутстрэп-выборке
          (более устойчиво при сильном дисбалансе, чем 'balanced').
        - n_estimators=100: стандартный минимум для стабильных оценок.
        - max_depth=20: ограничиваем глубину для предотвращения переобучения.
        - Масштабирование НЕ нужно (деревья инвариантны к шкале признаков).
    """
    print("\n[MODEL] Random Forest")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Кросс-валидация на train
    cv_report(model, X_train, y_train, "Random Forest")

    # Финальное обучение на полном train -> оценка на test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob, "Random Forest")


def run_xgboost(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> dict:
    """
    XGBoost — ансамбль на градиентном бустинге.

    Особенности:
        - scale_pos_weight: ключевой параметр для дисбаланса классов.
          Задаётся как отношение числа негативных примеров к позитивным
          (n_legit / n_fraud). При соотношении 668:1 это ~668.
          Это эквивалент class_weight='balanced' для XGBoost.
        - eval_metric='aucpr': оптимизируем PR-AUC напрямую при обучении,
          что согласуется с выбранной основной метрикой исследования.
        - tree_method='hist': быстрый алгоритм построения деревьев,
          эффективен на больших датасетах.
        - Масштабирование НЕ нужно (деревья инвариантны к шкале признаков).
    """
    print("\n[MODEL] XGBoost")

    # Вычисляем scale_pos_weight на основе train-выборки
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    spw     = n_legit / n_fraud
    print(f"  scale_pos_weight = {spw:.1f} (легит/фрод = {n_legit}/{n_fraud})")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    # Кросс-валидация на train
    cv_report(model, X_train, y_train, "XGBoost")

    # Финальное обучение на полном train -> оценка на test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob, "XGBoost")


# =============================================================================
# 5. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
# =============================================================================

def print_summary(results: list[dict]) -> pd.DataFrame:
    """
    Формирует и выводит сводную таблицу метрик всех моделей.

    Args:
        results: список словарей с метриками каждой модели

    Returns:
        DataFrame со сводной таблицей (для последующего сохранения/сравнения).
    """
    df_results = pd.DataFrame(results).set_index("Model")

    print("\n" + "=" * 60)
    print("  СВОДНАЯ ТАБЛИЦА — ЭТАП 3 (без FE, без SMOTE)")
    print("=" * 60)
    print(df_results.to_string())
    print("=" * 60)

    return df_results


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    from config import (
        TRANSACTIONS_PATH, USERS_PATH, CARDS_PATH,
        MCC_PATH, LABELS_PATH,
    )

    # --- Шаг 1: Загрузка, объединение, предобработка ---
    print("\n[STEP 1] Загрузка и предобработка данных...")
    df_raw       = load_and_merge(
        TRANSACTIONS_PATH, USERS_PATH, CARDS_PATH, MCC_PATH, LABELS_PATH
    )
    df_processed = preprocess(df_raw)

    # --- Шаг 2: Временно́е разбиение на train / test ---
    X_train_df, X_test_df, y_train, y_test = time_split(
        df_processed, test_cutoff_year=TEST_CUTOFF_YEAR
    )

    # Переводим в numpy для совместимости с sklearn / XGBoost
    X_train = X_train_df.values
    X_test  = X_test_df.values
    y_train = y_train.values
    y_test  = y_test.values

    print(f"\n  Признаки ({X_train_df.shape[1]}): {X_train_df.columns.tolist()}")

    # --- Шаг 3: Обучение и оценка ---
    print("\n[STEP 3] Обучение и оценка моделей...")
    results = []
    results.append(run_logistic_regression(X_train, X_test, y_train, y_test))
    results.append(run_random_forest(X_train, X_test, y_train, y_test))
    results.append(run_xgboost(X_train, X_test, y_train, y_test))

    # --- Шаг 4: Сводная таблица ---
    df_summary = print_summary(results)

    # --- Шаг 5: Сохранение результатов ---
    from config import RESULT_DIR
    import os
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = f"{RESULT_DIR}/stage3_results.csv"
    df_summary.to_csv(out_path)
    print(f"\n[INFO] Результаты сохранены: {out_path}")