"""
train_stage4.py
================
Этап 4: обучение и оценка ML-моделей на датасете с feature engineering.
Без SMOTE. Сравнивается с baseline Этапа 3 для оценки эффекта FE.

Модели и гиперпараметры -- ЗАФИКСИРОВАНЫ такими же, как в Этапе 3
(осознанное методологическое решение: варьируются только FE/SMOTE,
подбор гиперпараметров не входит в задачи исследования).

Порог классификации -- 0.5 (по итогам экспериментов с CV-порогом на
Этапе 3 было решено вернуться к дефолтному порогу для всех последующих
этапов, см. раздел 4.2 работы "Подбор порога классификации").

Метрики: Precision, Recall, F1-score, PR-AUC -- как на предыдущих этапах.
Стратегия валидации: временное разбиение train/test + TimeSeriesSplit CV
на train (справочно, для наблюдения за стабильностью модели во времени).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

from fe_preprocess_data import load_and_merge, time_split, preprocess_fe, add_mcc_risk_group


# =============================================================================
# 1. КОНФИГУРАЦИЯ (идентична Этапу 3)
# =============================================================================

RANDOM_STATE      = 42
CV_FOLDS          = 5
TEST_CUTOFF_YEAR  = 2018
CLASSIFICATION_THRESHOLD = 0.5   # зафиксирован по итогам Этапа 3


# =============================================================================
# 2. ВЫЧИСЛЕНИЕ МЕТРИК (порог 0.5, фиксированный)
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict:
    """Вычисляет Precision, Recall, F1, PR-AUC при пороге 0.5 и печатает отчёт."""
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
# 3. КРОСС-ВАЛИДАЦИЯ НА TRAIN (TimeSeriesSplit) -- справочно
# =============================================================================

def cv_report(
    model, X_train: np.ndarray, y_train: np.ndarray,
    model_name: str, dates_train: np.ndarray = None,
) -> None:
    """
    TimeSeriesSplit кросс-валидация на train -- только для наблюдения за
    стабильностью модели во времени (как в Этапе 3). Порог не подбирается,
    используется стандартный model.predict() (порог 0.5).
    """
    print(f"\n[CV] {model_name}: TimeSeriesSplit ({CV_FOLDS} фолдов)...")

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    precisions, recalls, f1s, pr_aucs = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        fraud_train = y_tr.sum()
        fraud_val   = y_val.sum()
        print(f"\n  Фолд {fold} | "
              f"train: {len(y_tr):,} записей, фрод: {fraud_train:,} "
              f"({fraud_train / len(y_tr) * 100:.3f}%) | "
              f"val: {len(y_val):,} записей, фрод: {fraud_val:,} "
              f"({fraud_val / len(y_val) * 100:.3f}%)")

        if dates_train is not None:
            dates = pd.Series(dates_train)
            tr_dates  = dates.iloc[tr_idx]
            val_dates = dates.iloc[val_idx]
            print(f"           train: {tr_dates.min().date()} — {tr_dates.max().date()} | "
                  f"val: {val_dates.min().date()} — {val_dates.max().date()}")

        if fraud_val == 0:
            print(f"  Фолд {fold}: нет фрода в val — пропущен")
            continue

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        precisions.append(precision_score(y_val, y_pred, zero_division=0))
        recalls.append(recall_score(y_val, y_pred, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        pr_aucs.append(average_precision_score(y_val, y_prob))

        print(f"  Результат : Precision={precisions[-1]:.4f} "
              f"Recall={recalls[-1]:.4f} F1={f1s[-1]:.4f} "
              f"PR-AUC={pr_aucs[-1]:.4f}")

    if f1s:
        print(f"\n  Среднее : Precision={np.mean(precisions):.4f} "
              f"Recall={np.mean(recalls):.4f} F1={np.mean(f1s):.4f} "
              f"PR-AUC={np.mean(pr_aucs):.4f}")


# =============================================================================
# 4. МОДЕЛИ (гиперпараметры идентичны Этапу 3)
# =============================================================================

def run_logistic_regression(
    X_train, X_test, y_train, y_test, dates_train=None,
) -> dict:
    print("\n[MODEL] Logistic Regression")

    scaler      = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver="lbfgs", max_iter=1000, tol=1e-3,
        class_weight="balanced", random_state=RANDOM_STATE,
    )

    cv_report(model, X_tr_scaled, y_train, "Logistic Regression", dates_train)

    model.fit(X_tr_scaled, y_train)
    y_pred = (model.predict_proba(X_te_scaled)[:, 1] >= CLASSIFICATION_THRESHOLD).astype(int)
    y_prob = model.predict_proba(X_te_scaled)[:, 1]

    return compute_metrics(y_test, y_pred, y_prob, "Logistic Regression")


def run_random_forest(
    X_train, X_test, y_train, y_test, dates_train=None,
) -> dict:
    print("\n[MODEL] Random Forest")

    model = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    cv_report(model, X_train, y_train, "Random Forest", dates_train)

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= CLASSIFICATION_THRESHOLD).astype(int)

    return compute_metrics(y_test, y_pred, y_prob, "Random Forest")


def run_xgboost(
    X_train, X_test, y_train, y_test, dates_train=None,
) -> dict:
    print("\n[MODEL] XGBoost")

    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    spw     = n_legit / n_fraud
    print(f"  scale_pos_weight = {spw:.1f} (легит/фрод = {n_legit}/{n_fraud})")

    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )

    cv_report(model, X_train, y_train, "XGBoost", dates_train)

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= CLASSIFICATION_THRESHOLD).astype(int)

    return compute_metrics(y_test, y_pred, y_prob, "XGBoost")


# =============================================================================
# 5. СВОДНАЯ ТАБЛИЦА
# =============================================================================

def print_summary(results: list) -> pd.DataFrame:
    df_results = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 60)
    print("  СВОДНАЯ ТАБЛИЦА — ЭТАП 4 (с FE, без SMOTE)")
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
        MCC_PATH, LABELS_PATH, RESULT_DIR,
    )
    import os

    # --- Шаг 1: Загрузка, объединение, предобработка + FE ---
    print("\n[STEP 1] Загрузка данных...")
    df_raw = load_and_merge(
        TRANSACTIONS_PATH, USERS_PATH, CARDS_PATH, MCC_PATH, LABELS_PATH
    )
    df_fe = preprocess_fe(df_raw)

    # --- Шаг 2: Временное разбиение (переиспользуем time_split из Этапа 3) ---
    X_train_df, X_test_df, y_train, y_test = time_split(
        df_fe, test_cutoff_year=TEST_CUTOFF_YEAR
    )

    # --- Шаг 3: mcc_risk_group -- строится ПОСЛЕ разбиения, только по train ---
    X_train_df, X_test_df = add_mcc_risk_group(X_train_df, X_test_df, y_train)

    # Извлекаем даты для отображения границ фолдов CV (как в Этапе 3)
    df_fe["_date_for_split"] = pd.to_datetime(df_fe["_date_for_split"])
    train_mask = df_fe["_date_for_split"].dt.year < TEST_CUTOFF_YEAR
    test_mask  = df_fe["_date_for_split"].dt.year >= TEST_CUTOFF_YEAR
    dates_train = df_fe.loc[train_mask, "_date_for_split"].values
    dates_test  = df_fe.loc[test_mask,  "_date_for_split"].values

    X_train = X_train_df.values
    X_test  = X_test_df.values
    y_train = y_train.values
    y_test  = y_test.values

    print(f"\n  Признаки ({X_train_df.shape[1]}): {X_train_df.columns.tolist()}")

    dates_test_s = pd.Series(pd.to_datetime(dates_test))
    print(f"\n  Тестовая выборка: "
          f"{dates_test_s.min().date()} — {dates_test_s.max().date()} "
          f"({len(dates_test_s):,} записей)")

    # --- Шаг 4: Обучение и оценка ---
    print("\n[STEP 4] Обучение и оценка моделей...")
    results = []
    results.append(run_logistic_regression(X_train, X_test, y_train, y_test, dates_train))
    results.append(run_random_forest(X_train, X_test, y_train, y_test, dates_train))
    results.append(run_xgboost(X_train, X_test, y_train, y_test, dates_train))

    # --- Шаг 5: Сводная таблица ---
    df_summary = print_summary(results)

    # --- Шаг 6: Сохранение результатов ---
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = f"{RESULT_DIR}/stage4_results.csv"
    df_summary.to_csv(out_path)
    print(f"\n[INFO] Результаты сохранены: {out_path}")