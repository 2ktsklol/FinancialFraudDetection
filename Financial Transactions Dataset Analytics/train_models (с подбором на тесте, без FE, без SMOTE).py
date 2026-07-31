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
    - TimeSeriesSplit кросс-валидация на train для оценки стабильности
      модели во времени

Почему PR-AUC, а не ROC-AUC:
    При дисбалансе классов ~668:1 ROC-AUC завышает реальное качество модели,
    т.к. учитывает TN, которых очень много. PR-AUC (Average Precision)
    фокусируется только на классе фрода и более информативен.

Подбор порога классификации:
    Метрики Precision/Recall/F1 при бинарном пороге по умолчанию (0.5)
    сильно зависят от него и могут вводить в заблуждение при экстремальном
    дисбалансе классов (см. baseline-результаты: RF Precision=1.0 при
    Recall=0.02 несмотря на PR-AUC=0.56). Поэтому compute_metrics
    дополнительно подбирает оптимальный порог по максимуму F1 на PR-кривой
    и выводит метрики при обоих порогах — 0.5 и оптимальном.

    ВАЖНО (методологическая оговорка для раздела "Ограничения" в статье):
    оптимальный порог здесь подбирается по PR-кривой НА ТЕСТОВОЙ выборке.
    Это методологически не совсем чисто — порог фактически "подсматривает"
    в тестовые метки. Такой порог показывает, ГДЕ теоретически лежит
    оптимум и насколько плохо/хорошо выбран порог 0.5 по умолчанию,
    но НЕ является порогом, который можно было бы использовать в проде
    без знания будущих меток. Пороги при 0.5 остаются основным честным
    сравнением между этапами 3-6; оптимальный порог — вспомогательная
    диагностика. Более строгий вариант (подбор порога по train/CV,
    фиксация и применение к тесту) можно внедрить позже при необходимости.
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
    confusion_matrix, precision_recall_curve,
)
from xgboost import XGBClassifier

from default_preprocess_data import load_and_merge, preprocess, time_split


# =============================================================================
# 1. КОНФИГУРАЦИЯ
# =============================================================================

RANDOM_STATE     = 42    # фиксируем seed для воспроизводимости
CV_FOLDS         = 5     # количество фолдов TimeSeriesSplit
TEST_CUTOFF_YEAR = 2018  # граница train/test по времени


# =============================================================================
# 2. ВЫЧИСЛЕНИЕ МЕТРИК (порог 0.5 + оптимальный порог по F1)
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict:
    """
    Вычисляет и печатает метрики при двух порогах классификации:

        1) 0.5              -- дефолтный порог (для честного сравнения
                                этапов 3-6 между собой и с baseline)
        2) порог, максимизирующий F1 на PR-кривой -- диагностический,
                                показывает потенциал модели при "идеальном"
                                выборе порога (см. методологическую оговорку
                                в docstring модуля)

    PR-AUC не зависит от порога и вычисляется один раз по y_prob.

    Args:
        y_true     : истинные метки тестовой выборки
        y_prob     : вероятности класса 1 (фрод), нужны для PR-AUC
                     и подбора порога
        model_name : название модели для вывода и сводной таблицы

    Returns:
        Словарь с метриками при обоих порогах (для сводной таблицы).
    """
    pr_auc = average_precision_score(y_true, y_prob)

    # ------------------------------------------------------------------
    # Порог 0.5 (дефолтный)
    # ------------------------------------------------------------------
    y_pred_default = (y_prob >= 0.5).astype(int)
    precision_d = precision_score(y_true, y_pred_default, zero_division=0)
    recall_d    = recall_score(y_true, y_pred_default, zero_division=0)
    f1_d        = f1_score(y_true, y_pred_default, zero_division=0)
    tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_true, y_pred_default).ravel()

    # ------------------------------------------------------------------
    # Оптимальный порог по F1 (подбор по PR-кривой)
    # ------------------------------------------------------------------
    precisions_c, recalls_c, thresholds = precision_recall_curve(y_true, y_prob)
    # precisions_c/recalls_c длиннее thresholds на 1 элемент (sklearn добавляет
    # финальную точку (precision=1, recall=0)) -- отбрасываем её при поиске argmax
    f1_curve = 2 * precisions_c * recalls_c / (precisions_c + recalls_c + 1e-9)
    best_idx = int(np.argmax(f1_curve[:-1])) if len(thresholds) > 0 else 0
    best_thr = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

    y_pred_opt = (y_prob >= best_thr).astype(int)
    precision_o = precision_score(y_true, y_pred_opt, zero_division=0)
    recall_o    = recall_score(y_true, y_pred_opt, zero_division=0)
    f1_o        = f1_score(y_true, y_pred_opt, zero_division=0)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_true, y_pred_opt).ravel()

    # ------------------------------------------------------------------
    # Вывод
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  PR-AUC (не зависит от порога): {pr_auc:.4f}")

    print(f"\n  --- Порог = 0.5000 (дефолтный) ---")
    print(f"  Precision : {precision_d:.4f}")
    print(f"  Recall    : {recall_d:.4f}")
    print(f"  F1-score  : {f1_d:.4f}")
    print(f"  Confusion Matrix: TP={tp_d}  FP={fp_d}  FN={fn_d}  TN={tn_d}")

    print(f"\n  --- Порог = {best_thr:.4f} (оптимальный по F1, диагностика) ---")
    print(f"  Precision : {precision_o:.4f}")
    print(f"  Recall    : {recall_o:.4f}")
    print(f"  F1-score  : {f1_o:.4f}")
    print(f"  Confusion Matrix: TP={tp_o}  FP={fp_o}  FN={fn_o}  TN={tn_o}")

    print(f"\n  Classification report (порог 0.5):")
    print(f"{classification_report(y_true, y_pred_default, zero_division=0)}")

    return {
        "Model":              model_name,
        "PR-AUC":             round(pr_auc, 4),
        "Precision@0.5":      round(precision_d, 4),
        "Recall@0.5":         round(recall_d, 4),
        "F1@0.5":             round(f1_d, 4),
        "Optimal Threshold":  round(best_thr, 4),
        "Precision@opt":      round(precision_o, 4),
        "Recall@opt":         round(recall_o, 4),
        "F1@opt":             round(f1_o, 4),
    }


# =============================================================================
# 3. КРОСС-ВАЛИДАЦИЯ НА TRAIN (TimeSeriesSplit)
# =============================================================================

def cv_report(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    dates_train: np.ndarray = None,
) -> None:
    """
    Запускает TimeSeriesSplit кросс-валидацию на обучающей выборке
    и выводит средние Precision / Recall / F1 / PR-AUC по фолдам.

    Порог классификации в CV намеренно НЕ подбирается (используется 0.5
    через model.predict) -- цель CV здесь наблюдение за стабильностью
    модели во времени, а не финальная оценка качества. Подбор порога
    выполняется только один раз, на тестовой выборке, в compute_metrics.

    TimeSeriesSplit не перемешивает данные: каждый следующий фолд
    использует бо́льшую обучающую часть и валидирует на более поздних данных.
    Это соответствует логике «модель обучается на прошлом, предсказывает будущее».

    Args:
        model        : sklearn-совместимый классификатор (уже инициализированный)
        X_train      : матрица признаков обучающей выборки
        y_train      : метки обучающей выборки
        model_name   : название модели для вывода
        dates_train  : массив дат в том же порядке, что и X_train;
                       используется для отображения временных границ
                       каждого фолда. Если None -- даты не выводятся.
    """
    print(f"\n[CV] {model_name}: TimeSeriesSplit ({CV_FOLDS} фолдов)...")

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    precisions, recalls, f1s, pr_aucs = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # Статистика по фолду перед обучением
        fraud_train = y_tr.sum()
        fraud_val   = y_val.sum()
        print(f"\n  Фолд {fold} | "
              f"train: {len(y_tr):,} записей, фрод: {fraud_train:,} "
              f"({fraud_train / len(y_tr) * 100:.3f}%) | "
              f"val: {len(y_val):,} записей, фрод: {fraud_val:,} "
              f"({fraud_val / len(y_val) * 100:.3f}%)")

        # Временны́е границы фолда (если даты переданы)
        if dates_train is not None:
            dates = pd.Series(dates_train)
            tr_dates  = dates.iloc[tr_idx]
            val_dates = dates.iloc[val_idx]
            print(f"           train: {tr_dates.min().date()} — {tr_dates.max().date()} | "
                  f"val: {val_dates.min().date()} — {val_dates.max().date()}")

        # Пропускаем фолды, где в val нет фрода (PR-AUC не определён)
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
# 4. МОДЕЛИ
# =============================================================================

def run_logistic_regression(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    dates_train: np.ndarray = None,
) -> dict:
    """
    Logistic Regression — линейная baseline-модель.

    Особенности:
        - Требует масштабирования: StandardScaler обучается ТОЛЬКО на train,
          затем применяется к test (предотвращает data leakage).
        - class_weight='balanced': автоматически взвешивает классы обратно
          пропорционально их частоте, компенсируя дисбаланс без SMOTE.
        - solver='lbfgs': быстрее saga на больших данных.
    """
    print("\n[MODEL] Logistic Regression")

    # Масштабирование — обязательно для LR
    scaler      = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # Кросс-валидация на train
    cv_report(model, X_tr_scaled, y_train, "Logistic Regression", dates_train)

    # Финальное обучение на полном train -> оценка на test
    model.fit(X_tr_scaled, y_train)
    y_prob = model.predict_proba(X_te_scaled)[:, 1]

    return compute_metrics(y_test, y_prob, "Logistic Regression")


def run_random_forest(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    dates_train: np.ndarray = None,
) -> dict:
    """
    Random Forest — ансамбль на бэггинге.

    Особенности:
        - class_weight='balanced_subsample': взвешивание классов
          вычисляется отдельно для каждого дерева на его бутстрэп-выборке.
        - n_estimators=100, max_depth=20.
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
    cv_report(model, X_train, y_train, "Random Forest", dates_train)

    # Финальное обучение на полном train -> оценка на test
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_prob, "Random Forest")


def run_xgboost(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    dates_train: np.ndarray = None,
) -> dict:
    """
    XGBoost — ансамбль на градиентном бустинге.

    Особенности:
        - scale_pos_weight: n_legit / n_fraud, вычисляется из train.
        - eval_metric='aucpr': оптимизация PR-AUC при обучении.
        - tree_method='hist': быстрый алгоритм построения деревьев.
        - Масштабирование НЕ нужно.
    """
    print("\n[MODEL] XGBoost")

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
    cv_report(model, X_train, y_train, "XGBoost", dates_train)

    # Финальное обучение на полном train -> оценка на test
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_prob, "XGBoost")


# =============================================================================
# 5. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
# =============================================================================

def print_summary(results: list[dict]) -> pd.DataFrame:
    """
    Формирует и выводит сводную таблицу метрик всех моделей
    (при пороге 0.5 и при оптимальном пороге).
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

    # Извлекаем даты напрямую из df_processed по временно́му условию,
    # пока _date_for_split ещё доступен
    df_processed["_date_for_split"] = pd.to_datetime(df_processed["_date_for_split"])
    train_mask = df_processed["_date_for_split"].dt.year < TEST_CUTOFF_YEAR
    test_mask  = df_processed["_date_for_split"].dt.year >= TEST_CUTOFF_YEAR

    dates_train = df_processed.loc[train_mask, "_date_for_split"].values
    dates_test  = df_processed.loc[test_mask, "_date_for_split"].values

    X_train = X_train_df.values
    X_test  = X_test_df.values
    y_train = y_train.values
    y_test  = y_test.values

    print(f"\n  Признаки ({X_train_df.shape[1]}): {X_train_df.columns.tolist()}")

    if dates_test is not None:
        dates_test_s = pd.Series(pd.to_datetime(dates_test))
        print(f"\n  Тестовая выборка: "
              f"{dates_test_s.min().date()} — {dates_test_s.max().date()} "
              f"({len(dates_test_s):,} записей)")

    # --- Шаг 3: Обучение и оценка ---
    print("\n[STEP 3] Обучение и оценка моделей...")
    results = []
    results.append(run_logistic_regression(X_train, X_test, y_train, y_test, dates_train))
    results.append(run_random_forest(X_train, X_test, y_train, y_test, dates_train))
    results.append(run_xgboost(X_train, X_test, y_train, y_test, dates_train))

    # --- Шаг 4: Сводная таблица ---
    df_summary = print_summary(results)

    # --- Шаг 5: Сохранение результатов ---
    from config import RESULT_DIR
    import os
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = f"{RESULT_DIR}/stage3_results.csv"
    df_summary.to_csv(out_path)
    print(f"\n[INFO] Результаты сохранены: {out_path}")