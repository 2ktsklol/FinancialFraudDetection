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

Подбор порога классификации (методологически честная версия):
    Метрики Precision/Recall/F1 при пороге 0.5 сильно вводят в заблуждение
    при экстремальном дисбалансе классов (см. baseline: RF Precision=1.0
    при Recall=0.02, несмотря на PR-AUC=0.56 -- модель хорошая, порог плохой).

    Порог подбирается НЕ на тестовой выборке (это было бы утечкой -- порог
    "подсматривал" бы в метки, по которым его же и оценивают), а на
    предсказаниях ПОСЛЕДНЕГО фолда TimeSeriesSplit кросс-валидации на train:
        1. Train разбивается на 5 фолдов TimeSeriesSplit.
        2. На каждом фолде модель обучается на "прошлом" и предсказывает
           вероятности на "будущем" (val), которое не видела при обучении.
        3. Для подбора порога используются предсказания ТОЛЬКО последнего
           (5-го) фолда -- он обучен на наибольшем объёме train (~6.0 млн
           записей из 7.2 млн), то есть максимально близок по объёму
           обучающих данных к финальной модели (обучается на всех 7.2 млн).
           Более ранние фолды (1-4) обучены на существенно меньшем объёме
           и хуже откалиброваны -- их включение в подбор порога исказило бы
           результат в сторону поведения слабых моделей (см. обсуждение
           ниже "Почему не агрегированный OOF по всем фолдам").
        4. На предсказаниях последнего фолда строится PR-кривая и
           подбирается порог, максимизирующий F1.
        5. Модель заново обучается на всём train, и уже готовый
           зафиксированный порог применяется к test -- тест НЕ участвует
           в подборе порога, только в финальной оценке.

    Почему не агрегированный OOF по всем фолдам (предыдущая версия):
        Более ранняя версия этого кода агрегировала предсказания со ВСЕХ
        5 фолдов. На практике это дало пороги, кардинально завышенные
        относительно того, что подошло бы финальной модели (например,
        0.97 для XGBoost) -- потому что итоговый оптимум подстраивался
        под шумные, хуже откалиброванные модели ранних фолдов (обученных
        на 1.2-3.6 млн записей), а не под финальную модель (7.2 млн).
        Использование только последнего фолда устраняет эту проблему,
        сохраняя главное требование -- порог не подбирается на тесте.

    Порог для каждой модели подбирается один раз -- на baseline-данных
    Этапа 3 -- и фиксируется. В Этапах 4-6 (FE, SMOTE, FE+SMOTE) для той
    же модели используется ТОТ ЖЕ порог (сохраняется в results/optimal_
    thresholds.json). Это сделано намеренно: если пересчитывать порог на
    каждом этапе заново, изменения Precision/Recall/F1 между этапами будут
    складываться из двух эффектов сразу -- собственно FE/SMOTE и смещения
    порога -- и их нельзя будет разделить. Фиксация порога изолирует
    влияние именно FE и SMOTE, что и является целью исследования.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
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

RANDOM_STATE       = 42    # фиксируем seed для воспроизводимости
CV_FOLDS           = 5     # количество фолдов TimeSeriesSplit
TEST_CUTOFF_YEAR   = 2018  # граница train/test по времени
THRESHOLDS_PATH    = "results/optimal_thresholds.json"  # куда сохраняем/откуда читаем пороги


# =============================================================================
# 2. ПОДБОР ПОРОГА ПО ПРЕДСКАЗАНИЯМ ПОСЛЕДНЕГО ФОЛДА
# =============================================================================

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Находит порог классификации, максимизирующий F1, по PR-кривой.

    Args:
        y_true : истинные метки последнего фолда кросс-валидации
        y_prob : предсказанные вероятности класса 1 последнего фолда

    Returns:
        Оптимальный порог. Если данных недостаточно для расчёта --
        возвращает 0.5 (дефолт) с предупреждением.
    """
    if len(y_true) == 0 or y_true.sum() == 0:
        print("  [WARN] Недостаточно данных с фродом в последнем фолде для "
              "подбора порога, используется 0.5")
        return 0.5

    precisions_c, recalls_c, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    # precisions_c/recalls_c длиннее thresholds на 1 (sklearn добавляет
    # финальную точку precision=1, recall=0, без соответствующего порога)
    f1_curve = 2 * precisions_c * recalls_c / (precisions_c + recalls_c + 1e-9)
    best_idx = int(np.argmax(f1_curve[:-1]))
    return float(thresholds[best_idx])


def load_or_init_thresholds(path: str = THRESHOLDS_PATH) -> dict:
    """Загружает ранее сохранённые пороги, если файл существует."""
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}


def save_thresholds(thresholds: dict, path: str = THRESHOLDS_PATH) -> None:
    """Сохраняет пороги в JSON -- используются на Этапах 4-6."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(thresholds, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Пороги сохранены: {path}")


# =============================================================================
# 3. ВЫЧИСЛЕНИЕ МЕТРИК (порог 0.5 + зафиксированный CV-порог)
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    cv_threshold: float,
) -> dict:
    """
    Вычисляет и печатает метрики на test при двух порогах:

        1) 0.5           -- дефолтный порог (справочно)
        2) cv_threshold  -- порог, подобранный по предсказаниям последнего
                             (наибольшего по объёму) фолда кросс-валидации
                             на train (см. docstring модуля). Именно эти
                             метрики -- основные для сравнения моделей и
                             этапов исследования.

    PR-AUC не зависит от порога и вычисляется один раз.

    Args:
        y_true       : истинные метки тестовой выборки
        y_prob       : вероятности класса 1 на тесте
        model_name   : название модели
        cv_threshold : зафиксированный порог, подобранный на CV/train

    Returns:
        Словарь с метриками при обоих порогах (для сводной таблицы).
    """
    pr_auc = average_precision_score(y_true, y_prob)

    def _metrics_at(threshold: float) -> tuple:
        y_pred = (y_prob >= threshold).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return p, r, f, (tp, fp, fn, tn), y_pred

    precision_d, recall_d, f1_d, cm_d, y_pred_d = _metrics_at(0.5)
    precision_c, recall_c, f1_c, cm_c, y_pred_c = _metrics_at(cv_threshold)

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  PR-AUC (не зависит от порога): {pr_auc:.4f}")

    print(f"\n  --- Порог = 0.5000 (дефолтный, справочно) ---")
    print(f"  Precision : {precision_d:.4f}")
    print(f"  Recall    : {recall_d:.4f}")
    print(f"  F1-score  : {f1_d:.4f}")
    print(f"  Confusion Matrix: TP={cm_d[0]}  FP={cm_d[1]}  FN={cm_d[2]}  TN={cm_d[3]}")

    print(f"\n  --- Порог = {cv_threshold:.4f} (подобран на CV/train, основной) ---")
    print(f"  Precision : {precision_c:.4f}")
    print(f"  Recall    : {recall_c:.4f}")
    print(f"  F1-score  : {f1_c:.4f}")
    print(f"  Confusion Matrix: TP={cm_c[0]}  FP={cm_c[1]}  FN={cm_c[2]}  TN={cm_c[3]}")

    print(f"\n  Classification report (порог {cv_threshold:.4f}):")
    print(f"{classification_report(y_true, y_pred_c, zero_division=0)}")

    return {
        "Model":            model_name,
        "PR-AUC":           round(pr_auc, 4),
        "Precision@0.5":    round(precision_d, 4),
        "Recall@0.5":       round(recall_d, 4),
        "F1@0.5":           round(f1_d, 4),
        "CV Threshold":     round(cv_threshold, 4),
        "Precision@CV_thr": round(precision_c, 4),
        "Recall@CV_thr":    round(recall_c, 4),
        "F1@CV_thr":        round(f1_c, 4),
    }


# =============================================================================
# 4. КРОСС-ВАЛИДАЦИЯ НА TRAIN (TimeSeriesSplit) + ПРЕДСКАЗАНИЯ ПОСЛЕДНЕГО ФОЛДА
# =============================================================================

def cv_report(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    dates_train: np.ndarray = None,
) -> tuple:
    """
    Запускает TimeSeriesSplit кросс-валидацию на обучающей выборке,
    выводит метрики по фолдам (при пороге 0.5, справочно) и отдельно
    сохраняет предсказания ПОСЛЕДНЕГО (5-го) фолда -- они используются
    дальше для подбора порога по F1 (см. обоснование в docstring модуля).

    Первый тренировочный блок (данные до первого val-фолда) никогда не
    попадает в val ни на одном фолде TimeSeriesSplit -- это особенность
    самого TimeSeriesSplit, а не ошибка кода.

    Args:
        model        : sklearn-совместимый классификатор (уже инициализированный)
        X_train      : матрица признаков обучающей выборки
        y_train      : метки обучающей выборки
        model_name   : название модели для вывода
        dates_train  : массив дат в том же порядке, что и X_train;
                       используется для отображения временных границ
                       каждого фолда. Если None -- даты не выводятся.

    Returns:
        (last_fold_true, last_fold_prob) -- истинные метки и предсказанные
        вероятности последнего (наибольшего по объёму) фолда, используются
        для подбора порога классификации.
    """
    print(f"\n[CV] {model_name}: TimeSeriesSplit ({CV_FOLDS} фолдов)...")

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

    precisions, recalls, f1s, pr_aucs = [], [], [], []
    last_fold_true, last_fold_prob = np.array([]), np.array([])

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

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Сохраняем предсказания последнего фолда для подбора порога.
        # На каждой итерации перезаписываем -- после цикла в переменных
        # останутся данные именно 5-го (последнего) фолда.
        last_fold_true, last_fold_prob = y_val, y_prob

        if fraud_val == 0:
            print(f"  Фолд {fold}: нет фрода в val — метрики не показательны, "
                  f"пропущены")
            continue

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

    print(f"\n  [INFO] Для подбора порога используются предсказания "
          f"последнего фолда ({len(last_fold_true):,} записей, "
          f"фрод: {int(last_fold_true.sum()):,})")

    return last_fold_true, last_fold_prob


# =============================================================================
# 5. МОДЕЛИ
# =============================================================================

def run_logistic_regression(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    dates_train: np.ndarray = None,
    fixed_threshold: float = None,
) -> dict:
    """
    Logistic Regression — линейная baseline-модель.

    Args:
        fixed_threshold : если передан (из results/optimal_thresholds.json
                           на Этапах 4-6) -- используется вместо подбора
                           нового порога по CV. На Этапе 3 всегда None --
                           порог подбирается впервые.
    """
    print("\n[MODEL] Logistic Regression")

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

    last_fold_true, last_fold_prob = cv_report(model, X_tr_scaled, y_train, "Logistic Regression", dates_train)

    if fixed_threshold is not None:
        threshold = fixed_threshold
        print(f"\n  [INFO] Используется зафиксированный ранее порог: {threshold:.4f}")
    else:
        threshold = find_optimal_threshold(last_fold_true, last_fold_prob)
        print(f"\n  [INFO] Порог подобран по последнему фолду CV: {threshold:.4f}")

    model.fit(X_tr_scaled, y_train)
    y_prob = model.predict_proba(X_te_scaled)[:, 1]

    return compute_metrics(y_test, y_prob, "Logistic Regression", threshold)


def run_random_forest(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    dates_train: np.ndarray = None,
    fixed_threshold: float = None,
) -> dict:
    """Random Forest — ансамбль на бэггинге."""
    print("\n[MODEL] Random Forest")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    last_fold_true, last_fold_prob = cv_report(model, X_train, y_train, "Random Forest", dates_train)

    if fixed_threshold is not None:
        threshold = fixed_threshold
        print(f"\n  [INFO] Используется зафиксированный ранее порог: {threshold:.4f}")
    else:
        threshold = find_optimal_threshold(last_fold_true, last_fold_prob)
        print(f"\n  [INFO] Порог подобран по последнему фолду CV: {threshold:.4f}")

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_prob, "Random Forest", threshold)


def run_xgboost(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    dates_train: np.ndarray = None,
    fixed_threshold: float = None,
) -> dict:
    """XGBoost — ансамбль на градиентном бустинге."""
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

    last_fold_true, last_fold_prob = cv_report(model, X_train, y_train, "XGBoost", dates_train)

    if fixed_threshold is not None:
        threshold = fixed_threshold
        print(f"\n  [INFO] Используется зафиксированный ранее порог: {threshold:.4f}")
    else:
        threshold = find_optimal_threshold(last_fold_true, last_fold_prob)
        print(f"\n  [INFO] Порог подобран по последнему фолду CV: {threshold:.4f}")

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    return compute_metrics(y_test, y_prob, "XGBoost", threshold)


# =============================================================================
# 6. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
# =============================================================================

def print_summary(results: list[dict]) -> pd.DataFrame:
    """Формирует и выводит сводную таблицу метрик всех моделей."""
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
        MCC_PATH, LABELS_PATH, RESULT_DIR,
    )
    import os

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

    # --- Шаг 3: Загружаем ранее сохранённые пороги (если есть) ---
    # На Этапе 3 файла ещё нет -- пороги подберутся впервые и сохранятся.
    # На Этапах 4-6 запусти этот же скрипт после смены STAGE_NAME на
    # соответствующий -- пороги подхватятся автоматически из json и
    # ПОДБИРАТЬСЯ ЗАНОВО НЕ БУДУТ.
    saved_thresholds = load_or_init_thresholds()
    if saved_thresholds:
        print(f"\n[INFO] Найдены сохранённые пороги: {saved_thresholds}")
        print("[INFO] Они будут использованы вместо повторного подбора.")

    # --- Шаг 4: Обучение и оценка ---
    print("\n[STEP 4] Обучение и оценка моделей...")
    results = []
    results.append(run_logistic_regression(
        X_train, X_test, y_train, y_test, dates_train,
        fixed_threshold=saved_thresholds.get("Logistic Regression"),
    ))
    results.append(run_random_forest(
        X_train, X_test, y_train, y_test, dates_train,
        fixed_threshold=saved_thresholds.get("Random Forest"),
    ))
    results.append(run_xgboost(
        X_train, X_test, y_train, y_test, dates_train,
        fixed_threshold=saved_thresholds.get("XGBoost"),
    ))

    # --- Шаг 5: Сводная таблица ---
    df_summary = print_summary(results)

    # --- Шаг 6: Сохранение результатов и порогов (порогов -- только если
    #             ещё не были сохранены ранее, т.е. это первый прогон/Этап 3) ---
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = f"{RESULT_DIR}/stage3_results.csv"
    df_summary.to_csv(out_path)
    print(f"\n[INFO] Результаты сохранены: {out_path}")

    if not saved_thresholds:
        thresholds_to_save = {r["Model"]: r["CV Threshold"] for r in results}
        save_thresholds(thresholds_to_save)