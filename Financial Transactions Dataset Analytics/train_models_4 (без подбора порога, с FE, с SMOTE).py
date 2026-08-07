"""
train_stage6.py
================
Этап 6: совместный эффект feature engineering И SMOTE (заключительный
эксперимент факторного дизайна 2х2: FE x SMOTE).

Признаковое пространство -- идентично Этапу 4 (preprocess_stage4.py:
preprocess_fe + add_mcc_risk_group). Балансировка классов -- идентична
Этапу 5 (SMOTE, целевое соотношение 10:1, применяется только к train,
отдельно внутри каждого CV-фолда, scale_pos_weight пересчитывается по
факту данных ПОСЛЕ SMOTE). Гиперпараметры моделей и порог классификации
(0.5) -- зафиксированы такими же, как во всех предыдущих этапах.

Единственное, что меняется по сравнению с Этапами 3-5 -- одновременное
наличие обоих факторов (FE и SMOTE), поэтому в этом файле не появляется
никакой новой логики -- он собирает вместе уже проверенные компоненты из
preprocess_stage4.py (FE) и train_stage5.py (SMOTE).

ВАЖНО (унаследованная от Этапа 4 методологическая особенность, стоит знать
при интерпретации CV-вывода): mcc_risk_group строится ОДИН РАЗ, по всему
train целиком (после временного разбиения train/test, но ДО разбиения
train на CV-фолды) -- см. add_mcc_risk_group() в preprocess_stage4.py. Это
означает, что при просмотре CV по фолдам train-часть раннего фолда (например,
2010-2011 год) содержит значения mcc_risk_group, вычисленные по fraud rate,
усреднённому по ВСЕМУ train (2010-2017), включая более поздние годы --
то есть частично использует информацию "из будущего" относительно этого
конкретного фолда. Эта особенность уже присутствовала в Этапе 4 (её там не
меняли) -- в этом файле она сохранена намеренно, для методологической
согласованности между Этапами 4 и 6. Она НЕ влияет на итоговые метрики на
test (mcc_risk_group для test построен корректно -- по всему train, что
легитимно, т.к. test полностью в будущем относительно всего train) --
затрагивает только числа, выводимые в CV-диагностике по фолдам, которая
и так используется исключительно как описательная иллюстрация стабильности
модели во времени, а не для принятия каких-либо решений.
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
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from fe_preprocess_data import load_and_merge, time_split, preprocess_fe, add_mcc_risk_group


# =============================================================================
# 1. КОНФИГУРАЦИЯ (идентична Этапам 3-5)
# =============================================================================

RANDOM_STATE              = 42
CV_FOLDS                  = 5
TEST_CUTOFF_YEAR          = 2018
CLASSIFICATION_THRESHOLD  = 0.5

# Целевое соотношение классов после SMOTE: majority:minority = 10:1
# (см. обоснование в train_stage5.py)
SMOTE_MINORITY_RATIO      = 0.1


# =============================================================================
# 2. SMOTE + МАСШТАБИРОВАНИЕ (идентично train_stage5.py)
# =============================================================================

def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_STATE,
    sampling_strategy: float = SMOTE_MINORITY_RATIO,
):
    """
    Применяет SMOTE к переданным данным, доводя соотношение классов до
    majority:minority = 1/sampling_strategy : 1 (при sampling_strategy=0.1
    -- до 10:1). ВСЕГДА вызывается только на train-данных (полном train
    для финальной модели или train-части конкретного CV-фолда) -- никогда
    на val/test.

    k_neighbors подбирается динамически (min(5, n_fraud-1)) на случай, если
    в переданных данных окажется мало объектов класса 1.

    Если текущее соотношение классов уже "лучше" целевого -- SMOTE
    пропускается (imbalanced-learn может только добавлять объекты
    минорити, а не удалять их).
    """
    n_fraud = int((y == 1).sum())
    n_legit = int((y == 0).sum())

    if n_fraud < 2:
        print(f"    [SMOTE] Объектов класса 1: {n_fraud} -- недостаточно "
              f"для построения соседей, SMOTE пропущен")
        return X, y

    target_fraud = int(round(sampling_strategy * n_legit))
    if target_fraud <= n_fraud:
        print(f"    [SMOTE] Текущее соотношение ({n_fraud}/{n_legit}) уже "
              f"не хуже целевого ({sampling_strategy}) -- SMOTE пропущен")
        return X, y

    k = min(5, n_fraud - 1)
    smote = SMOTE(
        random_state=random_state, k_neighbors=k,
        sampling_strategy=sampling_strategy,
    )
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def scale_and_smote(X_train_raw, X_eval_raw, y_train_raw):
    """
    Масштабирует признаки (StandardScaler, обучается ТОЛЬКО на train) и
    применяет SMOTE к масштабированному train. X_eval_raw (val или test)
    масштабируется тем же scaler, но SMOTE к нему не применяется никогда.

    Масштабирование выполняется до SMOTE для ВСЕХ моделей, включая RF и
    XGBoost -- см. подробное обоснование в train_stage5.py.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_eval_scaled  = scaler.transform(X_eval_raw)

    X_train_final, y_train_final = apply_smote(X_train_scaled, y_train_raw)

    return X_train_final, X_eval_scaled, y_train_final


# =============================================================================
# 3. ВЫЧИСЛЕНИЕ МЕТРИК (порог 0.5, как во всех предыдущих этапах)
# =============================================================================

def compute_metrics(y_true, y_pred, y_prob, model_name: str) -> dict:
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
# 4. КРОСС-ВАЛИДАЦИЯ С SMOTE ВНУТРИ КАЖДОГО ФОЛДА
# =============================================================================

def cv_report(model_factory, X_train, y_train, model_name, dates_train=None):
    """
    TimeSeriesSplit CV. model_factory -- функция, принимающая y обучающих
    данных ПОСЛЕ SMOTE и возвращающая новый экземпляр модели (нужно для
    динамического пересчёта scale_pos_weight у XGBoost -- см. run_xgboost).

    SMOTE применяется заново на train-части каждого фолда (после её
    масштабирования), val-часть остаётся нетронутой реальной выборкой.
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

        X_tr_final, X_val_scaled, y_tr_final = scale_and_smote(X_tr, X_val, y_tr)
        print(f"    [SMOTE] train фолда: {len(y_tr):,} -> {len(y_tr_final):,} записей "
              f"после балансировки ({int((y_tr_final == 1).sum()):,} фрод)")

        model = model_factory(y_tr_final)
        model.fit(X_tr_final, y_tr_final)
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

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
# 5. МОДЕЛИ
# =============================================================================

def run_logistic_regression(X_train, X_test, y_train, y_test, dates_train=None) -> dict:
    print("\n[MODEL] Logistic Regression")

    def make_model(y_for_weight):
        return LogisticRegression(
            solver="lbfgs", max_iter=1000, tol=1e-3,
            class_weight="balanced", random_state=RANDOM_STATE,
        )

    cv_report(make_model, X_train, y_train, "Logistic Regression", dates_train)

    X_train_final, X_test_scaled, y_train_final = scale_and_smote(X_train, X_test, y_train)
    print(f"\n  [SMOTE] Финальный train: {len(y_train):,} -> {len(y_train_final):,} "
          f"записей после балансировки ({int((y_train_final == 1).sum()):,} фрод)")

    model = make_model(y_train_final)
    model.fit(X_train_final, y_train_final)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= CLASSIFICATION_THRESHOLD).astype(int)

    return compute_metrics(y_test, y_pred, y_prob, "Logistic Regression")


def run_random_forest(X_train, X_test, y_train, y_test, dates_train=None) -> dict:
    print("\n[MODEL] Random Forest")

    def make_model(y_for_weight):
        return RandomForestClassifier(
            n_estimators=100, max_depth=20,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1,
        )

    cv_report(make_model, X_train, y_train, "Random Forest", dates_train)

    X_train_final, X_test_scaled, y_train_final = scale_and_smote(X_train, X_test, y_train)
    print(f"\n  [SMOTE] Финальный train: {len(y_train):,} -> {len(y_train_final):,} "
          f"записей после балансировки ({int((y_train_final == 1).sum()):,} фрод)")

    model = make_model(y_train_final)
    model.fit(X_train_final, y_train_final)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= CLASSIFICATION_THRESHOLD).astype(int)

    return compute_metrics(y_test, y_pred, y_prob, "Random Forest")


def run_xgboost(X_train, X_test, y_train, y_test, dates_train=None) -> dict:
    print("\n[MODEL] XGBoost")

    def make_model(y_for_weight):
        n_legit = int((y_for_weight == 0).sum())
        n_fraud = int((y_for_weight == 1).sum())
        spw = n_legit / n_fraud if n_fraud > 0 else 1.0
        print(f"    scale_pos_weight = {spw:.2f} "
              f"(легит/фрод в данных обучения модели = {n_legit}/{n_fraud})")
        return XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )

    cv_report(make_model, X_train, y_train, "XGBoost", dates_train)

    X_train_final, X_test_scaled, y_train_final = scale_and_smote(X_train, X_test, y_train)
    print(f"\n  [SMOTE] Финальный train: {len(y_train):,} -> {len(y_train_final):,} "
          f"записей после балансировки ({int((y_train_final == 1).sum()):,} фрод)")

    model = make_model(y_train_final)
    model.fit(X_train_final, y_train_final)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= CLASSIFICATION_THRESHOLD).astype(int)

    return compute_metrics(y_test, y_pred, y_prob, "XGBoost")


# =============================================================================
# 6. СВОДНАЯ ТАБЛИЦА
# =============================================================================

def print_summary(results: list) -> pd.DataFrame:
    df_results = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 60)
    print("  СВОДНАЯ ТАБЛИЦА — ЭТАП 6 (FE + SMOTE)")
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

    # --- Шаг 1: Загрузка, объединение, предобработка + FE (как в Этапе 4) ---
    print("\n[STEP 1] Загрузка данных...")
    df_raw = load_and_merge(
        TRANSACTIONS_PATH, USERS_PATH, CARDS_PATH, MCC_PATH, LABELS_PATH
    )
    df_fe = preprocess_fe(df_raw)

    # --- Шаг 2: Временное разбиение ---
    X_train_df, X_test_df, y_train, y_test = time_split(
        df_fe, test_cutoff_year=TEST_CUTOFF_YEAR
    )

    # --- Шаг 3: mcc_risk_group -- строится ПОСЛЕ разбиения, только по train ---
    X_train_df, X_test_df = add_mcc_risk_group(X_train_df, X_test_df, y_train)

    # Извлекаем даты для отображения границ фолдов CV (как в Этапах 3-5)
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

    # --- Шаг 4: Обучение и оценка моделей с SMOTE ---
    print("\n[STEP 4] Обучение и оценка моделей (FE + SMOTE)...")
    results = []
    results.append(run_logistic_regression(X_train, X_test, y_train, y_test, dates_train))
    results.append(run_random_forest(X_train, X_test, y_train, y_test, dates_train))
    results.append(run_xgboost(X_train, X_test, y_train, y_test, dates_train))

    # --- Шаг 5: Сводная таблица ---
    df_summary = print_summary(results)

    # --- Шаг 6: Сохранение результатов ---
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = f"{RESULT_DIR}/stage6_results.csv"
    df_summary.to_csv(out_path)
    print(f"\n[INFO] Результаты сохранены: {out_path}")