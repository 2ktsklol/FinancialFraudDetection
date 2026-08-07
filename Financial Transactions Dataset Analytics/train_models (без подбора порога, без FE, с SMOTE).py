"""
train_stage5.py
================
Этап 5: обучение и оценка ML-моделей с балансировкой классов методом SMOTE,
БЕЗ feature engineering (признаковое пространство идентично Этапу 3 /
baseline). Изолированная оценка эффекта SMOTE отдельно от эффекта FE.

Требуется пакет imbalanced-learn:  pip install imbalanced-learn

Ключевые методологические принципы:

1. SMOTE применяется ИСКЛЮЧИТЕЛЬНО к обучающим данным -- никогда к test и
   никогда к val-части CV-фолда. Это касается и кросс-валидации: SMOTE
   выполняется ЗАНОВО внутри КАЖДОГО фолда, только на train-части этого
   конкретного фолда, а не один раз на всём train до разбиения на фолды.
   Если бы SMOTE применялся один раз на весь train до TimeSeriesSplit,
   синтетические объекты в ранних фолдах могли бы быть интерполированы по
   парам реальных объектов, один из которых относится к более позднему
   периоду (SMOTE не учитывает время при поиске ближайших соседей) -- это
   давало бы утечку информации о будущих паттернах мошенничества в
   обучающие данные более ранних фолдов. Тот же принцип, которым мы
   руководствовались при работе с порогом классификации и mcc_risk_group.

2. Масштабирование (StandardScaler) применяется ДО SMOTE для ВСЕХ трёх
   моделей, включая RF и XGBoost, которым само по себе оно не требуется --
   см. обоснование в scale_and_smote().

3. scale_pos_weight для XGBoost (и, автоматически, class_weight='balanced'
   для LR/RF) пересчитывается по факту распределения классов В ДАННЫХ,
   НА КОТОРЫХ модель реально обучается -- то есть уже ПОСЛЕ SMOTE, а не по
   исходному дисбалансу 695:1. Гиперпараметр (сам механизм компенсации
   дисбаланса) остаётся неизменным по сравнению с Этапами 3-4 -- меняется
   только то, каким данным он теперь соответствует. При используемом здесь
   соотношении 10:1 (см. п.4) остаточный дисбаланс всё ещё компенсируется
   этим механизмом, но уже в гораздо более мягкой степени, чем при исходных
   695:1 -- это ожидаемое и корректное следствие частичной балансировки
   данных самим SMOTE, а не ошибка кода. Если считать scale_pos_weight по
   ИСХОДНОМУ (досмоуловскому) дисбалансу и применять его поверх уже
   частично сбалансированных SMOTE данных, получилась бы избыточная
   двойная компенсация дисбаланса.

4. SMOTE настроен на целевое соотношение классов 10:1 (majority:minority),
   а не на полную балансировку 1:1 -- задаётся параметром
   SMOTE_MINORITY_RATIO = 0.1 (доля минорити от мажорити после
   ресэмплинга). При исходном train ~695:1 (10 343 фрод против ~7,19 млн
   легитимных) целевое число фрод-объектов после SMOTE -- около 719 268
   (0,1 x 7 192 682), то есть будет сгенерировано около 708 925
   синтетических объектов, а итоговый train для финальной модели вырастет
   примерно до 7,91 млн записей (против ~14,4 млн при полной балансировке
   1:1) -- рост более умеренный, чем при используемых по умолчанию
   настройках SMOTE, хотя обучение RF и XGBoost (и CV-фолды) всё равно
   будет заметно медленнее, чем на Этапах 3-4.
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

from default_preprocess_data import load_and_merge, preprocess, time_split


# =============================================================================
# 1. КОНФИГУРАЦИЯ (идентична Этапам 3-4)
# =============================================================================

RANDOM_STATE              = 42
CV_FOLDS                  = 5
TEST_CUTOFF_YEAR          = 2018
CLASSIFICATION_THRESHOLD  = 0.5

# Целевое соотношение классов после SMOTE: доля минорити (фрод) от
# мажорити (легит). 0.1 соответствует majority:minority = 10:1
# (а не полной балансировке 1:1, которая была бы при значении 1.0).
SMOTE_MINORITY_RATIO      = 0.1


# =============================================================================
# 2. SMOTE + МАСШТАБИРОВАНИЕ (общий блок для всех трёх моделей)
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
    -- до 10:1), а не до полной балансировки 1:1. ВСЕГДА вызывается только
    на train-данных (полном train для финальной модели или train-части
    конкретного CV-фолда) -- никогда на val/test.

    k_neighbors подбирается динамически (min(5, n_fraud-1)) на случай, если
    в переданных данных окажется мало объектов класса 1 -- SMOTE не может
    построить k соседей, если объектов меньше k+1.

    Если текущее соотношение классов уже "лучше" (ближе к балансу), чем
    целевое sampling_strategy -- например, из-за случайного состава
    небольшого CV-фолда -- SMOTE пропускается: библиотека imbalanced-learn
    может только ДОБАВЛЯТЬ объекты минорити, а не удалять их, поэтому если
    целевое число объектов класса 1 оказывается меньше или равно текущему,
    оверсэмплинг не требуется и не выполняется.
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
    XGBoost, хотя им самим оно не требуется: SMOTE строит синтетические
    объекты по k ближайшим соседям в евклидовом пространстве признаков, а
    признаки датасета имеют кардинально разные шкалы (например, zip --
    порядка 10^4-10^5, is_online -- 0/1). Без масштабирования соседи для
    интерполяции подбирались бы почти исключительно по признакам с большим
    числовым диапазоном, искажая качество синтетических примеров. Само
    масштабирование результат обучения RF/XGBoost не меняет -- деревья
    инвариантны к монотонному преобразованию каждого признака по
    отдельности, поэтому это не влияет на сравнимость с Этапом 3.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_eval_scaled  = scaler.transform(X_eval_raw)

    X_train_final, y_train_final = apply_smote(X_train_scaled, y_train_raw)

    return X_train_final, X_eval_scaled, y_train_final


# =============================================================================
# 3. ВЫЧИСЛЕНИЕ МЕТРИК (порог 0.5, как в Этапах 3-4)
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
    данных ПОСЛЕ SMOTE и возвращающая новый экземпляр модели (нужно, т.к.
    scale_pos_weight у XGBoost пересчитывается по факту данных каждого
    фолда после балансировки -- см. п.3 в docstring модуля).

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
    print("  СВОДНАЯ ТАБЛИЦА — ЭТАП 5 (SMOTE, без FE)")
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

    print("\n[STEP 1] Загрузка и предобработка данных (без FE)...")
    df_raw = load_and_merge(
        TRANSACTIONS_PATH, USERS_PATH, CARDS_PATH, MCC_PATH, LABELS_PATH
    )
    df_processed = preprocess(df_raw)

    X_train_df, X_test_df, y_train, y_test = time_split(
        df_processed, test_cutoff_year=TEST_CUTOFF_YEAR
    )

    df_processed["_date_for_split"] = pd.to_datetime(df_processed["_date_for_split"])
    train_mask = df_processed["_date_for_split"].dt.year < TEST_CUTOFF_YEAR
    test_mask  = df_processed["_date_for_split"].dt.year >= TEST_CUTOFF_YEAR
    dates_train = df_processed.loc[train_mask, "_date_for_split"].values
    dates_test  = df_processed.loc[test_mask,  "_date_for_split"].values

    X_train = X_train_df.values
    X_test  = X_test_df.values
    y_train = y_train.values
    y_test  = y_test.values

    print(f"\n  Признаки ({X_train_df.shape[1]}): {X_train_df.columns.tolist()}")

    dates_test_s = pd.Series(pd.to_datetime(dates_test))
    print(f"\n  Тестовая выборка: "
          f"{dates_test_s.min().date()} — {dates_test_s.max().date()} "
          f"({len(dates_test_s):,} записей)")

    print("\n[STEP 2] Обучение и оценка моделей с SMOTE...")
    results = []
    results.append(run_logistic_regression(X_train, X_test, y_train, y_test, dates_train))
    results.append(run_random_forest(X_train, X_test, y_train, y_test, dates_train))
    results.append(run_xgboost(X_train, X_test, y_train, y_test, dates_train))

    df_summary = print_summary(results)

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = f"{RESULT_DIR}/stage5_results.csv"
    df_summary.to_csv(out_path)
    print(f"\n[INFO] Результаты сохранены: {out_path}")