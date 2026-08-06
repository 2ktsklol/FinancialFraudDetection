"""
preprocess_stage4.py
=====================
Предобработка данных для Этапа 4 эксперимента: feature engineering.

Цель этапа: обучение и тестирование моделей на датасете с добавленными
инженерными признаками, БЕЗ SMOTE. Сравнивается с baseline (Этап 3) для
изолированной оценки влияния feature engineering.

Новые признаки (согласованы по итогам EDA, см. раздел 4.1 работы):
    - hour, day_of_week, is_weekend  -- временные паттерны fraud rate
    - is_online                      -- разрыв fraud rate: 0.84% vs 0.03-0.10%
    - mcc_risk_group                 -- разброс fraud rate по MCC от ~3% до ~60%
    - log1p_amount                   -- скошенное распределение суммы
    - is_small_amount (<$1)          -- пик фрода на малых суммах
    - is_missing_geo                 -- систематические пропуски zip/merchant_state

Ключевое методологическое отличие от простой трансформации:
    mcc_risk_group строится на основе исторического fraud rate по категориям
    mcc -- это, по сути, target encoding. Такой признак НЕЛЬЗЯ считать по
    всему датасету сразу (это была бы утечка информации о fraud rate из
    тестового периода в обучающие признаки). Поэтому построение
    mcc_risk_group вынесено в отдельную функцию add_mcc_risk_group(),
    которая вызывается ПОСЛЕ временного разбиения на train/test и считает
    fraud rate по категориям ИСКЛЮЧИТЕЛЬНО на train -- аналогично тому, как
    StandardScaler обучается на train и затем применяется к test.

    Остальные 6 новых признаков -- детерминированные преобразования
    (дата, сумма, наличие пропуска), утечке не подвержены и вычисляются
    в общей функции preprocess_fe() до разбиения.

Что переиспользуется без изменений из Этапа 3 (default_preprocess_data.py):
    - load_and_merge()  -- загрузка и JOIN исходных файлов
    - time_split()      -- временное разбиение train/test по дате
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from default_preprocess_data import load_and_merge, time_split  # noqa: F401 (реэкспорт для train_stage4.py)


# =============================================================================
# 1. ПРЕДОБРАБОТКА + FEATURE ENGINEERING (детерминированная часть)
# =============================================================================

def preprocess_fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка для Этапа 4: те же обязательные технические шаги, что и
    в Этапе 3 (preprocess() из default_preprocess_data.py), плюс 6 новых
    признаков, не требующих статистики по классам (безопасны от утечки).

    Признак mcc_risk_group здесь НЕ создаётся -- см. add_mcc_risk_group().
    Столбец 'mcc' намеренно НЕ удаляется в этой функции -- он нужен после
    временного разбиения для построения mcc_risk_group.

    Args:
        df : результат load_and_merge() -- объединённый "сырой" датасет

    Returns:
        DataFrame с техническими преобразованиями и новыми FE-признаками,
        готовый к передаче в time_split(). Колонка 'mcc' присутствует.
    """
    print("\n[STEP 2] Предобработка + Feature Engineering...")
    df = df.copy()

    # ------------------------------------------------------------------
    # 2.1 Дата: числовые компоненты (как в Этапе 3) + FE-признаки времени
    # ------------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["_date_for_split"] = df["date"]

    df["tx_year"]  = df["date"].dt.year
    df["tx_month"] = df["date"].dt.month
    df["tx_day"]   = df["date"].dt.day

    # --- FE: временные паттерны (см. EDA, п. 5) ---
    df["hour"]        = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek       # 0 = Пн ... 6 = Вс
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    df.drop(columns=["date"], inplace=True)

    # ------------------------------------------------------------------
    # 2.2 Денежные столбцы: $ -> float (как в Этапе 3)
    # ------------------------------------------------------------------
    money_cols = [
        "amount", "user_per_capita_income", "user_yearly_income",
        "user_total_debt", "card_credit_limit",
    ]
    for col in money_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .replace("nan", np.nan)
                .astype(float)
            )

    # --- FE: сумма транзакции (см. EDA, п. 7) ---
    df["log1p_amount"]    = np.log1p(df["amount"].clip(lower=0))
    df["is_small_amount"] = (df["amount"] < 1.0).astype(int)

    # ------------------------------------------------------------------
    # 2.3 Даты карты -> числовые признаки (как в Этапе 3)
    # ------------------------------------------------------------------
    if "card_expires" in df.columns:
        df["card_expires_year"] = (
            pd.to_datetime(df["card_expires"], format="%m/%Y", errors="coerce").dt.year
        )
        df.drop(columns=["card_expires"], inplace=True)

    if "card_acct_open_date" in df.columns:
        df["card_acct_open_year"] = (
            pd.to_datetime(df["card_acct_open_date"], format="%m/%Y", errors="coerce").dt.year
        )
        df.drop(columns=["card_acct_open_date"], inplace=True)

    # ------------------------------------------------------------------
    # 2.4 FE: геолокация -- флаг пропуска (см. EDA, п. 2)
    # ------------------------------------------------------------------
    # Строится ДО заполнения пропусков zip/merchant_state.
    df["is_missing_geo"] = (df["zip"].isna() | df["merchant_state"].isna()).astype(int)

    # ------------------------------------------------------------------
    # 2.5 FE: способ оплаты -- is_online (см. EDA, п. 3)
    # ------------------------------------------------------------------
    # Строится ДО Label Encoding use_chip, по сырым строковым значениям.
    df["is_online"] = (df["use_chip"] == "Online Transaction").astype(int)

    # ------------------------------------------------------------------
    # 2.6 Label Encoding категориальных признаков (как в Этапе 3)
    # ------------------------------------------------------------------
    cat_cols = [
        "use_chip", "merchant_state", "user_gender",
        "card_card_brand", "card_card_type", "card_has_chip",
        "card_card_on_dark_web",
    ]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
            df[col] = le.fit_transform(df[col])

    # ------------------------------------------------------------------
    # 2.7 Заполнение пропусков (как в Этапе 3)
    # ------------------------------------------------------------------
    if "zip" in df.columns:
        df["zip"] = df["zip"].fillna(0).astype(int)

    num_user_cols = [
        "user_per_capita_income", "user_yearly_income",
        "user_total_debt", "card_credit_limit",
        "card_expires_year", "card_acct_open_year",
    ]
    for col in num_user_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ------------------------------------------------------------------
    # 2.8 Удаление нерелевантных столбцов (как в Этапе 3)
    # ------------------------------------------------------------------
    # ВАЖНО: 'mcc' здесь НЕ удаляется -- нужен для add_mcc_risk_group()
    # после временного разбиения.
    drop_cols = [
        "id", "client_id", "card_id", "merchant_id", "card_client_id",
        "merchant_city", "mcc_description", "user_address",
        "card_card_number", "card_cvv", "errors",
        "user_birth_year", "user_birth_month",
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing_drop, inplace=True)

    print(f"  [FE] Добавлены признаки: hour, day_of_week, is_weekend, "
          f"is_online, log1p_amount, is_small_amount, is_missing_geo")
    print(f"  [FE] 'mcc' сохранён для построения mcc_risk_group после сплита")
    print(f"  Признаки перед временным разбиением ({df.shape[1]}): "
          f"{df.columns.tolist()}")

    return df


# =============================================================================
# 2. FEATURE ENGINEERING, ТРЕБУЮЩИЙ FIT НА TRAIN: mcc_risk_group
# =============================================================================

def add_mcc_risk_group(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    n_groups: int = 3,
) -> tuple:
    """
    Строит признак mcc_risk_group на основе исторического fraud rate по
    категориям мерчанта (mcc). Fraud rate считается ИСКЛЮЧИТЕЛЬНО на train
    -- аналогично StandardScaler.fit(train)/.transform(test) -- чтобы
    информация о распределении фрода в test не попала в признак.

    Категории mcc из train делятся на n_groups групп риска по квантилям
    среднего fraud rate внутри категории (0 = низкий риск,
    n_groups-1 = высокий риск). Категории mcc, встретившиеся только в test
    (отсутствовавшие в train), получают "среднюю" (fallback) группу риска --
    единственный безопасный вариант для ранее не встречавшейся категории.

    Args:
        X_train, X_test : DataFrame с колонкой 'mcc' (числовой код)
        y_train          : целевая переменная train (для расчёта fraud rate)
        n_groups         : количество групп риска (по умолчанию 3)

    Returns:
        (X_train, X_test) с добавленной колонкой 'mcc_risk_group' и
        удалённой исходной 'mcc'.
    """
    print(f"\n[STEP 3] Построение mcc_risk_group по train ({n_groups} группы)...")

    y_train_arr = np.asarray(y_train)
    fraud_rate_by_mcc = (
        pd.Series(y_train_arr, index=X_train.index)
        .groupby(X_train["mcc"].values)
        .mean()
    )

    # pd.qcut на вырожденном распределении (мало уникальных значений
    # fraud rate) может либо бросить ValueError, либо -- что менее
    # очевидно -- молча вернуть NaN для части/всех категорий при
    # duplicates="drop". Оба случая обрабатываются одинаково: переход
    # на ранговое разбиение.
    try:
        mcc_group_labels = pd.qcut(
            fraud_rate_by_mcc, q=n_groups, labels=False, duplicates="drop"
        )
    except ValueError:
        mcc_group_labels = None

    if mcc_group_labels is None or mcc_group_labels.isna().any():
        mcc_group_labels = pd.Series(
            pd.factorize(fraud_rate_by_mcc.rank(method="dense"))[0],
            index=fraud_rate_by_mcc.index,
        )

    mcc_to_group   = mcc_group_labels.to_dict()
    fallback_group = int(np.median(list(mcc_to_group.values())))

    print(f"  Построено групп риска: {len(set(mcc_to_group.values()))}, "
          f"fallback для новых категорий mcc: {fallback_group}")

    for name, X in [("train", X_train), ("test", X_test)]:
        n_unseen = (~X["mcc"].isin(mcc_to_group.keys())).sum()
        if n_unseen > 0:
            print(f"  [{name}] Записей с mcc, не встречавшимся в train: "
                  f"{n_unseen:,} -> назначена fallback-группа {fallback_group}")

    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train["mcc_risk_group"] = (
        X_train["mcc"].map(mcc_to_group).fillna(fallback_group).astype(int)
    )
    X_test["mcc_risk_group"] = (
        X_test["mcc"].map(mcc_to_group).fillna(fallback_group).astype(int)
    )

    X_train.drop(columns=["mcc"], inplace=True)
    X_test.drop(columns=["mcc"], inplace=True)

    return X_train, X_test


# =============================================================================
# ТОЧКА ВХОДА (для автономного запуска / отладки препроцессинга)
# =============================================================================

if __name__ == "__main__":
    from config import (
        TRANSACTIONS_PATH, USERS_PATH, CARDS_PATH, MCC_PATH, LABELS_PATH,
        RESULT_DIR,
    )
    import os

    df_raw = load_and_merge(
        TRANSACTIONS_PATH, USERS_PATH, CARDS_PATH, MCC_PATH, LABELS_PATH
    )
    df_fe = preprocess_fe(df_raw)

    X_train_df, X_test_df, y_train, y_test = time_split(df_fe, test_cutoff_year=2018)
    X_train_df, X_test_df = add_mcc_risk_group(X_train_df, X_test_df, y_train)

    print("\n--- Финальные признаки (train) ---")
    print(f"Количество признаков: {X_train_df.shape[1]}")
    print(X_train_df.columns.tolist())
    print("\n--- Первые строки ---")
    print(X_train_df.head(3).T)

    # ------------------------------------------------------------------
    # Сохранение предобработанного датасета (train/test отдельно, каждый
    # с целевой переменной is_fraud) -- пригодится для проверки признаков
    # руками и для переиспользования без повторного запуска всего пайплайна.
    # CSV, а не XLSX: датасет (train+test) в сумме -- более 8.9 млн строк,
    # что превышает лимит Excel на строки в одном листе (~1 048 576).
    # ------------------------------------------------------------------
    os.makedirs(RESULT_DIR, exist_ok=True)

    train_out = X_train_df.copy()
    train_out["is_fraud"] = y_train.values

    test_out = X_test_df.copy()
    test_out["is_fraud"] = y_test.values

    train_path = f"{RESULT_DIR}/stage4_fe_train_data.csv"
    test_path  = f"{RESULT_DIR}/stage4_fe_test_data.csv"

    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"\n[INFO] Train с новыми признаками сохранён: {train_path} "
          f"({train_out.shape[0]:,} строк, {train_out.shape[1]} столбцов)")
    print(f"[INFO] Test с новыми признаками сохранён: {test_path} "
          f"({test_out.shape[0]:,} строк, {test_out.shape[1]} столбцов)")