"""
preprocess_stage3.py
====================
Предобработка данных для Этапа 3 эксперимента.

Цель этапа: обучение и тестирование моделей на «чистом» датасете —
без feature engineering и без SMOTE.

Источники данных:
    - transactions_data.csv
    - users_data.csv
    - cards_data.csv
    - mcc_codes.json
    - train_fraud_labels.json

Что делается (минимально необходимый preprocessing):
    1. Объединение таблиц (transactions + users + cards + labels)
    2. Очистка денежных столбцов от символа $ и запятых -> float
    3. Разбор datetime на числовые компоненты (year, month, day)
       !! Без hour, dayofweek — это уже feature engineering (Этап 4)
    4. Label Encoding категориальных признаков с низкой/средней кардинальностью
    5. Заполнение пропущенных значений
    6. Исключение нерелевантных столбцов (идентификаторы, высококардинальные строки,
       столбцы с >95% пропусков)
    7. Временно́е разбиение на train / test (по дате, не случайное)

Что НЕ делается на этом этапе:
    - Не создаются новые признаки (is_online, mcc_risk_group, hour, и т.д.)
    - Не применяется SMOTE
    - log-трансформации суммы не применяются
    - Агрегированные статистики по клиенту/карте не считаются
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# 1. ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ДАННЫХ
# =============================================================================

def load_and_merge(
    transactions_path: str,
    users_path: str,
    cards_path: str,
    mcc_path: str,
    labels_path: str,
) -> pd.DataFrame:
    """
    Загружает все файлы датасета и объединяет их в один DataFrame.

    Join-ключи:
        transactions.client_id -> users.id
        transactions.card_id   -> cards.id

    Метки фрода join-ятся по transactions.id.

    Returns:
        Объединённый DataFrame до какой-либо предобработки.
    """
    print("[STEP 1] Загрузка файлов...")

    transactions = pd.read_csv(transactions_path)
    users        = pd.read_csv(users_path)
    cards        = pd.read_csv(cards_path)

    with open(mcc_path, "r") as f:
        mcc_codes = json.load(f)

    with open(labels_path, "r") as f:
        labels_raw = json.load(f)

    print(f"  transactions : {transactions.shape}")
    print(f"  users        : {users.shape}")
    print(f"  cards        : {cards.shape}")
    print(f"  labels       : {len(labels_raw['target'])} записей")

    # --- Метки фрода ---
    labels = pd.Series(
        {int(k): 1 if v.strip().lower() == "yes" else 0
         for k, v in labels_raw["target"].items()},
        name="is_fraud",
    ).rename_axis("id").reset_index()

    # --- MCC: добавляем текстовое описание категории торговца ---
    # Оставляем числовой mcc как признак; описание пригодится в Этапе 4 (FE)
    transactions["mcc"] = transactions["mcc"].astype(str)
    transactions["mcc_description"] = transactions["mcc"].map(mcc_codes).fillna("Unknown")
    transactions["mcc"] = transactions["mcc"].astype(int)

    # --- Переименование столбцов users и cards перед JOIN ---
    # Добавляем префиксы, чтобы избежать коллизий имён
    users = users.rename(columns={"id": "client_id"})
    users.columns = (
        ["client_id"] + [f"user_{c}" for c in users.columns[1:]]
    )

    cards = cards.rename(columns={"id": "card_id", "client_id": "card_client_id"})
    cards.columns = (
        ["card_id", "card_client_id"] +
        [f"card_{c}" for c in cards.columns[2:]]
    )

    # --- Объединение ---
    df = (
        transactions
        .merge(labels,  on="id",        how="inner")   # отбрасываем транзакции без меток
        .merge(users,   on="client_id", how="left")
        .merge(cards,   on="card_id",   how="left")
    )

    print(f"\n  Итоговый датасет после JOIN: {df.shape}")
    print(f"  Фрод: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.3f}%)")

    return df


# =============================================================================
# 2. ПРЕДОБРАБОТКА
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Минимальная предобработка для Этапа 3 (без feature engineering).

    Шаги:
        2.1  Разбор даты на числовые компоненты (year, month, day)
        2.2  Очистка денежных столбцов ($) -> float
        2.3  Очистка дат карты -> числовые признаки
        2.4  Label Encoding категориальных столбцов
        2.5  Заполнение пропусков
        2.6  Удаление нерелевантных столбцов

    Returns:
        Предобработанный DataFrame с числовыми признаками и целевой переменной.
    """
    print("\n[STEP 2] Предобработка...")
    df = df.copy()

    # ------------------------------------------------------------------
    # 2.1  Дата транзакции -> числовые компоненты
    # ------------------------------------------------------------------
    # hour и dayofweek намеренно НЕ включаются — это feature engineering
    # и будет добавлено в Этапе 4.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Сохраняем дату отдельно для временного разбиения (используется в split)
    df["_date_for_split"] = df["date"]

    df["tx_year"]  = df["date"].dt.year
    df["tx_month"] = df["date"].dt.month
    df["tx_day"]   = df["date"].dt.day
    df.drop(columns=["date"], inplace=True)

    # ------------------------------------------------------------------
    # 2.2  Денежные столбцы: убираем $ и запятые, приводим к float
    # ------------------------------------------------------------------
    money_cols = [
        "amount",
        "user_per_capita_income",
        "user_yearly_income",
        "user_total_debt",
        "card_credit_limit",
    ]
    for col in money_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .replace("nan", np.nan)
                .astype(float)
            )

    # ------------------------------------------------------------------
    # 2.3  Даты карты -> числовые признаки
    # ------------------------------------------------------------------
    # card_expires: формат "MM/YYYY" -> год истечения
    if "card_expires" in df.columns:
        df["card_expires_year"] = (
            pd.to_datetime(df["card_expires"], format="%m/%Y", errors="coerce")
            .dt.year
        )
        df.drop(columns=["card_expires"], inplace=True)

    # card_acct_open_date: формат "MM/YYYY" -> год открытия счёта
    if "card_acct_open_date" in df.columns:
        df["card_acct_open_year"] = (
            pd.to_datetime(df["card_acct_open_date"], format="%m/%Y", errors="coerce")
            .dt.year
        )
        df.drop(columns=["card_acct_open_date"], inplace=True)

    # ------------------------------------------------------------------
    # 2.4  Label Encoding категориальных признаков
    # ------------------------------------------------------------------
    # Признаки с низкой/средней кардинальностью кодируем Label Encoding.
    # merchant_city исключается (тысячи уникальных значений ->
    #   случайные коды без смысла; агрегация по риску — это FE, Этап 4).
    # mcc_description исключается по той же причине (будет использована в FE).
    cat_cols = [
        "use_chip",          # 3 значения: Chip / Swipe / Online Transaction
        "merchant_state",    # ~50 штатов + Unknown
        "user_gender",       # Male / Female
        "card_card_brand",   # Visa / Mastercard / Discover / Amex
        "card_card_type",    # Credit / Debit / Debit (Prepaid)
        "card_has_chip",     # YES / NO
        "card_card_on_dark_web",  # Yes / No
    ]

    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
            df[col] = le.fit_transform(df[col])

    # ------------------------------------------------------------------
    # 2.5  Заполнение пропущенных значений
    # ------------------------------------------------------------------
    # zip: пропуски у онлайн-транзакций (нет физической геолокации) -> 0
    if "zip" in df.columns:
        df["zip"] = df["zip"].fillna(0).astype(int)

    # merchant_state уже обработан в 2.4 (fillna("Unknown") перед LE)

    # Числовые пользовательские признаки: медиана (нейтральная стратегия)
    num_user_cols = [
        "user_per_capita_income", "user_yearly_income",
        "user_total_debt", "card_credit_limit",
        "card_expires_year", "card_acct_open_year",
    ]
    for col in num_user_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ------------------------------------------------------------------
    # 2.6  Удаление нерелевантных столбцов
    # ------------------------------------------------------------------
    drop_cols = [
        # Идентификаторы — случайные числа, не несут предсказательной силы
        "id", "client_id", "card_id", "merchant_id", "card_client_id",
        # Высококардинальные строковые столбцы (будут задействованы в FE)
        "merchant_city", "mcc_description",
        # Адрес пользователя — неструктурированная строка
        "user_address",
        # Номер карты и CVV — случайные числа
        "card_card_number", "card_cvv",
        # errors: 98.4% пропусков — фактически пустой столбец
        "errors",
        # Дублирующие демографические поля
        "user_birth_year", "user_birth_month",
    ]
    existing_drop = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing_drop, inplace=True)

    print(f"  Удалено столбцов: {len(existing_drop)} -> {existing_drop}")
    print(f"  Итоговые признаки ({df.shape[1]}): {df.columns.tolist()}")
    print(f"\n  Пропущенные значения после предобработки:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("    Пропущенных значений нет.")
    else:
        print(missing)

    return df


# =============================================================================
# 3. ВРЕМЕННО́Е РАЗБИЕНИЕ НА TRAIN / TEST
# =============================================================================

def time_split(
    df: pd.DataFrame,
    test_cutoff_year: int = 2019,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разделяет данные на train и test по временно́му порогу.

    Логика:
        train : транзакции ДО  test_cutoff_year  (модель обучается на прошлом)
        test  : транзакции С   test_cutoff_year  (модель предсказывает будущее)

    Это соответствует реальному сценарию антифрод-системы и рекомендации
    научного руководителя использовать временно́е разбиение вместо
    случайного StratifiedKFold.

    Args:
        df               : предобработанный DataFrame (содержит _date_for_split)
        test_cutoff_year : год начала тестовой выборки (включительно)

    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"\n[STEP 3] Временно́е разбиение (test >= {test_cutoff_year})...")

    date_col = "_date_for_split"
    df = df.sort_values(date_col).reset_index(drop=True)

    train_mask = df[date_col].dt.year < test_cutoff_year
    test_mask  = df[date_col].dt.year >= test_cutoff_year

    train_df = df[train_mask].drop(columns=[date_col])
    test_df  = df[test_mask].drop(columns=[date_col])

    X_train = train_df.drop(columns=["is_fraud"])
    y_train = train_df["is_fraud"]
    X_test  = test_df.drop(columns=["is_fraud"])
    y_test  = test_df["is_fraud"]

    print(f"  Train: {len(X_train):,} записей "
          f"| фрод: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
    print(f"  Test : {len(X_test):,} записей "
          f"| фрод: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")

    return X_train, X_test, y_train, y_test


# =============================================================================
# ТОЧКА ВХОДА (для автономного запуска / отладки)
# =============================================================================

if __name__ == "__main__":
    from config import (
        TRANSACTIONS_PATH, USERS_PATH, CARDS_PATH,
        MCC_PATH, LABELS_PATH, DEFAULT_PREPROCESS_DATA,
    )

    df_raw = load_and_merge(
        TRANSACTIONS_PATH, USERS_PATH, CARDS_PATH, MCC_PATH, LABELS_PATH
    )
    df_processed = preprocess(df_raw)

    X_train, X_test, y_train, y_test = time_split(df_processed)

    # Сохраняем для последующего использования
    Path(DEFAULT_PREPROCESS_DATA).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(DEFAULT_PREPROCESS_DATA, index=False)
    print(f"\n[INFO] Сохранено: {DEFAULT_PREPROCESS_DATA}")

    print("\n--- Первые строки ---")
    print(df_processed.head(3).T)