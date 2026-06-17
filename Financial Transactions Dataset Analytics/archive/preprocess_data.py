import pandas as pd
import json
import numpy as np
from pathlib import Path

from config import TRANSACTIONS_PATH, LABELS_PATH, OUTPUT_PATH


# =============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =============================================================================

def load_data(transactions_path: str, labels_path: str) -> pd.DataFrame:
    """
    Загружает транзакции и метки фрода, объединяет их в один DataFrame.

    Args:
        transactions_path: путь к файлу transactions_data.csv
        labels_path:       путь к файлу train_fraud_labels.json

    Returns:
        Объединённый DataFrame с признаком isFraud
    """

    # --- Транзакции ---
    df = pd.read_csv(transactions_path)
    print(f"[INFO] Транзакций загружено: {len(df)}")
    print(f"[INFO] Столбцы: {df.columns.tolist()}")

    # --- Метки фрода ---
    with open(labels_path, "r") as f:
        labels_raw = json.load(f)

    # Структура JSON: {"target": {"1": "Yes", "2": "No", ...}}
    labels_dict = labels_raw["target"]

    # Приводим ключи к int и значения к 0/1
    # "Yes" -> 1 (мошенничество), "No" -> 0 (легитимная транзакция)
    labels_mapped = {
        int(k): 1 if v.strip().lower() == "yes" else 0
        for k, v in labels_dict.items()
    }

    labels_series = pd.Series(labels_mapped, name="isFraud")
    labels_series.index.name = "id"
    labels_series = labels_series.reset_index()  # столбцы: id, isFraud

    print(f"[INFO] Меток фрода загружено: {len(labels_series)}")

    # --- Объединение ---
    df_merged = df.merge(labels_series, on="id", how="inner")
    print(f"[INFO] Строк после объединения: {len(df_merged)}")
    print(f"[INFO] Распределение классов:\n{df_merged['isFraud'].value_counts()}\n")

    return df_merged

# =============================================================================
# 2. ПРЕДОБРАБОТКА ПРИЗНАКОВ
# =============================================================================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет очистку и преобразование признаков.

    Шаги:
        - Разбор столбца date на числовые компоненты
        - Очистка и приведение amount к float
        - Кодирование категориальных признаков (Label Encoding)
        - Заполнение пропущенных значений
        - Удаление нерелевантных столбцов

    Args:
        df: объединённый DataFrame (транзакции + метки)

    Returns:
        Предобработанный DataFrame, готовый для обучения моделей
    """

    df = df.copy()

    # --- 2.1 Дата -> числовые признаки ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek  # 0=Пн, 6=Вс
    df.drop(columns=["date"], inplace=True)

    # --- 2.2 Очистка amount ---
    # Возможный формат: "$1,234.56" -> убираем знак $ и запятые
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )

    # --- 2.3 Признак возврата/возмещения ---
    # Отрицательные суммы трактуются как возвраты (refund/chargeback).
    # Выносим это в отдельный бинарный признак, т.к. схемы chargeback fraud
    # могут быть связаны с мошенничеством.
    df["is_refund"] = (df["amount"] < 0).astype(int)
    refund_count = df["is_refund"].sum()
    print(f"[INFO] Обнаружено возвратов (is_refund=1): {refund_count} "
          f"({refund_count / len(df) * 100:.2f}%)")

    # --- 2.4 Кодирование категориальных признаков ---
    # use_chip: например "Chip Transaction", "Swipe Transaction", "Online Transaction"
    cat_cols = ["use_chip", "merchant_city", "merchant_state", "errors"]

    for col in cat_cols:
        if col in df.columns:
            # Заполняем пропуски перед кодированием
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].astype("category").cat.codes

    # --- 2.5 Заполнение пропущенных числовых значений ---
    # zip - может содержать NaN
    df["zip"] = df["zip"].fillna(0).astype(int)

    # --- 2.6 Удаление столбцов, не несущих предсказательной ценности ---
    # id, client_id, card_id, merchant_id - идентификаторы, не признаки
    # Примечание: в более сложном анализе можно извлечь агрегированные
    # статистики (например, частоту транзакций клиента), но для baseline
    # убираем их.
    id_cols = ["id", "client_id", "card_id", "merchant_id"]
    df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True)

    print("[INFO] Предобработка завершена.")
    print(f"[INFO] Итоговые признаки: {df.columns.tolist()}")
    print(f"[INFO] Размер датасета: {df.shape}")
    print(f"\n[INFO] Пропущенные значения:\n{df.isnull().sum()}")

    return df


# =============================================================================
# 3. СОХРАНЕНИЕ РЕЗУЛЬТАТА
# =============================================================================

def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """Сохраняет предобработанный датасет в CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] Файл сохранён: {output_path}")


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":
    TRANSACTIONS_PATH = TRANSACTIONS_PATH
    LABELS_PATH = LABELS_PATH
    OUTPUT_PATH = OUTPUT_PATH

    # Шаг 1: Загрузка и объединение
    df_merged = load_data(TRANSACTIONS_PATH, LABELS_PATH)

    # Шаг 2: Предобработка
    df_processed = preprocess(df_merged)

    # Шаг 3: Сохранение
    save_processed(df_processed, OUTPUT_PATH)

    # Краткий обзор результата
    print("\n--- Первые строки предобработанного датасета ---")
    print(df_processed.head())
    print("\n--- Статистика ---")
    print(df_processed.describe())
