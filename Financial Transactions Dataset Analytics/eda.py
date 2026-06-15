"""
Разведочный анализ данных (EDA) для задачи обнаружения фрода.
Датасет: Financial Transactions Dataset: Analytics
Источник: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets

Выполняется на сырых данных (до предобработки) на объединённом датасете:
    transactions_data.csv + users_data.csv + cards_data.csv + mcc_codes.json + train_fraud_labels.json

Результаты сохраняются в папку eda_output/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json
from pathlib import Path

from config import TRANSACTIONS_PATH, LABELS_PATH, EDA_OUTPUT_PATH, CARDS_PATH, MCC_PATH, USERS_PATH

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

TRANSACTIONS_PATH = TRANSACTIONS_PATH
USERS_PATH        = USERS_PATH
CARDS_PATH        = CARDS_PATH
MCC_PATH          = MCC_PATH
LABELS_PATH       = LABELS_PATH
OUTPUT_DIR        = Path(EDA_OUTPUT_PATH)

OUTPUT_DIR.mkdir(exist_ok=True)

# Единый стиль графиков
plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       11,
})
PALETTE = {"legitimate": "#4C72B0", "fraud": "#DD8452"}


# =============================================================================
# 1. ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ДАННЫХ
# =============================================================================

def load_and_merge() -> pd.DataFrame:
    """
    Загружает все файлы датасета и объединяет их в один DataFrame.
    Предобработка (кодирование, масштабирование и т.д.) НЕ выполняется.
    """
    print("[STEP 1] Загрузка файлов...")

    transactions = pd.read_csv(TRANSACTIONS_PATH)
    users        = pd.read_csv(USERS_PATH)
    cards        = pd.read_csv(CARDS_PATH)

    with open(MCC_PATH, "r") as f:
        mcc_codes = json.load(f)

    with open(LABELS_PATH, "r") as f:
        labels_raw = json.load(f)

    # --- Метки фрода ---
    labels = pd.Series(
        {int(k): 1 if v.strip().lower() == "yes" else 0
         for k, v in labels_raw["target"].items()},
        name="is_fraud"
    ).rename_axis("id").reset_index()

    # --- MCC расшифровка ---
    transactions["mcc"] = transactions["mcc"].astype(str)
    transactions["mcc_description"] = transactions["mcc"].map(mcc_codes).fillna("Unknown")

    # --- Переименование id-столбцов перед JOIN ---
    users = users.rename(columns={"id": "client_id"})
    users.columns = ["client_id"] + [f"user_{c}" for c in users.columns[1:]]

    cards = cards.rename(columns={"id": "card_id", "client_id": "card_client_id"})
    cards.columns = ["card_id", "card_client_id"] + [f"card_{c}" for c in cards.columns[2:]]

    # --- JOIN ---
    df = (
        transactions
        .merge(labels,  on="id",      how="inner")
        .merge(users,   on="client_id", how="left")
        .merge(cards,   on="card_id",   how="left")
    )

    # --- Очистка amount: убираем $ и запятые для числового анализа ---
    df["amount_num"] = (
        df["amount"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )

    # --- Парсинг даты ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- Сортировка по времени ---
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[INFO] Итоговый датасет: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"[INFO] Фрод: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.3f}%)")

    return df


# =============================================================================
# 2. БАЗОВАЯ ИНФОРМАЦИЯ
# =============================================================================

def basic_info(df: pd.DataFrame) -> None:
    """Типы признаков, мощность и описательная статистика."""
    print("\n[STEP 2] Базовая информация...")

    # --- Типы признаков ---
    dtypes_df = pd.DataFrame({
        "dtype":    df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "example":  df.iloc[0],
    })
    dtypes_df.to_csv(OUTPUT_DIR / "feature_types.csv")
    print(f"  Сохранено: feature_types.csv")

    # --- Мощность категориальных признаков ---
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    card_rows = []
    for col in cat_cols:
        top = df[col].value_counts().iloc[0] if df[col].nunique() > 0 else None
        card_rows.append({
            "column":    col,
            "n_unique":  df[col].nunique(),
            "top_value": df[col].value_counts().index[0] if df[col].nunique() > 0 else None,
            "top_freq":  top,
        })
    cardinality_df = pd.DataFrame(card_rows).sort_values("n_unique", ascending=False)
    cardinality_df.to_csv(OUTPUT_DIR / "cardinality_table.csv", index=False)
    print(f"  Сохранено: cardinality_table.csv")

    # --- Описательная статистика числовых признаков ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[num_cols].describe().T
    summary.to_csv(OUTPUT_DIR / "summary_stats.csv")
    print(f"  Сохранено: summary_stats.csv")


# =============================================================================
# 3. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ
# =============================================================================

def missing_values(df: pd.DataFrame) -> None:
    """Визуализация пропущенных значений."""
    print("\n[STEP 3] Пропущенные значения...")

    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    missing.to_csv(OUTPUT_DIR / "missing_values.csv", header=["missing_pct"])

    if missing.empty:
        print("  Пропущенных значений не обнаружено.")
        return

    fig, ax = plt.subplots(figsize=(9, max(3, len(missing) * 0.5)))
    bars = ax.barh(missing.index, missing.values, color="#4C72B0")
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlabel("Доля пропущенных значений (%)")
    ax.set_title("Пропущенные значения по столбцам")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "missing_values.png")
    plt.close()
    print(f"  Сохранено: missing_values.png, missing_values.csv")


# =============================================================================
# 4. ДИСБАЛАНС КЛАССОВ
# =============================================================================

def class_balance(df: pd.DataFrame) -> None:
    """Распределение классов и доля фрода с доверительным интервалом."""
    print("\n[STEP 4] Дисбаланс классов...")

    n       = len(df)
    n_fraud = df["is_fraud"].sum()
    p       = n_fraud / n
    # 95% доверительный интервал Уилсона
    z   = 1.96
    ci_lo = (p + z**2/(2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    ci_hi = (p + z**2/(2*n) + z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Абсолютные количества
    counts = df["is_fraud"].value_counts().sort_index()
    labels_names = ["Legitimate", "Fraud"]
    colors = [PALETTE["legitimate"], PALETTE["fraud"]]
    bars = axes[0].bar(labels_names, [counts.get(0, 0), counts.get(1, 0)], color=colors)
    axes[0].bar_label(bars, fmt="{:,.0f}", padding=4)
    axes[0].set_title("Transaction counts")
    axes[0].set_ylabel("Count")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Доля фрода с CI
    axes[1].bar(["Fraud rate"], [p * 100], color=PALETTE["fraud"],
                yerr=[[(p - ci_lo) * 100], [(ci_hi - p) * 100]], capsize=8, error_kw={"linewidth": 2})
    axes[1].set_title(f"Fraud rate = {p*100:.3f}%\n95% CI [{ci_lo*100:.3f}%, {ci_hi*100:.3f}%]")
    axes[1].set_ylabel("Fraud rate (%)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_balance.png")
    plt.close()
    print(f"  Сохранено: class_balance.png")


# =============================================================================
# 5. ДИНАМИКА ФРОДА ВО ВРЕМЕНИ
# =============================================================================

def fraud_over_time(df: pd.DataFrame) -> None:
    """Ежемесячный уровень и количество фрода."""
    print("\n[STEP 5] Динамика фрода во времени...")

    df_time = df.copy()
    df_time["month"] = df_time["date"].dt.to_period("M")

    monthly = df_time.groupby("month").agg(
        total=("is_fraud", "count"),
        fraud=("is_fraud", "sum")
    )
    monthly["fraud_rate"] = monthly["fraud"] / monthly["total"] * 100
    monthly.index = monthly.index.to_timestamp()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(monthly.index, monthly["fraud_rate"], color=PALETTE["fraud"], linewidth=1.5)
    axes[0].fill_between(monthly.index, monthly["fraud_rate"], alpha=0.3, color=PALETTE["fraud"])
    axes[0].set_title("Monthly fraud rate")
    axes[0].set_ylabel("Fraud rate (%)")

    axes[1].bar(monthly.index, monthly["fraud"], color=PALETTE["legitimate"], width=20)
    axes[1].set_title("Monthly fraud count")
    axes[1].set_ylabel("Fraud count")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fraud_over_time.png")
    plt.close()
    print(f"  Сохранено: fraud_over_time.png")


# =============================================================================
# 6. РАСПРЕДЕЛЕНИЕ СУММЫ ТРАНЗАКЦИЙ
# =============================================================================

def amount_distribution(df: pd.DataFrame) -> None:
    """Распределение суммы транзакций по классам (лог-шкала)."""
    print("\n[STEP 6] Распределение суммы транзакций...")

    df_pos = df[df["amount_num"] > 0].copy()   # убираем нули/отрицательные для лог-шкалы
    legit = df_pos[df_pos["is_fraud"] == 0]["amount_num"]
    fraud = df_pos[df_pos["is_fraud"] == 1]["amount_num"]

    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.logspace(np.log10(df_pos["amount_num"].min() + 0.01),
                       np.log10(df_pos["amount_num"].max()), 60)

    ax.hist(legit, bins=bins, density=True, alpha=0.6, color=PALETTE["legitimate"],
            label=f"Legitimate (n={len(legit):,})")
    ax.hist(fraud, bins=bins, density=True, alpha=0.6, color=PALETTE["fraud"],
            label=f"Fraud (n={len(fraud):,})")

    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction amount distribution by class (log scale)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amount_distribution.png")
    plt.close()
    print(f"  Сохранено: amount_distribution.png")


# =============================================================================
# 7. РАСПРЕДЕЛЕНИЕ ПО ЧИСЛОВЫМ ПРИЗНАКАМ
# =============================================================================

def numeric_distributions(df: pd.DataFrame) -> None:
    """Гистограммы числовых признаков с разбивкой по классам."""
    print("\n[STEP 7] Распределение числовых признаков...")

    num_cols = [
        "amount_num", "user_current_age", "user_credit_score",
        "user_yearly_income", "user_total_debt", "user_num_credit_cards"
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        for label, color in [(0, PALETTE["legitimate"]), (1, PALETTE["fraud"])]:
            subset = df[df["is_fraud"] == label][col].dropna()
            axes[i].hist(subset, bins=30, alpha=0.6, color=color, density=True,
                         label="Legitimate" if label == 0 else "Fraud")
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8)

    # Скрыть лишние оси
    for j in range(len(num_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric feature distributions by class", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "numeric_distributions.png", bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: numeric_distributions.png")


# =============================================================================
# 8. ФРОД ПО КАТЕГОРИАЛЬНЫМ ПРИЗНАКАМ
# =============================================================================

def categorical_fraud_rates(df: pd.DataFrame) -> None:
    """Доля фрода по категориальным признакам."""
    print("\n[STEP 8] Фрод по категориальным признакам...")

    cat_cols = {
        "use_chip":         "Способ оплаты",
        "mcc_description":  "Категория торговца (MCC)",
        "user_gender":      "Пол клиента",
        "card_card_brand":  "Платёжная система",
        "card_card_type":   "Тип карты",
        "card_has_chip":    "Наличие чипа",
    }
    cat_cols = {k: v for k, v in cat_cols.items() if k in df.columns}

    n_cols = 2
    n_rows = (len(cat_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()

    for i, (col, title) in enumerate(cat_cols.items()):
        fraud_rate = (
            df.groupby(col)["is_fraud"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "fraud_rate", "count": "n"})
            .sort_values("fraud_rate", ascending=True)
        )
        # Показываем топ-15 категорий для читаемости
        fraud_rate = fraud_rate.tail(15)
        bars = axes[i].barh(fraud_rate.index, fraud_rate["fraud_rate"] * 100,
                            color=PALETTE["fraud"], alpha=0.8)
        axes[i].set_title(title)
        axes[i].set_xlabel("Fraud rate (%)")
        axes[i].bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Fraud rate by categorical features", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "categorical_fraud_rates.png", bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: categorical_fraud_rates.png")


# =============================================================================
# 9. КОРРЕЛЯЦИОННАЯ МАТРИЦА
# =============================================================================

def correlation_matrix(df: pd.DataFrame) -> None:
    """Тепловая карта корреляций числовых признаков с целевой переменной."""
    print("\n[STEP 9] Корреляционная матрица...")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Исключаем идентификаторы
    exclude = ["id", "client_id", "card_id", "merchant_id",
               "user_birth_year", "user_birth_month"]
    num_cols = [c for c in num_cols if c not in exclude]

    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(max(10, len(num_cols)), max(8, len(num_cols) * 0.8)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}
    )
    ax.set_title("Correlation matrix (numeric features)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png", bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: correlation_matrix.png")


# =============================================================================
# 10. ФРОД ПО ЧАСАМ И ДНЯМ НЕДЕЛИ
# =============================================================================

def time_patterns(df: pd.DataFrame) -> None:
    """Паттерны фрода по часу суток и дню недели."""
    print("\n[STEP 10] Временные паттерны фрода...")

    df_t = df.copy()
    df_t["hour"]      = df_t["date"].dt.hour
    df_t["dayofweek"] = df_t["date"].dt.dayofweek
    days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # По часу суток
    hour_fraud = df_t.groupby("hour")["is_fraud"].mean() * 100
    axes[0].bar(hour_fraud.index, hour_fraud.values, color=PALETTE["fraud"], alpha=0.8)
    axes[0].set_title("Fraud rate by hour of day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Fraud rate (%)")
    axes[0].set_xticks(range(0, 24))

    # По дню недели
    dow_fraud = df_t.groupby("dayofweek")["is_fraud"].mean() * 100
    axes[1].bar([days[i] for i in dow_fraud.index], dow_fraud.values,
                color=PALETTE["fraud"], alpha=0.8)
    axes[1].set_title("Fraud rate by day of week")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Fraud rate (%)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_patterns.png")
    plt.close()
    print(f"  Сохранено: time_patterns.png")


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================

if __name__ == "__main__":

    df = load_and_merge()

    basic_info(df)
    missing_values(df)
    class_balance(df)
    fraud_over_time(df)
    amount_distribution(df)
    numeric_distributions(df)
    categorical_fraud_rates(df)
    correlation_matrix(df)
    time_patterns(df)

    print(f"\n[DONE] Все результаты EDA сохранены в папку: {OUTPUT_DIR.resolve()}")