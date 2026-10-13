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

# Цвета для способов оплаты (согласованы с категориями use_chip в датасете)
PAYMENT_PALETTE = {
    "Chip Transaction":   "#55A868",
    "Swipe Transaction":  "#4C72B0",
    "Online Transaction": "#DD8452",
}

# Дата EMV liability shift в США (переход ответственности за фрод по картам
# без чипа на сторону, не перешедшую на чиповые терминалы). Используется
# только как справочная вертикальная линия на графике, не как часть
# препроцессинга или признаков.
EMV_SHIFT_DATE = pd.Timestamp("2015-10-01")

# Денежные столбцы пользователей и карт, которые в сыром виде хранятся строками
# вида "$12345". Для них создаются числовые копии с суффиксом _num
# (по аналогии с amount -> amount_num); исходные строковые столбцы сохраняются.
USER_MONEY_COLS = ["user_per_capita_income", "user_yearly_income", "user_total_debt"]
CARD_MONEY_COLS = ["card_credit_limit"]

# Минимальное число транзакций в категории, чтобы показывать её fraud rate на
# графиках: при меньшем n оценка доли слишком шумная (например, 60% на 100
# транзакциях). Полная таблица по всем категориям сохраняется в CSV.
MIN_CATEGORY_N = 300


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def money_to_float(s: pd.Series) -> pd.Series:
    """'$1,234.50' / '$-77.00' -> float. Пропуски остаются NaN."""
    return s.astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)


def wilson_ci(k, n, z: float = 1.96):
    """
    95% доверительный интервал Уилсона для доли k/n.
    Векторизован: k и n могут быть числами или массивами/Series.
    Возвращает (нижняя граница, верхняя граница) в долях (не в процентах).
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (center - half) / denom, (center + half) / denom


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

    # --- Числовые копии денежных столбцов (*_num) ---
    # Чистим до JOIN: в users всего ~2000 строк, в cards ~6000, поэтому это
    # на порядки быстрее, чем парсить строки в 8.9 млн строк после слияния.
    for col in USER_MONEY_COLS:
        users[f"{col}_num"] = money_to_float(users[col])
    for col in CARD_MONEY_COLS:
        cards[f"{col}_num"] = money_to_float(cards[col])

    # --- JOIN ---
    df = (
        transactions
        .merge(labels,  on="id",      how="inner")
        .merge(users,   on="client_id", how="left")
        .merge(cards,   on="card_id",   how="left")
    )

    # --- Очистка amount: убираем $ и запятые для числового анализа ---
    df["amount_num"] = money_to_float(df["amount"])

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
    ci_lo, ci_hi = (float(x) for x in wilson_ci(n_fraud, n))

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

    # Сохраняем таблицу — переиспользуется в payment_method_over_time()
    # для сопоставления fraud rate с динамикой способа оплаты.
    monthly.to_csv(OUTPUT_DIR / "fraud_over_time.csv")

    print(f"  Сохранено: fraud_over_time.png, fraud_over_time.csv")


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

    # user_yearly_income и user_total_debt берём в числовом виде (*_num):
    # исходные столбцы — строки "$12345", гистограмма по ним строится как по
    # категориальным значениям с нечитаемой осью.
    num_cols = [
        "amount_num", "user_current_age", "user_credit_score",
        "user_yearly_income_num", "user_total_debt_num", "user_num_credit_cards"
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
    """
    Доля фрода по категориальным признакам.

    На графике показываются только категории с n >= MIN_CATEGORY_N, для каждой
    указаны n (в подписи оси) и 95% доверительный интервал Уилсона (усы).
    Это защищает от вывода вроде "категория с 60% фрода", построенного на
    нескольких десятках транзакций. Таблица по ВСЕМ категориям (включая
    скрытые с графика) сохраняется в categorical_fraud_rates.csv.
    """
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten()

    all_stats = []

    for i, (col, title) in enumerate(cat_cols.items()):
        stats = (
            df.groupby(col)["is_fraud"]
            .agg(fraud="sum", n="count")
            .reset_index()
            .rename(columns={col: "category"})
        )
        stats["fraud_rate"] = stats["fraud"] / stats["n"]
        stats["ci_low"], stats["ci_high"] = wilson_ci(stats["fraud"], stats["n"])
        stats["shown"] = stats["n"] >= MIN_CATEGORY_N

        # Полная таблица (в процентах) — для текста работы и проверки скрытых категорий
        out = stats.copy()
        out.insert(0, "feature", col)
        for c in ("fraud_rate", "ci_low", "ci_high"):
            out[f"{c}_pct"] = out.pop(c) * 100
        all_stats.append(out)

        # Сообщаем, что скрыто с графика (и почему)
        hidden = stats[~stats["shown"]].sort_values("fraud_rate", ascending=False)
        if len(hidden):
            top_hidden = ", ".join(
                f"{r.category} (n={r.n:,}, {r.fraud_rate*100:.1f}%)"
                for r in hidden.head(3).itertuples()
            )
            print(f"  {col}: скрыто {len(hidden)} категорий с n < {MIN_CATEGORY_N}; "
                  f"с наибольшим fraud rate: {top_hidden}")

        # Топ-15 по fraud rate среди категорий с достаточным n
        shown = (
            stats[stats["shown"]]
            .sort_values("fraud_rate", ascending=True)
            .tail(15)
        )
        if shown.empty:
            axes[i].set_visible(False)
            continue

        rate = shown["fraud_rate"].to_numpy() * 100
        lo   = shown["ci_low"].to_numpy() * 100
        hi   = shown["ci_high"].to_numpy() * 100
        labels = [f"{c}  (n={n:,})" for c, n in zip(shown["category"], shown["n"])]

        axes[i].barh(labels, rate, xerr=[rate - lo, hi - rate],
                     color=PALETTE["fraud"], alpha=0.8,
                     error_kw={"linewidth": 1, "capsize": 2})
        # Подписи значений ставим правее верхней границы доверительного интервала,
        # чтобы они не накладывались на усы
        for y, (r, h) in enumerate(zip(rate, hi)):
            axes[i].text(h, y, f" {r:.2f}%", va="center", fontsize=8)
        axes[i].set_xlim(0, hi.max() * 1.18)
        axes[i].set_title(title)
        axes[i].set_xlabel("Fraud rate (%), 95% CI Уилсона")
        axes[i].tick_params(axis="y", labelsize=9)

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Fraud rate by categorical features "
                 f"(категории с n ≥ {MIN_CATEGORY_N})", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "categorical_fraud_rates.png", bbox_inches="tight")
    plt.close()

    pd.concat(all_stats, ignore_index=True).to_csv(
        OUTPUT_DIR / "categorical_fraud_rates.csv", index=False
    )
    print(f"  Сохранено: categorical_fraud_rates.png, categorical_fraud_rates.csv")


# =============================================================================
# 9. КОРРЕЛЯЦИОННАЯ МАТРИЦА
# =============================================================================

def correlation_matrix(df: pd.DataFrame) -> None:
    """Тепловая карта корреляций числовых признаков с целевой переменной."""
    print("\n[STEP 9] Корреляционная матрица...")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Исключаем идентификаторы
    exclude = ["id", "client_id", "card_id", "merchant_id",
               "user_birth_year", "user_birth_month",
               # Идентификаторы и коды: числовое значение не имеет количественного
               # смысла, корреляции с ними (например, zip ↔ longitude = -0.86)
               # отражают географию или случайную нумерацию, а не связь признаков.
               "card_client_id", "card_card_number", "card_cvv", "zip"]
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
# 11. ДИНАМИКА СПОСОБА ОПЛАТЫ ВО ВРЕМЕНИ
# =============================================================================

def payment_method_over_time(df: pd.DataFrame) -> None:
    """
    Динамика структуры способов оплаты (use_chip) по месяцам и её
    сопоставление с динамикой fraud rate (STEP 5).

    Мотивация: по данным categorical_fraud_rates доля фрода для
    Online Transaction на порядок выше, чем для Chip/Swipe. Если доля
    онлайн-транзакций менялась во времени — например, из-за перехода
    на чиповые карты после EMV liability shift в США (октябрь 2015,
    отмечен на графике вертикальной линией как справочная точка,
    не как часть препроцессинга) — это может объяснять часть
    нестационарности fraud rate, видимой на fraud_over_time.png.

    Корреляция считается описательно (Pearson, по месячным рядам) и
    НЕ подразумевает причинно-следственной связи — совпадающая динамика
    может объясняться и другими общими факторами.
    """
    print("\n[STEP 11] Динамика способа оплаты во времени...")

    if "use_chip" not in df.columns:
        print("  Пропущено: нет столбца use_chip.")
        return

    df_t = df.copy()
    df_t["month"] = df_t["date"].dt.to_period("M")

    # --- Доля каждого способа оплаты по месяцам (среди ВСЕХ транзакций месяца) ---
    monthly_counts = (
        df_t.groupby(["month", "use_chip"])
        .size()
        .unstack(fill_value=0)
    )
    monthly_share = monthly_counts.div(monthly_counts.sum(axis=1), axis=0) * 100
    monthly_share.index = monthly_share.index.to_timestamp()

    # --- Fraud rate по месяцам (та же агрегация, что и в fraud_over_time) ---
    monthly_fraud = df_t.groupby("month")["is_fraud"].mean() * 100
    monthly_fraud.index = monthly_fraud.index.to_timestamp()

    # --- Корреляция доли Online-транзакций и fraud rate по месяцам ---
    online_col = "Online Transaction"
    corr = None
    if online_col in monthly_share.columns:
        joined = pd.concat(
            [monthly_share[online_col], monthly_fraud], axis=1
        ).dropna()
        joined.columns = ["online_share", "fraud_rate"]
        if len(joined) > 2:
            corr = joined["online_share"].corr(joined["fraud_rate"])

    # --- График: доля способов оплаты + fraud rate на общей временной оси ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for col in monthly_share.columns:
        axes[0].plot(
            monthly_share.index, monthly_share[col],
            label=col, linewidth=1.8,
            color=PAYMENT_PALETTE.get(col),
        )
    axes[0].set_title("Доля способов оплаты по месяцам")
    axes[0].set_ylabel("Доля транзакций (%)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].axvline(EMV_SHIFT_DATE, color="gray", linestyle="--", linewidth=1)
    axes[0].text(
        EMV_SHIFT_DATE, axes[0].get_ylim()[1] * 0.95,
        " EMV liability shift (окт. 2015)",
        fontsize=8, color="gray", va="top",
    )

    axes[1].plot(monthly_fraud.index, monthly_fraud.values,
                 color=PALETTE["fraud"], linewidth=1.5)
    axes[1].fill_between(monthly_fraud.index, monthly_fraud.values,
                          alpha=0.3, color=PALETTE["fraud"])
    axes[1].set_title("Monthly fraud rate (для сравнения)")
    axes[1].set_ylabel("Fraud rate (%)")
    axes[1].set_xlabel("Date")
    axes[1].axvline(EMV_SHIFT_DATE, color="gray", linestyle="--", linewidth=1)

    title = "Способ оплаты во времени vs. fraud rate"
    if corr is not None:
        title += f"  (r Online share ↔ fraud rate = {corr:.2f})"
    plt.suptitle(title, fontsize=12, y=1.01)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "payment_method_over_time.png", bbox_inches="tight")
    plt.close()

    monthly_share.to_csv(OUTPUT_DIR / "payment_method_over_time.csv")

    print(f"  Сохранено: payment_method_over_time.png, payment_method_over_time.csv")
    if corr is not None:
        print(f"  Корреляция доли Online-транзакций и fraud rate по месяцам: r = {corr:.3f}")
    else:
        print("  Корреляция не посчитана (недостаточно пересекающихся месяцев "
              "или нет столбца 'Online Transaction').")


# =============================================================================
# 12. ДИНАМИКА MCC-КАТЕГОРИЙ ВЫСОКОГО РИСКА ВО ВРЕМЕНИ
# =============================================================================

def mcc_risk_over_time(df: pd.DataFrame, top_n: int = 6, min_n: int = MIN_CATEGORY_N) -> None:
    """
    Доля транзакций по MCC-категориям с наивысшим fraud rate (из
    categorical_fraud_rates) — динамика во времени.

    Проверяет вторую гипотезу в том же духе, что и payment_method_over_time:
    не связана ли нестационарность fraud rate также со сдвигом структуры
    торговых категорий (например, ростом доли Cruise Lines / Music Stores
    в отдельные периоды), а не только со способом оплаты.

    Доля каждой категории считается от ОБЩЕГО числа транзакций за месяц
    (не только среди top_n категорий), чтобы цифры были сопоставимы
    с payment_method_over_time.png.
    """
    print("\n[STEP 12] Динамика MCC-категорий высокого риска во времени...")

    if "mcc_description" not in df.columns:
        print("  Пропущено: нет столбца mcc_description.")
        return

    # --- Отбор top_n категорий по fraud rate среди категорий с n >= min_n ---
    mcc_stats = (
        df.groupby("mcc_description")["is_fraud"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "n"})
    )
    mcc_stats = mcc_stats[mcc_stats["n"] >= min_n]
    if mcc_stats.empty:
        print(f"  Пропущено: нет MCC-категорий с n >= {min_n}.")
        return

    top_categories = (
        mcc_stats.sort_values("fraud_rate", ascending=False)
        .head(top_n)
        .index.tolist()
    )

    # --- Доля каждой top-категории от ВСЕХ транзакций за месяц ---
    df_t = df.copy()
    df_t["month"] = df_t["date"].dt.to_period("M")

    total_per_month = df_t.groupby("month").size()
    top_counts = (
        df_t[df_t["mcc_description"].isin(top_categories)]
        .groupby(["month", "mcc_description"])
        .size()
        .unstack(fill_value=0)
        .reindex(total_per_month.index, fill_value=0)
    )
    # На случай, если какая-то топ-категория отсутствует после reindex
    top_counts = top_counts.reindex(columns=top_categories, fill_value=0)
    monthly_share = top_counts.div(total_per_month, axis=0) * 100
    monthly_share.index = monthly_share.index.to_timestamp()

    # --- График ---
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in top_categories:
        ax.plot(monthly_share.index, monthly_share[col], label=col, linewidth=1.6)
    ax.set_title(f"Доля транзакций по топ-{top_n} MCC-категориям высокого риска "
                 f"(от общего трафика, %)")
    ax.set_ylabel("Доля транзакций (%)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mcc_risk_over_time.png", bbox_inches="tight")
    plt.close()

    monthly_share.to_csv(OUTPUT_DIR / "mcc_risk_over_time.csv")
    mcc_stats.loc[top_categories].to_csv(OUTPUT_DIR / "mcc_risk_top_categories.csv")

    print(f"  Топ-{top_n} категорий по fraud rate (n >= {min_n}): {top_categories}")
    print(f"  Сохранено: mcc_risk_over_time.png, mcc_risk_over_time.csv, "
          f"mcc_risk_top_categories.csv")


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
    payment_method_over_time(df)
    mcc_risk_over_time(df)

    print(f"\n[DONE] Все результаты EDA сохранены в папку: {OUTPUT_DIR.resolve()}")