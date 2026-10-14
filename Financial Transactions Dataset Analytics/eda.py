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
import matplotlib.dates as mdates
from matplotlib.patches import Patch
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

# Категории с n меньше этого порога НЕ скрываются с графика fraud rate по
# категориям, а помечаются звёздочкой (их доверительный интервал шире).
# Скрывать по n нельзя: именно в малых категориях могут быть самые сильные
# сигналы (например, Cruise Lines: n=276, из них 165 фрод, 59.8%), а
# неопределённость уже показана доверительным интервалом.
SMALL_N_MARK = 300

# Минимум транзакций в категории для графика ДОЛИ ТРАФИКА по месяцам
# (mcc_risk_over_time): при ~2 транзакциях в месяц месячная доля — чистый шум.
# Это ограничение именно для помесячной динамики, не для оценки fraud rate.
MIN_N_FOR_MONTHLY_SHARE = 300


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

    Для каждой категории указаны n (в подписи оси) и 95% доверительный интервал
    Уилсона (усы). Категории НЕ отфильтровываются по n: показываются топ-15,
    отсортированные по НИЖНЕЙ границе доверительного интервала. Такая сортировка
    сама защищает от шума: категория с малым n и случайно высокой долей уйдёт
    вниз из-за широкого интервала, а категория с малым n, но убедительным
    сигналом (интервал далеко выше базовой доли) останется наверху.
    Категории с n < SMALL_N_MARK помечены звёздочкой.
    Таблица по ВСЕМ категориям сохраняется в categorical_fraud_rates.csv.
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
        stats["small_n"] = stats["n"] < SMALL_N_MARK

        # Полная таблица (в процентах) — для текста работы
        out = stats.copy()
        out.insert(0, "feature", col)
        for c in ("fraud_rate", "ci_low", "ci_high"):
            out[f"{c}_pct"] = out.pop(c) * 100
        all_stats.append(out)

        # Топ-15 по нижней границе доверительного интервала (без фильтра по n)
        shown = stats.sort_values("ci_low", ascending=True).tail(15)

        rate = shown["fraud_rate"].to_numpy() * 100
        lo   = shown["ci_low"].to_numpy() * 100
        hi   = shown["ci_high"].to_numpy() * 100
        labels = [
            f"{c}{'*' if s else ''}  (n={n:,})"
            for c, n, s in zip(shown["category"], shown["n"], shown["small_n"])
        ]

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
                 f"(топ-15 по нижней границе 95% CI; * — категория с n < {SMALL_N_MARK})",
                 fontsize=13, y=1.01)
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

def mcc_risk_over_time(df: pd.DataFrame, top_n: int = 6, min_n: int = MIN_N_FOR_MONTHLY_SHARE) -> None:
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
# ОБЩИЕ ПОМОЩНИКИ ДЛЯ ШАГОВ 13–16
# =============================================================================

ZERO_MONTH_COLOR = "#9A9A9A"   # заливка месяцев без выявленного фрода

# Граница train/test из экспериментальной части (временной сплит)
TEST_START_DATE = pd.Timestamp("2018-01-01")

# Порог «категория заметно рискованнее среднего»: фрода в категории
# больше, чем ожидалось бы при средней доле по всем транзакциям, в LIFT_THRESHOLD раз
LIFT_THRESHOLD = 2.0

# Стековый график по MCC: первые 7 слотов проверенной категориальной палитры
# (порядок важен — соседние цвета различимы и при дальтонизме); остальное — серый.
MCC_STACK_TOP_N = 7
MCC_STACK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MCC_OTHER_COLOR = "#b8b8b4"
MCC_OTHER_LABEL = "Остальные категории"

# Доли по месяцам считаются только для месяцев, где фрода не меньше этого числа
# (при 1–3 случаях долевая структура — шум)
MIN_FRAUD_FOR_SHARE = 10


def _contiguous_blocks(periods) -> list:
    """Непрерывные блоки подряд идущих месяцев: список (начало, конец) в Period."""
    blocks = []
    if len(periods) == 0:
        return blocks
    start = prev = periods[0]
    for p in periods[1:]:
        if (p - prev).n != 1:
            blocks.append((start, prev))
            start = p
        prev = p
    blocks.append((start, prev))
    return blocks


def _shade_blocks(axes, blocks) -> None:
    """Серая заливка блоков месяцев на всех осях."""
    for s, e in blocks:
        x0 = s.to_timestamp() - pd.Timedelta(days=15)
        x1 = e.to_timestamp() + pd.Timedelta(days=15)
        for ax in axes:
            ax.axvspan(x0, x1, color=ZERO_MONTH_COLOR, alpha=0.25, linewidth=0, zorder=0)


def _annotate_no_overlap(ax, points, fontsize: float = 8.5) -> None:
    """
    Подписи точек без пересечений друг с другом и без выхода за пределы осей.
    points — список (x, y, текст) в порядке важности. Подпись, для которой
    не нашлось свободного места, пропускается.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axbox = ax.get_window_extent(renderer)
    placed = []
    candidates = [(8, 6, "left"), (8, -12, "left"), (-8, 6, "right"), (-8, -12, "right"),
                  (8, 20, "left"), (-8, 20, "right"), (8, -26, "left"), (-8, -26, "right")]
    for x, y, text in points:
        for dx, dy, ha in candidates:
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                              fontsize=fontsize, ha=ha, color="#222222")
            bb = ann.get_window_extent(renderer).expanded(1.02, 1.1)
            inside = (axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1
                      and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1)
            if inside and not any(bb.overlaps(p) for p in placed):
                placed.append(bb)
                break
            ann.remove()


def _short(text: str, width: int = 34) -> str:
    """Сокращает длинное название категории для подписи на графике."""
    text = str(text).replace(" - Musical Instruments", "")
    return text if len(text) <= width else text[: width - 1] + "…"


# =============================================================================
# 13. ОБЪЁМ ТРАНЗАКЦИЙ И ФРОД ПО МЕСЯЦАМ (ПРОВЕРКА «НУЛЕВЫХ» МЕСЯЦЕВ)
# =============================================================================


def volume_vs_fraud_over_time(df: pd.DataFrame) -> None:
    """
    Сопоставляет общее число транзакций и число выявленного фрода по месяцам.

    Цель: показать, что в месяцы с нулевым числом фрода поток транзакций
    НЕ пропадает (то есть ноль — не следствие отсутствия данных за месяц).

    Две панели с общей осью X вместо двойной оси Y: масштабы несопоставимы
    (десятки тысяч транзакций против сотен случаев фрода). Месяцы с fraud = 0
    залиты серым на обеих панелях.

    Важно: график показывает, что ТРАНЗАКЦИИ в таких месяцах есть, но не
    объясняет, почему там нет фрода (реальное отсутствие или неразмеченный фрод).
    """
    print("\n[STEP 13] Объём транзакций и фрод по месяцам...")

    month = df["date"].dt.to_period("M")
    monthly = df.groupby(month).agg(total=("is_fraud", "count"),
                                    fraud=("is_fraud", "sum"))
    monthly.index = monthly.index.to_timestamp()
    monthly["zero_fraud"] = monthly["fraud"] == 0

    # Календарная полнота: есть ли месяцы без единой транзакции в данных
    full_range = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    absent = full_range.difference(monthly.index)
    print(f"  Месяцев в диапазоне: {len(full_range)}; "
          f"месяцев без единой транзакции: {len(absent)}")

    # Непрерывные блоки нулевых месяцев
    zero_idx = monthly.index[monthly["zero_fraud"]]
    blocks = _contiguous_blocks(zero_idx.to_period("M"))

    rows = []
    for s, e in blocks:
        part = monthly.loc[s.to_timestamp():e.to_timestamp()]
        rows.append({
            "start": str(s), "end": str(e), "n_months": len(part),
            "total_transactions": int(part["total"].sum()),
            "mean_monthly_transactions": round(part["total"].mean()),
            "fraud": int(part["fraud"].sum()),
        })
    blocks_df = pd.DataFrame(rows)

    zero_m = monthly[monthly["zero_fraud"]]
    act_m = monthly[~monthly["zero_fraud"]]
    n_zero, n_all = len(zero_m), len(monthly)
    if n_zero:
        print(f"  Месяцев с fraud = 0: {n_zero} из {n_all}, блоков: {len(blocks)}")
        print(f"  Транзакций в нулевых месяцах: медиана {zero_m['total'].median():,.0f} "
              f"(мин {zero_m['total'].min():,}, макс {zero_m['total'].max():,}); "
              f"всего {zero_m['total'].sum():,} ({zero_m['total'].sum() / monthly['total'].sum():.1%} всех транзакций)")
        print(f"  Транзакций в остальных месяцах: медиана {act_m['total'].median():,.0f} "
              f"(мин {act_m['total'].min():,}, макс {act_m['total'].max():,})")
        print(blocks_df.to_string(index=False))
    else:
        print("  Месяцев с fraud = 0 нет.")

    # ---- график ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    axes[0].bar(monthly.index, monthly["total"], width=20, color=PALETTE["legitimate"])
    axes[0].set_title("Общее число транзакций по месяцам", loc="left")
    axes[0].set_ylabel("Транзакций в месяц")
    axes[0].set_ylim(0, monthly["total"].max() * 1.25)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    axes[1].bar(monthly.index, monthly["fraud"], width=20, color=PALETTE["fraud"])
    axes[1].set_title("Выявленный фрод по месяцам (is_fraud = 1)", loc="left")
    axes[1].set_ylabel("Случаев фрода в месяц")
    axes[1].set_ylim(0, max(monthly["fraud"].max() * 1.1, 1))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlabel("Месяц")

    _shade_blocks(axes, blocks)

    if n_zero:
        label = (f"Месяцы без выявленного фрода: {n_zero} из {n_all} "
                 f"(в них в среднем {zero_m['total'].mean():,.0f} транзакций/мес, "
                 f"в остальных {act_m['total'].mean():,.0f})")
        axes[0].legend(handles=[Patch(facecolor=ZERO_MONTH_COLOR, alpha=0.25, label=label)],
                       loc="upper left", frameon=False, fontsize=10)

    fig.suptitle("Объём транзакций и выявленный фрод по месяцам", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "volume_vs_fraud_over_time.png")
    plt.close()

    monthly.to_csv(OUTPUT_DIR / "volume_vs_fraud_over_time.csv")
    blocks_df.to_csv(OUTPUT_DIR / "zero_fraud_blocks.csv", index=False)
    print("  Сохранено: volume_vs_fraud_over_time.png, volume_vs_fraud_over_time.csv, "
          "zero_fraud_blocks.csv")


# =============================================================================
# 14. ОБЪЁМ ТРАНЗАКЦИЙ И ФРОД ПО MCC-КАТЕГОРИЯМ
# =============================================================================

def mcc_volume_vs_fraud(df: pd.DataFrame) -> None:
    """
    Для каждой MCC-категории: число транзакций (ось X) и число случаев фрода
    (ось Y), обе оси логарифмические. Пунктир — ожидаемое число случаев фрода
    при средней доле по всем транзакциям; точки выше пунктира — категории,
    где фрода больше ожидаемого. Категории без фрода показаны на отметке «0».

    Показывает то, чего не видно на графике доли фрода (STEP 8): сколько фрода
    приходится на крупные категории и сколько — на редкие.
    Категории с n < SMALL_N_MARK помечены пустыми маркерами.
    """
    print("\n[STEP 14] Объём транзакций и фрод по MCC-категориям...")

    codes = (df[["mcc_description", "mcc"]].drop_duplicates()
             .assign(mcc=lambda d: d["mcc"].astype(str))
             .groupby("mcc_description")["mcc"].agg(lambda s: "/".join(sorted(s))))
    g = df.groupby("mcc_description").agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum"))
    g["mcc_codes"] = codes

    total_n, total_fraud = g["n"].sum(), g["fraud"].sum()
    avg = total_fraud / total_n
    g["fraud_rate_pct"] = g["fraud"] / g["n"] * 100
    g["expected_fraud"] = g["n"] * avg
    g["lift"] = g["fraud"] / g["expected_fraud"]
    g["share_of_fraud_pct"] = g["fraud"] / total_fraud * 100
    g["share_of_n_pct"] = g["n"] / total_n * 100
    g["small_n"] = g["n"] < SMALL_N_MARK
    lo, hi = wilson_ci(g["fraud"], g["n"])
    g["ci_low_pct"], g["ci_high_pct"] = lo * 100, hi * 100
    g = g.sort_values("fraud", ascending=False)

    top5 = g.head(5)
    risky = g[g["lift"] >= LIFT_THRESHOLD]
    zero = g[g["fraud"] == 0]
    print(f"  Категорий: {len(g)}; средняя доля фрода {avg:.3%}")
    print(f"  Топ-5 по числу фрода: {top5['share_of_fraud_pct'].sum():.1f}% фрода "
          f"на {top5['share_of_n_pct'].sum():.1f}% транзакций")
    print(f"  Категорий с лифтом >= {LIFT_THRESHOLD:g}: {len(risky)}; в них "
          f"{risky['share_of_fraud_pct'].sum():.1f}% фрода на {risky['share_of_n_pct'].sum():.1f}% транзакций")
    print(f"  Категорий без фрода: {len(zero)} ({zero['n'].sum():,} транзакций)")

    # ---- график ----
    fig, ax = plt.subplots(figsize=(11, 7.2))
    y = g["fraud"].clip(lower=0.5)
    xmin, xmax = g["n"].min() * 0.6, g["n"].max() * 2.5
    ymax = g["fraud"].max() * 3
    xs = np.logspace(np.log10(xmin), np.log10(xmax), 50)
    ax.plot(xs, xs * avg, color="#9A9A9A", lw=1, ls="--", zorder=1)
    ax.plot(xs, xs * avg * 10, color="#9A9A9A", lw=1.4, ls=":", zorder=1)

    big, small = g[~g["small_n"]], g[g["small_n"]]
    ax.scatter(big["n"], y[big.index], s=34, color=PALETTE["legitimate"],
               edgecolors="white", linewidths=0.8, zorder=3)
    ax.scatter(small["n"], y[small.index], s=40, facecolors="white",
               edgecolors=PALETTE["legitimate"], linewidths=1.6, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.35, ymax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    decades = [10 ** k for k in range(0, int(np.ceil(np.log10(ymax))) + 1) if 10 ** k <= ymax]
    ax.set_yticks([0.5] + decades)
    ax.set_yticklabels(["0"] + [f"{d:,.0f}".replace(",", " ") for d in decades])
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Транзакций в категории (логарифмическая шкала)")
    ax.set_ylabel("Случаев фрода в категории (логарифмическая шкала)")
    ax.set_title(f"Объём транзакций и фрод по MCC-категориям ({len(g)} категорий)", loc="left")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#9A9A9A", ls="--", lw=1, label=f"ожидаемый фрод при средней доле {avg:.2%}"),
        Line2D([], [], color="#9A9A9A", ls=":", lw=1.4, label="при доле в 10 раз выше средней"),
        Line2D([], [], marker="o", ls="", color=PALETTE["legitimate"], markeredgecolor="white",
               label=f"категория, n ≥ {SMALL_N_MARK}"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=PALETTE["legitimate"],
               markeredgewidth=1.6, label=f"n < {SMALL_N_MARK} (широкий доверительный интервал)"),
    ], loc="upper left", frameon=False, fontsize=9)

    # Подписи: крупнейшие по числу фрода + самые «перекошенные» редкие и крупные
    pick = list(g.head(6).index)
    pick += list(g[g["small_n"]].sort_values("lift", ascending=False).head(3).index)
    pick += list(g[~g["small_n"]].sort_values("lift", ascending=False).head(3).index)
    pick = list(dict.fromkeys(pick))
    _annotate_no_overlap(ax, [(g.loc[nm, "n"], y[nm], _short(nm)) for nm in pick])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mcc_volume_vs_fraud.png")
    plt.close()

    cols = ["mcc_codes", "n", "fraud", "fraud_rate_pct", "ci_low_pct", "ci_high_pct", "lift",
            "expected_fraud", "share_of_fraud_pct", "share_of_n_pct", "small_n"]
    g[cols].to_csv(OUTPUT_DIR / "mcc_volume_vs_fraud.csv")
    print("  Сохранено: mcc_volume_vs_fraud.png, mcc_volume_vs_fraud.csv")


# =============================================================================
# 15. ПОМЕСЯЧНЫЙ ФРОД В РАЗРЕЗЕ MCC-КАТЕГОРИЙ
# =============================================================================

def mcc_fraud_over_time_stacked(df: pd.DataFrame, top_n: int = MCC_STACK_TOP_N) -> None:
    """
    Верхняя панель: число случаев фрода по месяцам, столбцы разбиты по MCC
    (top_n категорий по общему числу фрода + «остальные»).
    Нижняя панель: структура фрода по категориям (%), только для месяцев,
    где фрода не меньше MIN_FRAUD_FOR_SHARE.
    Серые зоны — месяцы без фрода (по всем категориям сразу).

    Вопросы к графику: исчезает ли фрод в нулевых блоках во всех категориях
    одновременно (тогда это общий эффект разметки/данных, а не динамика
    отдельных категорий) и меняется ли структура фрода между train и test.
    """
    print("\n[STEP 15] Помесячный фрод в разрезе MCC-категорий...")

    fr = df[df["is_fraud"] == 1]
    top = fr["mcc_description"].value_counts().head(top_n).index.tolist()
    cat = fr["mcc_description"].where(fr["mcc_description"].isin(top), MCC_OTHER_LABEL)
    month = fr["date"].dt.to_period("M")
    pivot = pd.crosstab(month, cat)

    all_months = pd.period_range(df["date"].min().to_period("M"),
                                 df["date"].max().to_period("M"), freq="M")
    pivot = pivot.reindex(all_months, fill_value=0)
    cols = top + ([MCC_OTHER_LABEL] if MCC_OTHER_LABEL in pivot.columns else [])
    pivot = pivot.reindex(columns=cols, fill_value=0)

    totals = pivot.sum(axis=1)
    tx_months = set(df["date"].dt.to_period("M").unique())
    zero_months = [p for p in pivot.index if p in tx_months and totals[p] == 0]
    blocks = _contiguous_blocks(zero_months)

    colors = {c: MCC_STACK_COLORS[i] for i, c in enumerate(top)}
    colors[MCC_OTHER_LABEL] = MCC_OTHER_COLOR
    ts = pivot.index.to_timestamp()

    # Структура фрода: train против test
    is_test = ts >= TEST_START_DATE
    comp = pd.DataFrame({"train_fraud": pivot.loc[~is_test].sum(),
                         "test_fraud": pivot.loc[is_test].sum()})
    comp["train_share_pct"] = comp["train_fraud"] / comp["train_fraud"].sum() * 100
    comp["test_share_pct"] = comp["test_fraud"] / comp["test_fraud"].sum() * 100
    comp["change_pp"] = comp["test_share_pct"] - comp["train_share_pct"]
    print("  Структура фрода по категориям, train (до 2018) и test (с 2018), %:")
    print(comp[["train_share_pct", "test_share_pct", "change_pp"]].round(1).to_string())

    # ---- график ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 9.5), sharex=True)

    bottom = np.zeros(len(pivot))
    for c in pivot.columns:
        axes[0].bar(ts, pivot[c].values, bottom=bottom, width=20, color=colors[c],
                    edgecolor="white", linewidth=0.4, label=_short(c, 40))
        bottom += pivot[c].values
    axes[0].set_title("Выявленный фрод по месяцам, разбивка по MCC-категориям", loc="left")
    axes[0].set_ylabel("Случаев фрода в месяц")

    enough = totals.values >= MIN_FRAUD_FOR_SHARE
    shares = pivot.div(totals.replace(0, np.nan), axis=0) * 100
    bottom = np.zeros(len(pivot))
    for c in pivot.columns:
        vals = np.where(enough, shares[c].fillna(0).values, 0.0)
        axes[1].bar(ts, vals, bottom=bottom, width=20, color=colors[c],
                    edgecolor="white", linewidth=0.4)
        bottom += vals
    axes[1].set_ylim(0, 100)
    axes[1].set_title(f"Структура фрода по категориям, % (только месяцы с ≥ {MIN_FRAUD_FOR_SHARE} случаев фрода)",
                      loc="left")
    axes[1].set_ylabel("Доля в фроде месяца, %")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlabel("Месяц")

    _shade_blocks(axes, blocks)

    handles, labels = axes[0].get_legend_handles_labels()
    if blocks:
        handles.append(Patch(facecolor=ZERO_MONTH_COLOR, alpha=0.25))
        labels.append("месяцы без выявленного фрода")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    plt.tight_layout(rect=(0, 0.09, 1, 1))
    plt.savefig(OUTPUT_DIR / "mcc_fraud_over_time_stacked.png")
    plt.close()

    pivot.rename_axis("month").to_csv(OUTPUT_DIR / "mcc_fraud_over_time_stacked.csv")
    comp.rename_axis("mcc_description").to_csv(OUTPUT_DIR / "mcc_fraud_share_train_vs_test.csv")
    print("  Сохранено: mcc_fraud_over_time_stacked.png, mcc_fraud_over_time_stacked.csv, "
          "mcc_fraud_share_train_vs_test.csv")


# =============================================================================
# 16. MCC И СПОСОБ ОПЛАТЫ: ОБЪЯСНЯЕТ ЛИ КАНАЛ РИСК КАТЕГОРИИ
# =============================================================================

def mcc_by_channel(df: pd.DataFrame) -> None:
    """
    Проверка гипотезы: «риск MCC-категории — это в основном риск онлайн-канала».

    Для каждой категории считается ожидаемое число фрода двумя способами:
      - по средней доле фрода на всех транзакциях  -> лифт без поправки;
      - по доле фрода в каждом способе оплаты (Online / Chip / Swipe) с учётом
        того, как операции категории распределены по способам -> лифт
        с поправкой на канал.
    Если лифт с поправкой заметно ниже лифта без поправки, риск категории
    объясняется её каналом; если нет — категория рискованна и внутри канала.
    """
    print("\n[STEP 16] MCC и способ оплаты...")

    chans = {"Online Transaction": "online", "Chip Transaction": "chip", "Swipe Transaction": "swipe"}
    d = df[["mcc_description", "use_chip", "is_fraud"]].copy()
    d["ch"] = d["use_chip"].map(chans)
    d = d.dropna(subset=["ch", "mcc_description"])

    rate_ch = d.groupby("ch")["is_fraud"].mean()
    print("  Доля фрода по способам оплаты: "
          + ", ".join(f"{c} {rate_ch[c]:.3%}" for c in chans.values() if c in rate_ch.index))

    cnt = (d.groupby(["mcc_description", "ch"])
             .agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum"))
             .unstack("ch", fill_value=0))
    cnt.columns = [f"{a}_{b}" for a, b in cnt.columns]
    for c in chans.values():
        for k in ("n", "fraud"):
            if f"{k}_{c}" not in cnt.columns:
                cnt[f"{k}_{c}"] = 0

    out = pd.DataFrame(index=cnt.index)
    out["n"] = sum(cnt[f"n_{c}"] for c in chans.values())
    out["fraud"] = sum(cnt[f"fraud_{c}"] for c in chans.values())
    avg = out["fraud"].sum() / out["n"].sum()
    out["small_n"] = out["n"] < SMALL_N_MARK
    out["online_share_pct"] = cnt["n_online"] / out["n"] * 100
    out["fraud_rate_pct"] = out["fraud"] / out["n"] * 100
    n_off = cnt["n_chip"] + cnt["n_swipe"]
    out["fraud_rate_online_pct"] = np.where(cnt["n_online"] > 0, cnt["fraud_online"] / cnt["n_online"].replace(0, np.nan) * 100, np.nan)
    out["fraud_rate_offline_pct"] = np.where(n_off > 0, (cnt["fraud_chip"] + cnt["fraud_swipe"]) / n_off.replace(0, np.nan) * 100, np.nan)
    exp_channel = sum(cnt[f"n_{c}"] * rate_ch.get(c, 0.0) for c in chans.values())
    out["lift_raw"] = out["fraud"] / (out["n"] * avg)
    out["lift_channel_adj"] = out["fraud"] / exp_channel
    out = pd.concat([out, cnt], axis=1).sort_values("fraud", ascending=False)

    total_fraud = out["fraud"].sum()
    risky_raw = out[out["lift_raw"] >= LIFT_THRESHOLD]
    risky_adj = out[out["lift_channel_adj"] >= LIFT_THRESHOLD]
    rho = out["online_share_pct"].corr(out["fraud_rate_pct"], method="spearman")
    print(f"  Корреляция Спирмена: доля Online в категории ↔ доля фрода в категории: {rho:.2f}")
    print(f"  Категорий с лифтом >= {LIFT_THRESHOLD:g} без поправки: {len(risky_raw)} "
          f"({risky_raw['fraud'].sum() / total_fraud:.1%} фрода)")
    print(f"  Категорий с лифтом >= {LIFT_THRESHOLD:g} после поправки на способ оплаты: {len(risky_adj)} "
          f"({risky_adj['fraud'].sum() / total_fraud:.1%} фрода)")
    print(f"  Остаются рискованными после поправки из рискованных без неё: "
          f"{len(risky_raw.index.intersection(risky_adj.index))} из {len(risky_raw)}")

    # ---- график ----
    pos = out[out["fraud"] > 0]
    n_zero = int((out["fraud"] == 0).sum())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))

    def _scatter(ax, data, xcol, ycol):
        big, small = data[~data["small_n"]], data[data["small_n"]]
        ax.scatter(big[xcol], big[ycol], s=34, color=PALETTE["legitimate"],
                   edgecolors="white", linewidths=0.8, zorder=3)
        ax.scatter(small[xcol], small[ycol], s=40, facecolors="white",
                   edgecolors=PALETTE["legitimate"], linewidths=1.6, zorder=3)

    # Левая панель: доля Online в категории и доля фрода
    ax = axes[0]
    _scatter(ax, pos, "online_share_pct", "fraud_rate_pct")
    ax.axhline(avg * 100, color="#9A9A9A", lw=1, ls="--", zorder=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Доля онлайн-операций в категории, %")
    ax.set_ylabel("Доля фрода в категории, % (лог. шкала)")
    ax.set_title("Онлайн-доля категории и доля фрода", loc="left")
    ax.grid(True, which="major", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    pick = list(pos.sort_values("lift_raw", ascending=False).head(6).index)
    pick += list(pos.sort_values("fraud", ascending=False).head(3).index)
    _annotate_no_overlap(ax, [(pos.loc[nm, "online_share_pct"], pos.loc[nm, "fraud_rate_pct"], _short(nm, 28))
                              for nm in dict.fromkeys(pick)])

    # Правая панель: лифт без поправки и с поправкой на канал
    ax = axes[1]
    _scatter(ax, pos, "lift_raw", "lift_channel_adj")
    lo = min(pos["lift_raw"].min(), pos["lift_channel_adj"].min()) * 0.7
    hi = max(pos["lift_raw"].max(), pos["lift_channel_adj"].max()) * 1.5
    ax.plot([lo, hi], [lo, hi], color="#9A9A9A", lw=1, ls="--", zorder=1)
    ax.axvline(LIFT_THRESHOLD, color="#9A9A9A", lw=1, ls=":", zorder=1)
    ax.axhline(LIFT_THRESHOLD, color="#9A9A9A", lw=1, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Лифт без поправки (фрод / ожидаемый по средней доле)")
    ax.set_ylabel("Лифт с поправкой на способ оплаты")
    ax.set_title("Объясняется ли риск категории способом оплаты", loc="left")
    ax.grid(True, which="major", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    pick = list(pos.sort_values("lift_raw", ascending=False).head(6).index)
    pick += list(pos.sort_values("fraud", ascending=False).head(4).index)
    _annotate_no_overlap(ax, [(pos.loc[nm, "lift_raw"], pos.loc[nm, "lift_channel_adj"], _short(nm, 28))
                              for nm in dict.fromkeys(pick)])
    ax.text(0.98, 0.03, "ниже диагонали — риск объясняется способом оплаты;\n"
                        "на диагонали — категория рискованна и внутри канала",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#555555")

    fig.suptitle(f"MCC и способ оплаты (категорий без фрода не показано: {n_zero}; "
                 f"пустые маркеры — n < {SMALL_N_MARK})", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mcc_by_channel.png")
    plt.close()

    out.rename_axis("mcc_description").to_csv(OUTPUT_DIR / "mcc_by_channel.csv")
    print("  Сохранено: mcc_by_channel.png, mcc_by_channel.csv")


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
    volume_vs_fraud_over_time(df)
    mcc_volume_vs_fraud(df)
    mcc_fraud_over_time_stacked(df)
    mcc_by_channel(df)

    print(f"\n[DONE] Все результаты EDA сохранены в папку: {OUTPUT_DIR.resolve()}")