import itertools

import pandas as pd
import plotly.express as px
import streamlit as st

# Страница
st.set_page_config(
    page_title="Анализ межрегиональной торговли",
    layout="wide",
)


# Список регионов-исключений
EXCLUDE = {
    "российская федерация",
    "центральный федеральный округ",
    "северо-западный федеральный округ",
    "чукотский автономный округ",
    "южный федеральный округ",
    "северо-кавказский федеральный округ",
    "приволжский федеральный округ",
    "уральский федеральный округ",
    "сибирский федеральный округ",
    "дальневосточный федеральный округ",
    "ненецкий автономный округ (архангельская область)",
    "ханты-мансийский автономный округ — югра (тюменская область)",
    "ямало-ненецкий автономный округ (тюменская область)",
    "тюменская область (кроме ханты-мансийского автономного округа - югры и ямало-ненецкого автономного округа)",
}

# Датасет
def load_data(file_path) -> pd.DataFrame:
    # читаем без заголовков: после первых 3 строк идут 2 строки "шапки"
    raw = pd.read_excel(file_path, header=None, skiprows=3)

    # 0-я строка: тип товара (merged -> будут NaN)
    header_product = raw.iloc[0].copy().ffill()

    # 1-я строка: названия регионов (реальные колонки)
    header_region = raw.iloc[1].copy()

    # данные начинаются со 2-й строки
    df = raw.iloc[2:].copy()

    # собираем уникальные названия колонок
    cols = []
    for j in range(df.shape[1]):
        if j == 0:
            cols.append("region_from")
        else:
            prod = (
                str(header_product.iloc[j]).strip()
                if pd.notna(header_product.iloc[j])
                else "UNKNOWN_PRODUCT"
            )
            reg = (
                str(header_region.iloc[j]).strip()
                if pd.notna(header_region.iloc[j])
                else f"COL_{j}"
            )
            cols.append(f"{prod} | {reg}")

    df.columns = cols

    # убираем из колонок РФ и федеральные округа
    cols_keep = ["region_from"]
    for c in df.columns[1:]:
        s = str(c)

        if "|" in s:
            region_part = s.split("|", 1)[1].strip().lower()
        else:
            region_part = s.strip().lower()

        if region_part in EXCLUDE:
            continue

        cols_keep.append(c)

    df = df[cols_keep].copy()
    df["region_from"] = df["region_from"].astype(str).str.strip()

    return df


def remove_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    first_col = df.columns[0]
    df[first_col] = df[first_col].astype(str).str.strip()

    # убираем строки-агрегаты
    df = df[~df[first_col].str.lower().isin(EXCLUDE)].copy()

    # убираем столбцы-агрегаты (по названию колонки)
    cols_keep = []
    for c in df.columns:
        cc = str(c).strip().lower()
        if cc not in EXCLUDE and "федеральный округ" not in cc and cc != "российская федерация":
            cols_keep.append(c)

    return df[cols_keep].copy()


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    id_col = "region_from"
    value_cols = [c for c in df.columns if c != id_col]

    df_long = df.melt(
        id_vars=[id_col],
        value_vars=value_cols,
        var_name="prod_region",
        value_name="value",
    )

    pr = df_long["prod_region"].astype(str).str.split("|", n=1, expand=True, regex=False)
    df_long["product"] = pr[0].astype(str).str.strip()
    df_long["region_to"] = pr[1].astype(str).str.strip()

    mask_bad = df_long["region_to"].isin(["", "None", "nan", "<NA>"])
    df_long.loc[mask_bad, "region_to"] = df_long.loc[mask_bad, "prod_region"].astype(str).str.strip()

    df_long = df_long.drop(columns=["prod_region"])

    df_long["region_from"] = df_long["region_from"].astype(str).str.strip()
    df_long["region_to"] = df_long["region_to"].astype(str).str.strip()

    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long = df_long.dropna(subset=["value"])
    df_long = df_long[df_long["value"] != 0]

    # убираем петли
    df_long = df_long[df_long["region_from"] != df_long["region_to"]]

    # убираем агрегаты после split
    df_long = df_long[
        ~df_long["region_from"].str.lower().isin(EXCLUDE)
        & ~df_long["region_to"].str.lower().isin(EXCLUDE)
        & ~df_long["region_to"].str.lower().str.contains("федеральный округ", na=False)
        & (df_long["region_to"].str.lower() != "российская федерация")
    ]

    return df_long


#  Расчет индексов 
def compute_wbi_indices(
    df_long: pd.DataFrame,
    quota_q: float,
    top_in_neighbors: int = 15,
) -> pd.DataFrame:
    """
    wBI1 / wBI2

    Вершина i = region_to (получатель).
    Вес ребра w_{ji} = поставки (тонн) из region_from=j в region_to=i.

    Критическая группа S для i: сумма входящих поставок от S >= q.
    Без ограничения на размер группы
    """
    edges = (
        df_long.groupby(["region_from", "region_to"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "w"})
    ) # Агрегация нужна что бы не считать одно ребро два раза

    total_weight_network = float(edges["w"].sum()) # Общий вес сети. Знаменатель wBI_2
    if total_weight_network <= 0:
        return pd.DataFrame(columns=["region", "wBI1", "wBI2", "in_weight", "in_neighbors_used", "critical_groups_count"])

    regions = sorted(set(edges["region_from"]).union(set(edges["region_to"])))

    results = []  # Знаменатель wBI_1
    for region_i in regions:
        inc = edges[edges["region_to"] == region_i].copy()
        inc = inc[inc["w"] > 0]
        in_weight = float(inc["w"].sum())

        if in_weight <= 0 or inc.empty:
            results.append(
                dict(
                    region=region_i,
                    wBI1=0.0,
                    wBI2=0.0,
                    in_weight=in_weight,
                    in_neighbors_used=0,
                    critical_groups_count=0,
                )
            )
            continue

        inc = inc.sort_values("w", ascending=False).head(int(top_in_neighbors)) # Ограничение сложности расчета для ускорения по топ поставщикам

        # список входящих поставщиков (region_from)
        in_neighbors = tuple(inc["region_from"].astype(str).tolist())

        weights = inc["w"].astype(float).tolist()
        m = len(weights)

        wbi1_i = 0.0
        wbi2_i = 0.0
        critical_groups_count = 0

        for r in range(1, m + 1): # Расчет потенциальных критических групп 
            for idxs in itertools.combinations(range(m), r):
                s = 0.0
                for t in idxs: # 
                    s += weights[t]
                if s >= quota_q: # Если вес S >= q - группа критическая
                    critical_groups_count += 1
                    wbi1_i += s / in_weight
                    wbi2_i += s / total_weight_network

        results.append(
            dict(
                region=region_i,
                wBI1=float(wbi1_i),
                wBI2=float(wbi2_i),
                in_weight=float(in_weight),
                in_neighbors_used=int(m),
                in_neighbors=in_neighbors,  
                critical_groups_count=int(critical_groups_count),
            )
        )

    return pd.DataFrame(results).sort_values("wBI1", ascending=False)


# Cтиль  чартов 
def apply_dark_style(fig, height: int, title: str, x_title: str, y_title: str):
    # максимально совместимый стиль (не использует спорные поля типа titlefont)
    fig.update_layout(
        template="plotly_dark",
        title={"text": title, "x": 0.0, "xanchor": "left"},
        height=height,
        paper_bgcolor="#0b0f18",
        plot_bgcolor="#0b0f18",
        margin=dict(l=60, r=40, t=70, b=50),
        font=dict(color="white", size=14),

        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.10)",
            zeroline=False,
            tickfont=dict(color="white"),
        ),
        yaxis=dict(
            title=y_title,
            showgrid=False,
            tickfont=dict(color="white"),
            autorange="reversed",
        ),
        legend=dict(font=dict(color="white")),
    )

    # общий стиль трейсиков
    fig.update_traces(
        marker=dict(line=dict(width=0)),
        textposition="outside",
        textfont=dict(color="white"),
    )
    return fig



# Интерфейс
st.title("Анализ межрегиональной торговли основными пищевыми продуктами и зерном")

with st.sidebar:
    st.header("Загрузка")
    uploaded_files = st.file_uploader(
        "Выберите файлы Excel",
        type=["xlsx"],
        accept_multiple_files=True,
    )

    st.header("Параметры индексов")
    quota_q = st.number_input(
        "Квота q (тонн)",
        min_value=0.0,
        value=1000.0,
        step=100.0,
        help="Критическая группа S для региона i: сумма входящих поставок от S >= q.",
        key="quota_q",
    )

    top_in_neighbors = st.number_input(
        "topN (входящих поставщиков на регион)",
        min_value=3,
        value=5,
        step=1,
        help="Ускорение: считаем критические группы только по topN крупнейшим входящим поставщикам каждого региона.",
        key="top_in_neighbors",
    )

    top_show = st.slider(
        "Количество отображаемых регионов на графиках wBI",
        min_value=5,
        max_value=100,
        value=15,
        step=5,
        help="Это влияет только на отображение графиков wBI1 и wBI2 (не на расчёт индексов).",
        key="top_show",
    )


if not uploaded_files:
    st.info("Загрузи один или несколько Excel-файлов.")
    st.stop()

# UNION. Читаем все файлы и склеиваем строки

dfs = [load_data(f) for f in uploaded_files]
df = pd.concat(dfs, ignore_index=True)
df = remove_aggregates(df)

st.subheader("Объединённый датасет")
st.caption(f"Файлов загружено: {len(uploaded_files)} | Строк всего: {len(df)}")

rows_to_show = st.slider(
    "Выберите количество отображаемых строк",
    min_value=10,
    max_value=300,
    value=100,
    step=10,
    key="rows_to_show",
)
st.dataframe(df.head(rows_to_show), use_container_width=True)


# Выбор регионов-источников (region_from)

region_col = df.columns[0]
region_values = sorted(
    df[region_col].dropna().astype(str).str.strip().unique().tolist()
)

ALL_OPTION = "Все регионы"
options = [ALL_OPTION] + region_values

selected_regions = st.multiselect(
    f"Выбери регионы из поля '{region_col}' (можно несколько)",
    options=options,
    default=[ALL_OPTION],
    key="selected_regions",
)

if (not selected_regions) or (ALL_OPTION in selected_regions):
    selected_regions_effective = region_values
else:
    selected_regions_effective = selected_regions

if not selected_regions:
    st.warning("Выбери хотя бы один регион.")
    st.stop()


df_long = build_long(df)


# Фильтр по типу товара

products = sorted(df_long["product"].dropna().unique().tolist())
product_selected = st.selectbox(
    "Тип товара",
    options=["(Все)"] + products,
    index=0,
    key="product_selected",
)

if product_selected != "(Все)":
    df_long_f = df_long[df_long["product"] == product_selected].copy()
else:
    df_long_f = df_long.copy()

# фильтр по выбранным регионам-источникам
df_long_f = df_long_f[
    df_long_f["region_from"].isin([str(x).strip() for x in selected_regions_effective])
]

if df_long_f.empty:
    st.info("После фильтрации по регионам/товару не осталось данных.")
    st.stop()


#  СНАЧАЛА ГРАФИК СУММАРНЫХ ПОСТАВОК 
st.subheader("Суммарные поставки → регионы-партнёры")

partners = (
    df_long_f
    .groupby("region_to", as_index=False)["value"]
    .sum()
    .rename(columns={"value": "metric_value"})
    .sort_values("metric_value", ascending=False)
)

partners_cnt = len(partners)
st.caption(f"Всего регионов-партнёров (ненулевых): {partners_cnt}")

show_all = st.checkbox(
    f"Показать все регионы-партнёры ({partners_cnt})",
    value=True,
    key="show_all",
)

partners_plot = partners.copy()
if not show_all and partners_cnt >= 1:
    top_n = st.slider(
        "Top N партнёров",
        min_value=1,
        max_value=partners_cnt,
        value=min(20, partners_cnt),
        step=1,
        key="top_n",
    )
    partners_plot = partners_plot.head(top_n)

partners_plot["metric_value"] = partners_plot["metric_value"].astype(float)

fig_supply = px.bar(
    partners_plot.sort_values("metric_value", ascending=True),
    x="metric_value",
    y="region_to",
    orientation="h",
    text="metric_value",
)

fig_supply.update_traces(
    marker_color="#74b9ff",  # “неоново-голубой” 
    texttemplate="%{text:,.0f}".replace(",", " "),
    hovertemplate="<b>%{y}</b><br>%{x:,.0f}".replace(",", " ") + " тонн<extra></extra>",
)

fig_supply = apply_dark_style(
    fig_supply,
    height=420 + 26 * len(partners_plot),
    title=f"Суммарные поставки из выбранных регионов → топ {len(partners_plot)} регионов-партнёров",
    x_title="Сумма поставок, тонн",
    y_title="Регион-партнёр",
)

st.plotly_chart(fig_supply, use_container_width=True)


#  РАСЧЁТ ИНДЕКСОВ
st.subheader("Weighted Bundle Index 1 (wBI1)")
st.caption(
    "Вершины: регионы. Вес ребра: поставки (тонн) из region_from в region_to. "
    "Критическая группа S для региона i: сумма входящих поставок от S >= q."
)

indices_df = compute_wbi_indices(
    df_long=df_long_f,
    quota_q=float(quota_q),
    top_in_neighbors=int(top_in_neighbors),
)

if indices_df.empty:
    st.info("Не удалось посчитать индексы: в сети нет положительных весов.")
    st.stop()

COLUMN_RENAME = {
    "region": "Регион",
    "wBI1": "wBI₁",
    "wBI2": "wBI₂",
    "in_weight": "Суммарные входящие поставки, тонн",
    "in_neighbors_used": "Число учтённых поставщиков",
    "in_neighbors": "Ключевые регионы-поставщики",
    "critical_groups_count": "Число критических групп",
}

indices_df_view = indices_df.rename(columns=COLUMN_RENAME)
st.dataframe(indices_df_view, use_container_width=True)



top_show = min(int(top_show), len(indices_df))
plot_df = indices_df.head(top_show).sort_values("wBI1", ascending=True)

fig_wbi = px.bar(
    plot_df,
    x="wBI1",
    y="region",
    orientation="h",
    text="wBI1",
)

fig_wbi.update_traces(
    marker_color="#74b9ff",
    texttemplate="%{text:.3f}",
    hovertemplate="<b>%{y}</b><br>wBI1=%{x:.3f}<extra></extra>",
)

fig_wbi = apply_dark_style(
    fig_wbi,
    height=420 + 26 * len(plot_df),
    title=f"Топ-{top_show} регионов по wBI1 (q={quota_q}, topN={top_in_neighbors})",
    x_title="wBI1",
    y_title="Регион",
)

st.plotly_chart(fig_wbi, use_container_width=True)


# ГРАФИК wBI2
st.subheader("Weighted Bundle Index 2 (wBI2)")

plot_df_wbi2 = (
    indices_df
    .sort_values("wBI2", ascending=False)
    .head(top_show)
    .sort_values("wBI2", ascending=True)
)

fig_wbi2 = px.bar(
    plot_df_wbi2,
    x="wBI2",
    y="region",
    orientation="h",
    text="wBI2",
)

fig_wbi2.update_traces(
    marker_color="#74b9ff",
    texttemplate="%{text:.4f}",
    hovertemplate="<b>%{y}</b><br>wBI2=%{x:.4f}<extra></extra>",
)

fig_wbi2 = apply_dark_style(
    fig_wbi2,
    height=420 + 26 * len(plot_df_wbi2),
    title=f"Топ-{top_show} регионов по wBI2 (q={quota_q}, topN={top_in_neighbors})",
    x_title="wBI2",
    y_title="Регион",
)

st.plotly_chart(fig_wbi2, use_container_width=True)

