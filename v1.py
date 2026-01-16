import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Анализ межрегиональной торговли",
    layout="wide",
)


# -----------------------------
# Список регионов-исключенний
# -----------------------------
EXCLUDE = {
    "российская федерация",
    "центральный федеральный округ",
    "северо-западный федеральный округ",
    "Чукотский автономный округ",
    "чукотский автономный округ",
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


def load_data(file_path) -> pd.DataFrame:
    # читаем без заголовков: после первых 3 строк идут 2 строки "шапки"
    raw = pd.read_excel(file_path, header=None, skiprows=3)

    # 0-я строка: тип товара (merged -> будут NaN)  
    header_product = raw.iloc[0].copy()
    header_product = header_product.ffill()

    # 1-я строка: названия регионов (реальные колонки)
    header_region = raw.iloc[1].copy()

    # данные начинаются со 2-й строки
    df = raw.iloc[2:].copy()

    # собираем уникальные названия колонок
    cols = []
    for j in range(df.shape[1]):
        if j == 0:
            # первый столбец — регион-источник (ввоз)
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

    # -------------------------------------------------
    # убираем из колонок РФ и федеральные округа
    # используя EXCLUDE
    # -------------------------------------------------
    cols_keep = ["region_from"]

    for c in df.columns[1:]:
        s = str(c)

        # берём правую часть после "|", это регион
        if "|" in s:
            region_part = s.split("|", 1)[1].strip().lower()
        else:
            region_part = s.strip().lower()

        # если регион в списке EXCLUDE — выкидываем колонку
        if region_part in EXCLUDE:
            continue

        cols_keep.append(c)

    df = df[cols_keep].copy()

    # чистим region_from
    df["region_from"] = df["region_from"].astype(str).str.strip()

    return df



def remove_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    # Первый столбец — это region_from 
    first_col = df.columns[0]

    # чистим строки
    df[first_col] = df[first_col].astype(str).str.strip()

    # убираем строки-агрегаты
    df = df[~df[first_col].str.lower().isin(EXCLUDE)].copy()

    # убираем столбцы-агрегаты (по названию колонки)
    cols_keep = []
    for c in df.columns:
        cc = str(c).strip().lower()
        if cc not in EXCLUDE and "федеральный округ" not in cc and cc != "российская федерация":
            cols_keep.append(c)

    df = df[cols_keep].copy()
    return df


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    id_col = "region_from"
    value_cols = [c for c in df.columns if c != id_col]

    df_long = df.melt(
        id_vars=[id_col],
        value_vars=value_cols,
        var_name="prod_region",
        value_name="value",
    )

     # разбираем "Товар | Регион"  
    pr = df_long["prod_region"].astype(str).str.split("|", n=1, expand=True, regex=False)

    df_long["product"] = pr[0].astype(str).str.strip()
    df_long["region_to"] = pr[1].astype(str).str.strip()

    # если вдруг где-то не нашёлся разделитель  
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

    # -----------------------------------
    # УБИРАЕМ АГРЕГАТЫ ПОСЛЕ SPLIT  
    # -----------------------------------
    df_long = df_long[
        ~df_long["region_from"].str.lower().isin(EXCLUDE)
        & ~df_long["region_to"].str.lower().isin(EXCLUDE)
        & ~df_long["region_to"].str.lower().str.contains("федеральный округ", na=False)
        & (df_long["region_to"].str.lower() != "российская федерация")
    ]

    return df_long


# -----------------------------
# UI
# -----------------------------
st.title("Анализ межрегиональной торговли основными пищевыми продуктами и зерном")

with st.sidebar:
    st.header("Загрузка")
    uploaded_files = st.file_uploader(
        "Выберите файлы Excel",
        type=["xlsx"],
        accept_multiple_files=True
    )

if not uploaded_files:
    st.info("Загрузи один или несколько Excel-файлов.")
    st.stop()



# -----------------------------
# UNION ALL: читаем все файлы и склеиваем строки
# -----------------------------
dfs = []
for f in uploaded_files:
    df_tmp = load_data(f)
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)

# (опционально) чистим строки-агрегаты (region_from) уже на объединённом датасете
df = remove_aggregates(df)

st.subheader("Объединённый датасет")
st.caption(f"Файлов загружено: {len(uploaded_files)} | Строк всего: {len(df)}")

# -----------------------------
# UI  
# -----------------------------
rows_to_show = st.slider(
    "Выберите количество отображаемых строк",
    min_value=10,
    max_value=300,
    value=100,
    step=10,
    key="rows_to_show",
)
st.dataframe(df.head(rows_to_show), use_container_width=True)

# Мультивыбор регионов (только по region_from)
region_col = df.columns[0]
region_values = sorted(
    df[region_col]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

ALL_OPTION = "Все регионы"
options = [ALL_OPTION] + region_values

selected_regions = st.multiselect(
    f"Выбери регионы из поля '{region_col}' (можно несколько)",
    options=options,
    default=[ALL_OPTION],
    key="selected_regions",
)

# ЛОГИКА "Все регионы"
if (not selected_regions) or (ALL_OPTION in selected_regions):
    selected_regions_effective = region_values
else:
    selected_regions_effective = selected_regions

if not selected_regions:
    st.warning("Выбери хотя бы один регион для построения графика.")
    st.stop()

# Long-format
df_long = build_long(df)

# -----------------------------
# Фильтр по типу товара
# -----------------------------
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

# Агрегация: суммарные поставки из выбранных регионов -> в партнеров
partners = (
    df_long_f
    .groupby("region_to", as_index=False)["value"]
    .sum()
    .rename(columns={"value": "metric_value"})
    .sort_values("metric_value", ascending=False)
)

partners["metric_value"] = partners["metric_value"].round(2)

partners_cnt = len(partners)
st.caption(f"Всего регионов-партнёров (ненулевых): {partners_cnt}")

show_all = st.checkbox(
    f"Показать все регионы-партнёры ({partners_cnt})",
    value=True,
    key="show_all",
)

if not show_all and partners_cnt >= 1:
    top_n = st.slider(
        "Top N партнёров",
        min_value=0,
        max_value=partners_cnt,
        value=min(20, partners_cnt),
        step=1,
        key="top_n",
    )
    partners = partners.head(top_n)

# График
fig = px.bar(
    partners,
    x="metric_value",
    y="region_to",
    orientation="h",
    text="metric_value",
    color_discrete_sequence=["#4E79A7"],
)

fig.update_traces(
    texttemplate="%{text:,.2f}".replace(",", " ").replace(".", ","),
    textposition="outside",
    textfont=dict(family="Montserrat", size=14, color="#111"),
    marker=dict(line=dict(width=0)),
    hovertemplate="<b>%{y}</b><br>%{x:,.2f}".replace(",", " ").replace(".", ",") + " тонн<extra></extra>",
)

fig.update_layout(
    font=dict(family="Montserrat", size=20, color="#111"),
    title=dict(
        text=f"Суммарные поставки из выбранных регионов → топ {len(partners)} регионов-партнёров",
        x=0,
        xanchor="left",
        font=dict(family="Montserrat", size=16, color="#111"),
    ),
    height=420 + 28 * len(partners),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=40, r=40, t=70, b=40),
    xaxis=dict(
        title="Сумма поставок,тонн",
        title_font=dict(family="Montserrat", size=16, color="#111"),
        tickfont=dict(family="Montserrat", size=14, color="#111"),
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        tickformat=",.0f",
        separatethousands=True,
    ),
    yaxis=dict(
        title="Регион-партнёр",
        title_font=dict(family="Montserrat", size=16, color="#111"),
        tickfont=dict(family="Montserrat", size=16, color="#111"),
        autorange="reversed",
        showgrid=False,
    ),
)

st.plotly_chart(fig, use_container_width=True)
