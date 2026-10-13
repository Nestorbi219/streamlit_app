import itertools
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# Страница
# =========================
st.set_page_config(
    page_title="TradeFlow — анализ межрегиональной торговли",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp { background: #07111f; }
        [data-testid="stSidebar"] { background: #0b1728; }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #102239 0%, #0b192b 100%);
            border: 1px solid rgba(122, 162, 247, 0.18);
            border-radius: 14px;
            padding: 16px 18px;
        }
        .tradeflow-hero {
            background:
                radial-gradient(circle at 90% 15%, rgba(56, 189, 248, .20), transparent 30%),
                linear-gradient(135deg, #112b46 0%, #0b182a 65%, #0d2035 100%);
            border: 1px solid rgba(125, 211, 252, .20);
            border-radius: 20px;
            padding: 30px 34px;
            margin-bottom: 22px;
        }
        .tradeflow-kicker {
            color: #7dd3fc;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .14em;
            text-transform: uppercase;
        }
        .tradeflow-hero h1 {
            color: #f8fafc;
            font-size: clamp(2rem, 4vw, 3.25rem);
            line-height: 1.05;
            margin: 8px 0 12px;
        }
        .tradeflow-hero p {
            color: #cbd5e1;
            font-size: 1.05rem;
            max-width: 820px;
            margin: 0;
        }
        .tradeflow-note {
            border-left: 3px solid #38bdf8;
            color: #cbd5e1;
            padding: 4px 0 4px 14px;
            margin: 8px 0 18px;
        }
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 10px;
            font-weight: 650;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Список регионов-исключений
# =========================
# Агрегаты (РФ, федеральные округа) и «составные» субъекты, которые в таблицах Росстата
# дублируют свои части: Архангельская область = Архангельская обл. (кроме НАО) + НАО,
# Тюменская область = Тюменская обл. (кроме ХМАО и ЯНАО) + ХМАО + ЯНАО.
# Части (НАО, ХМАО, ЯНАО, Чукотский АО и др.) — самостоятельные узлы сети.
EXCLUDE = {
    "российская федерация",
    "центральный федеральный округ",
    "северо-западный федеральный округ",
    "южный федеральный округ",
    "северо-кавказский федеральный округ",
    "приволжский федеральный округ",
    "уральский федеральный округ",
    "сибирский федеральный округ",
    "дальневосточный федеральный округ",
    "архангельская область",
    "тюменская область",
}

# =========================
# Датасет
# =========================
# Ориентация таблиц Росстата «Межрегиональная торговля основными пищевыми продуктами
# и зерном»: по горизонтали (столбцы) — субъекты, откуда производится ВЫВОЗ,
# по вертикали (строки) — субъекты, куда осуществляется ВВОЗ.
# Поэтому первый столбец — регион-получатель (region_to), а регион в шапке столбца —
# регион-поставщик (region_from).
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    # читаем без заголовков: после первых 3 строк идут 2 строки "шапки"
    raw = pd.read_excel(BytesIO(file_bytes), header=None, skiprows=3)

    if raw.shape[0] < 3 or raw.shape[1] < 2:
        raise ValueError(
            "Файл не соответствует ожидаемой структуре: после первых трёх строк "
            "нужны две строки заголовка и хотя бы одна строка данных."
        )

    # 0-я строка: тип товара (merged -> будут NaN)
    header_product = raw.iloc[0].copy().ffill()

    # 1-я строка: регионы-поставщики (вывоз)
    header_region = raw.iloc[1].copy()

    # данные начинаются со 2-й строки
    df = raw.iloc[2:].copy()

    # собираем уникальные названия колонок
    cols = []
    for j in range(df.shape[1]):
        if j == 0:
            cols.append("region_to")  # первый столбец — регион-получатель (ввоз)
        else:
            prod = (
                str(header_product.iloc[j]).replace("\xa0", " ").strip()
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

    # убираем из колонок РФ, федеральные округа и составные субъекты
    cols_keep = ["region_to"]
    for c in df.columns[1:]:
        s = str(c)

        if "|" in s:
            region_part = s.rsplit("|", 1)[1].strip().lower()
        else:
            region_part = s.strip().lower()

        if region_part in EXCLUDE or "федеральный округ" in region_part:
            continue

        cols_keep.append(c)

    df = df[cols_keep].copy()
    df["region_to"] = df["region_to"].astype(str).str.strip()
    df = df[~df["region_to"].str.lower().isin({"", "nan", "none"})]

    return df


@st.cache_data(show_spinner=False)
def build_demo_data() -> pd.DataFrame:
    """Создаёт небольшой синтетический набор данных для знакомства с интерфейсом."""
    regions = [
        "Республика Адыгея",
        "Краснодарский край",
        "Ростовская область",
        "Ставропольский край",
        "Волгоградская область",
        "Воронежская область",
        "Астраханская область",
        "Республика Дагестан",
    ]
    products = ["Зерно, тонна", "Овощи, тонна"]
    # та же ориентация, что у таблиц Росстата: строки — получатели, столбцы — поставщики
    data: dict[str, list[object]] = {"region_to": regions}

    for product_index, product in enumerate(products):
        for source_index, source in enumerate(regions):
            values = []
            for destination_index, _destination in enumerate(regions):
                if source_index == destination_index:
                    values.append(0)
                    continue
                amount = (
                    ((source_index + 3) * (destination_index + 5) * 173)
                    + (product_index + 1) * 620
                ) % 5400 + 250
                values.append(float(amount))
            data[f"{product} | {source}"] = values

    return pd.DataFrame(data)


def format_tonnes(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


def remove_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    first_col = df.columns[0]
    df[first_col] = df[first_col].astype(str).str.strip()

    # убираем строки-агрегаты
    df = df[~df[first_col].str.lower().isin(EXCLUDE)].copy()

    # убираем столбцы-агрегаты (по региону в названии колонки)
    cols_keep = []
    for c in df.columns:
        cc = str(c).rsplit("|", 1)[-1].strip().lower()
        if cc not in EXCLUDE and "федеральный округ" not in cc:
            cols_keep.append(c)

    return df[cols_keep].copy()


def split_product_unit(name: str) -> tuple[str, str]:
    """«Мука, тонна» -> («Мука», «тонна»). Единица — текст после последней запятой."""
    name = str(name).strip()
    if "," in name:
        product, unit = name.rsplit(",", 1)
        return product.strip(), unit.strip()
    return name, "не указана"


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """Широкая таблица Росстата -> рёбра (region_from, region_to, product, unit, value).

    region_to — значение первого столбца (регион ввоза),
    region_from — регион из шапки столбца (регион вывоза).
    """
    id_col = "region_to"
    value_cols = [c for c in df.columns if c != id_col]

    df_long = df.melt(
        id_vars=[id_col],
        value_vars=value_cols,
        var_name="prod_region",
        value_name="value",
    )

    pr = df_long["prod_region"].astype(str).str.rsplit("|", n=1, expand=True)
    if pr.shape[1] < 2:
        pr[1] = None
    df_long["region_from"] = pr[1].astype(str).str.strip()
    prod_unit = pr[0].astype(str).map(split_product_unit)
    df_long["product"] = prod_unit.str[0]
    df_long["unit"] = prod_unit.str[1]

    mask_bad = df_long["region_from"].isin(["", "None", "nan", "<NA>"])
    df_long.loc[mask_bad, "region_from"] = df_long.loc[mask_bad, "prod_region"].astype(str).str.strip()

    df_long = df_long.drop(columns=["prod_region"])

    df_long["region_from"] = df_long["region_from"].astype(str).str.strip()
    df_long["region_to"] = df_long["region_to"].astype(str).str.strip()

    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")
    df_long = df_long.dropna(subset=["value"])
    df_long = df_long[df_long["value"] != 0]

    # убираем петли
    df_long = df_long[df_long["region_from"] != df_long["region_to"]]

    # убираем агрегаты после split (с обеих сторон ребра)
    df_long = df_long[
        ~df_long["region_from"].str.lower().isin(EXCLUDE)
        & ~df_long["region_to"].str.lower().isin(EXCLUDE)
        & ~df_long["region_from"].str.lower().str.contains("федеральный округ", na=False)
        & ~df_long["region_to"].str.lower().str.contains("федеральный округ", na=False)
    ]

    return df_long[["region_from", "region_to", "product", "unit", "value"]]


# =========================
#  Расчет индексов
# =========================
INDEX_COLUMNS = [
    "region",
    "wBI1",
    "wBI1_norm",
    "wBI2",
    "in_weight",
    "quota",
    "in_neighbors_used",
    "in_neighbors",
    "top1_supplier",
    "top1_share",
    "min_group_size",
    "critical_groups_count",
    "critical_groups_top3",
    "critical_groups_top3_supply",
]


def empty_indices_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=INDEX_COLUMNS)


def compute_wbi_indices(
    df_long: pd.DataFrame,
    quota_q: float,
    top_in_neighbors: int = 15,
    quota_mode: str = "share",
    quota_map: dict | None = None,
) -> pd.DataFrame:
    """
    wBI1 / wBI2

    Вершина i = region_to (получатель).
    Вес ребра w_{ji} = поставки из region_from=j в region_to=i.

    Критическая группа S для i: сумма входящих поставок от S >= q_i, где
      quota_mode="share":    q_i = quota_q * (суммарный входящий поток i), 0 < quota_q <= 1;
      quota_mode="absolute": q_i = quota_q (в единицах веса);
      quota_map — явные квоты {регион: q_i} (используются для графа непрямого влияния).

    wBI1_norm = wBI1 / 2^(m-1), где m — число учтённых поставщиков: верхняя граница wBI1
    равна 2^(m-1), поэтому нормированный индекс лежит в [0, 1] и сопоставим для регионов
    с разным числом поставщиков.
    """

    edges = (
        df_long.groupby(["region_from", "region_to"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "w"})
    )

    total_weight_network = float(edges["w"].sum())
    if total_weight_network <= 0:
        return empty_indices_frame()

    regions = sorted(set(edges["region_from"]).union(set(edges["region_to"])))

    results = []
    for region_i in regions:
        inc = edges[edges["region_to"] == region_i].copy()
        inc = inc[inc["w"] > 0]
        in_weight = float(inc["w"].sum())

        if quota_map is not None:
            quota_i = float(quota_map.get(region_i, float("inf")))
        elif quota_mode == "share":
            quota_i = float(quota_q) * in_weight
        else:
            quota_i = float(quota_q)

        if in_weight <= 0 or inc.empty:
            results.append(
                dict(
                    region=region_i,
                    wBI1=0.0,
                    wBI1_norm=0.0,
                    wBI2=0.0,
                    in_weight=in_weight,
                    quota=quota_i,
                    in_neighbors_used=0,
                    in_neighbors=(),
                    top1_supplier="",
                    top1_share=0.0,
                    min_group_size=0,
                    critical_groups_count=0,
                    critical_groups_top3="",
                    critical_groups_top3_supply="",
                )
            )
            continue

        # оставляем topN входящих поставщиков (это НЕ pruning непрямых ребер, это ограничение сложности комбинаторики)
        inc = inc.sort_values("w", ascending=False).head(int(top_in_neighbors))

        in_neighbors = tuple(inc["region_from"].astype(str).tolist())
        weights = inc["w"].astype(float).tolist()
        m = len(weights)

        wbi1_i = 0.0
        wbi2_i = 0.0
        critical_groups_count = 0
        min_group_size = 0
        critical_groups = []

        for r in range(1, m + 1):
            for idxs in itertools.combinations(range(m), r):
                s = 0.0
                for t in idxs:
                    s += weights[t]
                if s >= quota_i:
                    critical_groups_count += 1
                    if min_group_size == 0:
                        min_group_size = r
                    wbi1_i += s / in_weight
                    wbi2_i += s / total_weight_network
                    names = tuple(in_neighbors[t] for t in idxs)
                    critical_groups.append((float(s), names))

        critical_groups_sorted = sorted(critical_groups, key=lambda x: x[0], reverse=True)[:3]

        critical_groups_top3 = ", ".join(
            ["{" + ", ".join(g[1]) + "}" for g in critical_groups_sorted]
        )
        critical_groups_top3_supply = ", ".join([f"{g[0]:,.0f}".replace(",", " ") for g in critical_groups_sorted])

        results.append(
            dict(
                region=region_i,
                wBI1=float(wbi1_i),
                wBI1_norm=float(wbi1_i) / (2 ** (m - 1)),
                wBI2=float(wbi2_i),
                in_weight=float(in_weight),
                quota=quota_i,
                in_neighbors_used=int(m),
                in_neighbors=in_neighbors,
                top1_supplier=in_neighbors[0],
                top1_share=weights[0] / in_weight,
                min_group_size=int(min_group_size),
                critical_groups_count=int(critical_groups_count),
                critical_groups_top3=str(critical_groups_top3),
                critical_groups_top3_supply=str(critical_groups_top3_supply),
            )
        )

    return pd.DataFrame(results).sort_values("wBI1", ascending=False)


# =========================
# Cтиль чартов
# =========================
def apply_dark_style(fig, height: int, title: str, x_title: str, y_title: str):
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

    fig.update_traces(
        marker=dict(line=dict(width=0)),
        textposition="inside",
        textfont=dict(color="white"),
        texttemplate="%{x:,.2f}",
    )
    return fig


# =========================
# Интерфейс
# =========================
st.markdown(
    """
    <section class="tradeflow-hero">
        <div class="tradeflow-kicker">TradeFlow Analytics</div>
        <h1>Межрегиональная торговля — в цифрах и связях</h1>
        <p>
            Исследуйте товарные потоки между регионами, находите критические группы
            поставщиков и сравнивайте прямое и непрямое влияние в торговой сети.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📊 TradeFlow")
    st.caption("Аналитика межрегиональных товарных потоков")
    st.divider()

    st.subheader("1. Данные")
    uploaded_files = st.file_uploader(
        "Excel-файлы Росстата",
        type=["xlsx"],
        accept_multiple_files=True,
        help="Можно выбрать несколько файлов с одинаковой структурой — строки будут объединены.",
    )
    use_demo = st.checkbox(
        "Использовать демо-данные",
        value=True,
        disabled=bool(uploaded_files),
        help="Синтетический набор позволяет посмотреть интерфейс без загрузки файлов.",
    )

    st.subheader("2. Параметры модели")
    quota_mode_label = st.radio(
        "Как задать квоту",
        options=["Доля входящего потока", "Абсолютное значение"],
        index=0,
        help=(
            "Доля: группа S критична для региона i, если обеспечивает не менее α·100% его ввоза — "
            "регионы разного масштаба сравнимы. Абсолютное значение: порог q в единицах товара."
        ),
        key="quota_mode",
    )
    quota_mode = "share" if quota_mode_label == "Доля входящего потока" else "absolute"
    if quota_mode == "share":
        quota_pct = st.slider(
            "Квота α, % входящего потока",
            min_value=10,
            max_value=90,
            value=50,
            step=5,
            help="Критическая группа S для региона i: Σ поставок от S ≥ α · (весь ввоз региона i).",
            key="quota_pct",
        )
        quota_q = quota_pct / 100.0
    else:
        quota_q = st.number_input(
            "Квота q (в единицах товара)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="Критическая группа S для региона i: сумма входящих поставок от S ≥ q.",
            key="quota_q",
        )

    top_in_neighbors = st.number_input(
        "Крупнейших поставщиков",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
        help=(
            "Критические группы считаются по N крупнейшим входящим поставщикам. "
            "Число комбинаций растёт экспоненциально: 2ᴺ − 1."
        ),
        key="top_in_neighbors",
    )

    top_show = st.slider(
        "Регионов на графиках",
        min_value=5,
        max_value=40,
        value=15,
        step=5,
        help="Влияет только на графики, но не на расчёт индексов.",
        key="top_show",
    )

if uploaded_files:
    dfs = []
    load_errors = []
    for uploaded_file in uploaded_files:
        try:
            dfs.append(load_data(uploaded_file.getvalue()))
        except Exception as exc:
            load_errors.append(f"{uploaded_file.name}: {exc}")

    if load_errors:
        with st.expander("Не удалось прочитать часть файлов", expanded=True):
            for message in load_errors:
                st.error(message)
    if not dfs:
        st.error("Ни один файл не удалось прочитать. Проверьте структуру Excel-файлов.")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)
    source_label = f"Загружено файлов: {len(dfs)}"
elif use_demo:
    df = build_demo_data()
    source_label = "Демо-данные"
else:
    st.info(
        "Загрузите один или несколько Excel-файлов в боковой панели либо включите "
        "демо-данные, чтобы познакомиться с возможностями приложения."
    )
    with st.expander("Как должен быть устроен Excel-файл?"):
        st.markdown(
            """
            Приложение пропускает первые три строки. Следующие две строки должны содержать:

            1. товар с единицей измерения через запятую, например «Мука, тонна»
               (объединённые ячейки допустимы);
            2. регион-поставщик (вывоз).

            В первом столбце таблицы ожидается регион-получатель (ввоз), в остальных — объёмы поставок.
            Это стандартная ориентация таблиц Росстата: «по горизонтали — субъекты, откуда
            производится вывоз; по вертикали — субъекты, куда осуществляется ввоз».
            """
        )
    st.stop()

df = remove_aggregates(df)

if df.empty or len(df.columns) < 2:
    st.error("После удаления агрегатов в наборе не осталось данных для анализа.")
    st.stop()

df_long = build_long(df)

if df_long.empty:
    st.error("После очистки в наборе не осталось ненулевых поставок.")
    st.stop()

region_values = sorted(df_long["region_from"].dropna().astype(str).str.strip().unique().tolist())
ALL_OPTION = "Все регионы"
units = sorted(df_long["unit"].dropna().unique().tolist())
default_unit = units.index("тонна") if "тонна" in units else 0
with st.sidebar:
    st.subheader("3. Фильтры")
    unit_selected = st.selectbox(
        "Единица измерения",
        options=units,
        index=default_unit,
        help=(
            "Объёмы в разных единицах (тонны, тыс. условных банок, тыс. дкл) нельзя складывать, "
            "поэтому сеть строится по товарам одной единицы измерения."
        ),
        key="unit_selected",
    )
    selected_regions = st.multiselect(
        "Регионы-поставщики",
        options=[ALL_OPTION] + region_values,
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

df_long_unit = df_long[df_long["unit"] == unit_selected]
products = sorted(df_long_unit["product"].dropna().unique().tolist())
with st.sidebar:
    product_selected = st.selectbox(
        "Тип товара",
        options=["Все товары"] + products,
        index=0,
        help="«Все товары» — все товары выбранной единицы измерения.",
        key="product_selected",
    )
    st.divider()
    st.caption(f"Источник: {source_label}")

if product_selected != "Все товары":
    df_long_f = df_long_unit[df_long_unit["product"] == product_selected].copy()
else:
    df_long_f = df_long_unit.copy()

# фильтр по выбранным регионам-источникам
df_long_f = df_long_f[df_long_f["region_from"].isin([str(x).strip() for x in selected_regions_effective])]

if df_long_f.empty:
    st.info("После фильтрации по регионам/товару не осталось данных.")
    st.stop()

total_supply = float(df_long_f["value"].sum())
metric_cols = st.columns(4)
metric_cols[0].metric("Объём поставок", f"{format_tonnes(total_supply)} {unit_selected}")
metric_cols[1].metric("Торговых связей", f"{len(df_long_f):,}".replace(",", " "))
metric_cols[2].metric("Регионов-поставщиков", df_long_f["region_from"].nunique())
metric_cols[3].metric("Регионов-получателей", df_long_f["region_to"].nunique())

st.markdown(
    '<div class="tradeflow-note">Настройте квоту, число поставщиков и фильтры в боковой панели — расчёты обновятся автоматически.</div>',
    unsafe_allow_html=True,
)

# =========================
# Суммарные поставки
# =========================
st.header("Обзор товарных потоков")
st.caption("Суммарный объём поставок из выбранных регионов по регионам назначения.")

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
    value=False,
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
    marker_color="#74b9ff",
    texttemplate="%{text:,.0f}".replace(",", " "),
    hovertemplate="<b>%{y}</b><br>%{x:,.0f}".replace(",", " ") + f" {unit_selected}<extra></extra>",
)

fig_supply = apply_dark_style(
    fig_supply,
    height=min(980, 380 + 28 * len(partners_plot)),
    title=f"Куда направляются поставки: {len(partners_plot)} регионов",
    x_title=f"Сумма поставок, {unit_selected}",
    y_title="Регион-получатель",
)

st.plotly_chart(fig_supply, width="stretch")

# =========================
# Прямые индексы
# =========================
st.divider()
st.header("Прямое влияние")
st.subheader("Weighted Bundle Indices — wBI₁ и wBI₂")
quota_text = (
    f"не ниже {quota_q:.0%} входящего потока региона"
    if quota_mode == "share"
    else f"не ниже {format_tonnes(float(quota_q))} {unit_selected}"
)
st.caption(
    "wBI₁ оценивает значимость критических групп относительно входящего потока региона, "
    "а wBI₂ — относительно всей торговой сети. wBI₁ (норм.) = wBI₁ / 2^(m−1) лежит в [0, 1] "
    "и сопоставим для регионов с разным числом поставщиков. Критической считается группа "
    f"поставщиков с совокупным объёмом {quota_text}."
)

indices_df = compute_wbi_indices(
    df_long=df_long_f,
    quota_q=float(quota_q),
    top_in_neighbors=int(top_in_neighbors),
    quota_mode=quota_mode,
)

if indices_df.empty:
    st.info("Не удалось посчитать индексы: в сети нет положительных весов.")
    st.stop()

COLUMN_RENAME = {
    "region": "Регион",
    "wBI1": "wBI₁",
    "wBI1_norm": "wBI₁ (норм.)",
    "wBI2": "wBI₂",
    "in_weight": f"Суммарный ввоз, {unit_selected}",
    "quota": f"Квота q_i, {unit_selected}",
    "in_neighbors_used": "Число учтённых поставщиков",
    "in_neighbors": "Ключевые регионы-поставщики",
    "top1_supplier": "Крупнейший поставщик",
    "top1_share": "Доля крупнейшего поставщика",
    "min_group_size": "Размер минимальной критической группы",
    "critical_groups_count": "Число критических групп",
    "critical_groups_top3": "Критические группы",
    "critical_groups_top3_supply": "Суммарные поставки",
}

indices_df_view = indices_df.rename(columns=COLUMN_RENAME)
top3_table = (
    indices_df[["region", "critical_groups_top3", "critical_groups_top3_supply"]]
    .rename(columns={
        "region": "Регион",
        "critical_groups_top3": "Топ-3 критические группы",
        "critical_groups_top3_supply": "Суммарные поставки",
    })
)

with st.expander("Таблица индексов и критических групп"):
    st.dataframe(indices_df_view, width="stretch", hide_index=True)
    st.markdown("#### Топ-3 критические группы для каждого региона")
    st.dataframe(
        top3_table,
        width="stretch",
        hide_index=True,
        column_config={
            "Топ-3 критические группы": st.column_config.TextColumn(width="large"),
            "Суммарные поставки": st.column_config.TextColumn(width="small"),
        },
    )

top_show = min(int(top_show), len(indices_df))
plot_df = indices_df.head(top_show).sort_values("wBI1", ascending=True)
common_height = min(900, 420 + 26 * top_show)

fig_wbi = px.bar(
    plot_df,
    x="wBI1",
    y="region",
    orientation="h",
    text="wBI1",
)
fig_wbi.update_traces(
    marker_color="#74b9ff",
    texttemplate="%{text:.2f}",
    hovertemplate="<b>%{y}</b><br>wBI1=%{x:.3f}<extra></extra>",
)
fig_wbi = apply_dark_style(
    fig_wbi,
    height=common_height,
    title=f"Топ-{top_show} регионов по wBI1 (q={quota_q}, topN={top_in_neighbors})",
    x_title="wBI1",
    y_title="Регион",
)

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
    texttemplate="%{text:.2f}",
    hovertemplate="<b>%{y}</b><br>wBI2=%{x:.4f}<extra></extra>",
)
fig_wbi2 = apply_dark_style(
    fig_wbi2,
    height=common_height,
    title=f"Топ-{top_show} регионов по wBI2 (q={quota_q}, topN={top_in_neighbors})",
    x_title="wBI2",
    y_title="Регион",
)

tab_wbi1, tab_wbi2 = st.tabs(["wBI₁ · локальная зависимость", "wBI₂ · роль в общей сети"])
with tab_wbi1:
    st.plotly_chart(fig_wbi, width="stretch")
with tab_wbi2:
    st.plotly_chart(fig_wbi2, width="stretch")

# =========================
# Непрямое влияние 
# =========================
def _indirect_edges_maxmin_no_pruning(edges: pd.DataFrame, d: int = 2) -> pd.DataFrame:
    """
    Строим "скрытые" ребра длины d:
        w^(d)_{j->i} = max_{paths j -> ... -> i, length=d} min(weights along path)

    ВАЖНО: здесь НЕТ pruning (никаких head/top per source, никаких hard_cap).
    На плотных графах количество непрямых ребер может взорваться по памяти/времени.
    """

    if d < 2:
        raise ValueError("d must be >= 2 for indirect influence")

    base = edges[["region_from", "region_to", "w"]].copy()
    base = base[base["w"] > 0].copy()

    # paths: (src -> mid) с bottleneck bw
    paths = base.rename(columns={"region_from": "src", "region_to": "mid", "w": "bw"})

    for step in range(2, d + 1):
        nxt = paths.merge(
            base.rename(columns={"region_from": "mid", "region_to": "dst", "w": "w2"}),
            on="mid",
            how="inner",
        )

        # убираем петли src == dst
        nxt = nxt[nxt["src"] != nxt["dst"]]
        if nxt.empty:
            return pd.DataFrame(columns=["region_from", "region_to", "w"])

        # bottleneck на пути
        nxt["bw"] = nxt[["bw", "w2"]].min(axis=1)

        # из src в dst берем лучший bottleneck (max)
        nxt = (
            nxt.groupby(["src", "dst"], as_index=False)["bw"]
            .max()
            .rename(columns={"bw": "bw"})
        )

        if step < d:
            paths = nxt.rename(columns={"dst": "mid"})
        else:
            paths = nxt.rename(columns={"dst": "region_to", "bw": "w"})
            paths = paths.rename(columns={"src": "region_from"})

    return paths[["region_from", "region_to", "w"]]


def compute_wbi_indices_indirect_no_pruning(
    df_long: pd.DataFrame,
    quota_q: float,
    d: int = 2,
    top_in_neighbors: int = 15,
    quota_mode: str = "share",
    direct_in_weight: dict | None = None,
) -> pd.DataFrame:
    """
    wBI1/wBI2 на графе непрямого влияния длины d (БЕЗ pruning).

    Шаги:
      1) строим прямые ребра (агрегация),
      2) преобразуем их в ребра непрямого влияния длины d по max-min (без pruning),
      3) пересчитываем wBI1/wBI2 обычной функцией compute_wbi_indices.

    Квота — параметр вершины: при quota_mode="share" для региона i используется
    q_i = α · (прямой ввоз региона i), т.е. тот же порог в единицах товара, что и
    на прямом графе (сумма весов графа непрямого влияния физическим потоком не является).
    """
    edges = (
        df_long.groupby(["region_from", "region_to"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "w"})
    )
    edges = edges[edges["w"] > 0].copy()

    if edges.empty:
        return empty_indices_frame()

    edges_ind = _indirect_edges_maxmin_no_pruning(edges=edges, d=int(d))

    if edges_ind.empty:
        return empty_indices_frame()

    df_long_ind = edges_ind.rename(columns={"w": "value"})

    quota_map = None
    if quota_mode == "share":
        if direct_in_weight is None:
            direct_in_weight = edges.groupby("region_to")["w"].sum().to_dict()
        quota_map = {r: float(quota_q) * float(w) for r, w in direct_in_weight.items()}

    return compute_wbi_indices(
        df_long=df_long_ind,
        quota_q=float(quota_q),
        top_in_neighbors=int(top_in_neighbors),
        quota_mode=quota_mode,
        quota_map=quota_map,
    )


# UI для непрямого влияния
st.divider()
st.header("Непрямое влияние")
st.caption(
    "Показывает потенциальные связи через цепочки посредников. Вес непрямой связи — "
    "лучшее узкое место среди всех путей выбранной длины (max–min)."
)

use_indirect = st.checkbox(
    "Посчитать индексы wBI₁/wBI₂ на графе непрямого влияния",
    value=False,
    help=(
        "Строим ребра длины d по правилу: вес j→i = max по всем путям длины d от j к i "
        "(минимум ребер на пути). Затем пересчитываем wBI₁/wBI₂ уже на новом графе. "
        "В этой версии pruning отключен."
    ),
)

if use_indirect:
    col_a, col_b = st.columns([1, 2])
    with col_a:
        d_indirect = st.number_input("Длина пути", min_value=2, max_value=5, value=2, step=1)
    with col_b:
        st.warning(
            "Расчёт выполняется без отсечения путей и на плотной сети может занять заметное время.",
            icon="⚠️",
        )

    with st.spinner("Строим сеть непрямого влияния…"):
        indices_ind_df = compute_wbi_indices_indirect_no_pruning(
            df_long=df_long_f,
            quota_q=float(quota_q),
            d=int(d_indirect),
            top_in_neighbors=int(top_in_neighbors),
            quota_mode=quota_mode,
            direct_in_weight=dict(zip(indices_df["region"], indices_df["in_weight"])),
        )

    if indices_ind_df.empty:
        st.info("Не удалось посчитать непрямое влияние: нет путей нужной длины или веса нулевые.")
    else:
        indices_ind_view = indices_ind_df.rename(columns=COLUMN_RENAME)
        cmp = (
            indices_df[["region", "wBI1", "wBI2"]]
            .merge(
                indices_ind_df[["region", "wBI1", "wBI2"]],
                on="region",
                how="outer",
                suffixes=("_direct", "_indirect"),
            )
            .fillna(0.0)
        )
        cmp["Δ wBI₁"] = cmp["wBI1_indirect"] - cmp["wBI1_direct"]
        cmp["Δ wBI₂"] = cmp["wBI2_indirect"] - cmp["wBI2_direct"]
        cmp = cmp.sort_values("Δ wBI₁", ascending=False)

        cmp_view = cmp.rename(columns={
            "region": "Регион",
            "wBI1_direct": "wBI₁ (прямое)",
            "wBI2_direct": "wBI₂ (прямое)",
            "wBI1_indirect": "wBI₁ (непрямое)",
            "wBI2_indirect": "wBI₂ (непрямое)",
        })
        with st.expander("Сравнение прямого и непрямого влияния"):
            st.dataframe(cmp_view, width="stretch", hide_index=True)
            st.markdown("#### Индексы непрямого влияния")
            st.dataframe(indices_ind_view, width="stretch", hide_index=True)

        # Отбираем top-N отдельно для каждого индекса — заголовок и данные теперь совпадают.
        df_wbi_indirect = (
            cmp[cmp["wBI1_indirect"] > 0]
            .sort_values("wBI1_indirect", ascending=False)
            .head(top_show)
            .sort_values("wBI1_indirect", ascending=True)
        )
        df_wbi2_indirect = (
            cmp[cmp["wBI2_indirect"] > 0]
            .sort_values("wBI2_indirect", ascending=False)
            .head(top_show)
            .sort_values("wBI2_indirect", ascending=True)
        )

        tab_ind_1, tab_ind_2 = st.tabs(["wBI₁ · непрямое влияние", "wBI₂ · непрямое влияние"])

        with tab_ind_1:
            fig_wbi_indirect = px.bar(
                df_wbi_indirect,
                x="wBI1_indirect",
                y="region",
                orientation="h",
                text="wBI1_indirect",
            )
            fig_wbi_indirect.update_traces(
                marker_color="#74b9ff",
                texttemplate="%{text:.2f}",
                hovertemplate="<b>%{y}</b><br>wBI₁ (непрямое)=%{x:.3f}<extra></extra>",
            )
            fig_wbi_indirect = apply_dark_style(
                fig_wbi_indirect,
                height=min(900, 420 + 26 * len(df_wbi_indirect)),
                title=f"Топ-{len(df_wbi_indirect)} регионов по wBI₁ — непрямое влияние",
                x_title="wBI₁ (непрямое влияние)",
                y_title="Регион",
            )
            st.plotly_chart(fig_wbi_indirect, width="stretch")

        with tab_ind_2:
            fig_wbi2_indirect = px.bar(
                df_wbi2_indirect,
                x="wBI2_indirect",
                y="region",
                orientation="h",
                text="wBI2_indirect",
            )
            fig_wbi2_indirect.update_traces(
                marker_color="#74b9ff",
                texttemplate="%{text:.4f}",
                hovertemplate="<b>%{y}</b><br>wBI₂ (непрямое)=%{x:.4f}<extra></extra>",
            )
            fig_wbi2_indirect = apply_dark_style(
                fig_wbi2_indirect,
                height=min(900, 420 + 26 * len(df_wbi2_indirect)),
                title=f"Топ-{len(df_wbi2_indirect)} регионов по wBI₂ — непрямое влияние",
                x_title="wBI₂ (непрямое влияние)",
                y_title="Регион",
            )
            st.plotly_chart(fig_wbi2_indirect, width="stretch")

st.divider()
st.header("Данные и экспорт")
st.caption("Проверьте преобразованный набор и скачайте результаты текущего расчёта.")

with st.expander("Предпросмотр исходной таблицы"):
    max_preview_rows = min(300, len(df))
    if max_preview_rows <= 5:
        rows_to_show = max_preview_rows
    else:
        rows_to_show = st.slider(
            "Количество строк",
            min_value=5,
            max_value=max_preview_rows,
            value=min(25, max_preview_rows),
            step=5,
            key="rows_to_show",
        )
    st.dataframe(df.head(rows_to_show), width="stretch", hide_index=True)

download_left, download_right = st.columns(2)
with download_left:
    st.download_button(
        "Скачать индексы CSV",
        data=indices_df_view.to_csv(index=False).encode("utf-8-sig"),
        file_name="tradeflow_indices.csv",
        mime="text/csv",
        width="stretch",
    )
with download_right:
    st.download_button(
        "Скачать потоки CSV",
        data=df_long_f.to_csv(index=False).encode("utf-8-sig"),
        file_name="tradeflow_flows.csv",
        mime="text/csv",
        width="stretch",
    )

st.caption(
    "Демо-набор синтетический и предназначен только для знакомства с интерфейсом. "
    "Расчёты по загруженным файлам выполняются в памяти текущей сессии."
)
