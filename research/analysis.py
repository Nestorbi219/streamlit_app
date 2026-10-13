"""Полный пересчёт для статьи: таблицы (results.json) и рисунки (fig/*.png).

Запуск: python3 -I scripts/analysis.py  (из /home/claude/nirs)
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from load import APP_EXCLUDE, clean  # noqa: E402
from wbi import edges_from, indirect, wbi  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA, N, D = 0.5, 5, 2

RAW = ROOT / "raw_long.pkl"
if not RAW.exists():  # первый запуск: читаем исходные таблицы Росстата (срезы A и B)
    from load import read_long  # noqa: E402
    pickle.dump({"2": read_long(str(ROOT / "data/file2.xlsx")), "3": read_long(str(ROOT / "data/file3.xlsx"))}, open(RAW, "wb"))
dfs = pickle.load(open(RAW, "rb"))
cl = {k: clean(d) for k, d in dfs.items()}
both = pd.concat(cl.values(), ignore_index=True)
T = both[both["unit"] == "тонна"]
E = edges_from(T, "exporter", "importer")
R = {}


def rho(a, b, col="wBI1"):
    j = a[[col]].join(b[[col]], lsuffix="_a", rsuffix="_b", how="inner")
    return float(spearmanr(j.iloc[:, 0], j.iloc[:, 1]).statistic)


def top(r, col="wBI1", k=10):
    return set(r.sort_values(col, ascending=False).head(k).index)


def add_norm(r):
    m = r["n_used"].clip(lower=1)
    r["wBI1n"] = r["wBI1"] / (2.0 ** (m - 1))
    r["kmin"] = np.where(r["min_group"] == "", 0, r["min_group"].str.count(",") + 1)
    return r


# ---------- 1. Описание данных и сети ----------
nodes = sorted(set(E.u) | set(E.v))
outdeg = E.groupby("u").size()
indeg = E.groupby("v").size()
R["data"] = dict(
    files=len(cl),
    records_raw=int(sum(len(d) for d in dfs.values())),
    products_all=int(both["product"].nunique()),
    products_t=int(T["product"].nunique()),
    records_t=int(len(T)),
    volume_mt=float(T["value"].sum() / 1e6),
    nodes=len(nodes), edges=int(len(E)),
    density=float(len(E) / (len(nodes) * (len(nodes) - 1))),
    exporters=int(E.u.nunique()),
    no_export=sorted(set(nodes) - set(E.u)),
    indeg_mean=float(indeg.mean()), indeg_median=float(indeg.median()),
    indeg_min=int(indeg.min()), indeg_max=int(indeg.max()),
    reciprocity=float(
        len(E.merge(E, left_on=["u", "v"], right_on=["v", "u"])) / len(E)
    ),
    units=both.groupby("unit")["product"].nunique().to_dict(),
)
prod_share = (T.groupby("product")["value"].sum() / T["value"].sum()).sort_values(ascending=False)
R["prod_share_top"] = [(p, float(s)) for p, s in prod_share.head(5).items()]
exp_tot = (E.groupby("u")["w"].sum() / E["w"].sum()).sort_values(ascending=False)
imp_tot = (E.groupby("v")["w"].sum() / E["w"].sum()).sort_values(ascending=False)
R["top_exp"] = [(r, float(E[E.u == r].w.sum() / 1e6), float(s)) for r, s in exp_tot.head(7).items()]
R["top_imp"] = [(r, float(E[E.v == r].w.sum() / 1e6), float(s)) for r, s in imp_tot.head(7).items()]
R["top5_exp_share"] = float(exp_tot.head(5).sum())
R["top5_imp_share"] = float(imp_tot.head(5).sum())

# ---------- 2. Прямые индексы ----------
t0 = time.time()
base = add_norm(wbi(E, ALPHA, N, "rel").set_index("region"))
R["time_direct_N5"] = time.time() - t0
cols = ["wBI1", "wBI1n", "wBI2", "in_w", "n_in", "n_crit", "top1", "top1_share", "kmin", "min_group", "hhi"]
R["direct_top"] = base.sort_values("wBI1", ascending=False).head(15)[cols].reset_index().to_dict("records")
R["direct_bottom"] = base.sort_values("wBI1").head(8)[cols].reset_index().to_dict("records")
R["wbi2_top"] = base.sort_values("wBI2", ascending=False).head(8)[cols].reset_index().to_dict("records")
R["kmin_dist"] = base["kmin"].value_counts().sort_index().to_dict()
R["few_suppliers"] = base[base.n_in < N].sort_values("wBI1n", ascending=False)[cols].reset_index().to_dict("records")
R["n_lt_N"] = int((base.n_in < N).sum())
sp = base[base.kmin == 1].groupby("top1").size().sort_values(ascending=False)
R["single_crit_suppliers"] = sp.to_dict()
R["top1_counts"] = base["top1"].value_counts().head(6).to_dict()
R["altai_dep"] = base[base.top1 == "Алтайский край"].sort_values("top1_share", ascending=False)[["top1_share", "kmin"]].reset_index().to_dict("records")
R["corr_wbi1"] = {
    c: float(spearmanr(base["wBI1"], base[c]).statistic) for c in ["hhi", "top1_share", "n_in"]
}
R["corr_wbi1n"] = {
    c: float(spearmanr(base["wBI1n"], base[c]).statistic) for c in ["hhi", "top1_share", "n_in"]
}
R["corr_wbi1_wbi2"] = float(spearmanr(base["wBI1"], base["wBI2"]).statistic)
R["corr_wbi2_inw"] = float(spearmanr(base["wBI2"], base["in_w"]).statistic)

# Продуктовый состав ключевых пар
def pair(a, b):
    x = T[(T.exporter == a) & (T.importer == b)].groupby("product")["value"].sum()
    return [(p, float(v / x.sum())) for p, v in x.sort_values(ascending=False).head(3).items()]


R["pairs"] = {
    "Республика Крым→г. Севастополь": pair("Республика Крым", "г. Севастополь"),
    "Алтайский край→Республика Саха (Якутия)": pair("Алтайский край", "Республика Саха (Якутия)"),
    "Белгородская область→Курская область": pair("Белгородская область", "Курская область"),
    "Алтайский край→Кемеровская область — Кузбасс": pair("Алтайский край", "Кемеровская область — Кузбасс"),
}

# ---------- 3. Неверная ориентация осей (как в исходной версии приложения) ----------
raw_both = pd.concat(dfs.values(), ignore_index=True)
mask = (
    ~raw_both.exporter.str.lower().isin(APP_EXCLUDE)
    & ~raw_both.importer.str.lower().isin(APP_EXCLUDE)
    & ~raw_both.exporter.str.lower().str.contains("федеральный округ")
    & ~raw_both.importer.str.lower().str.contains("федеральный округ")
    & (raw_both.exporter != raw_both.importer)
)
app = raw_both[mask]
app_rev = edges_from(app[app.unit == "тонна"], "importer", "exporter")  # строки как источник
R["app_top_out"] = (app.groupby("importer")["value"].sum().sort_values(ascending=False).head(5).index.tolist())
rev = wbi(edges_from(T, "importer", "exporter"), ALPHA, N, "rel").set_index("region")
R["rho_reversed"] = rho(base, rev)
R["overlap_reversed"] = len(top(base) & top(rev))
# Сатурация при абсолютной квоте и смешении единиц (настройки по умолчанию исходной версии)
app_all = edges_from(app, "importer", "exporter")
sat = wbi(app_all, 1000.0, 5, "abs")
R["app_default_saturation"] = dict(
    share_at_16=float((sat.n_crit == 31).mean()),
    wbi1_median=float(sat.wBI1.median()),
    wbi1_q90=float(sat.wBI1.quantile(0.9)),
)

# ---------- 4. Непрямое влияние ----------
qmap = (ALPHA * base["in_w"]).to_dict()
ind = {}
for d in [2, 3]:
    t0 = time.time()
    Ed = indirect(E, d)
    rd = add_norm(wbi(Ed, None, N, q_map=qmap).set_index("region"))
    R[f"time_indirect_d{d}"] = time.time() - t0
    R[f"edges_d{d}"] = int(len(Ed))
    cmp = base[["wBI1", "wBI1n", "n_in", "top1", "top1_share"]].join(
        rd[["wBI1", "wBI1n", "n_in", "top1", "top1_share", "n_crit"]], rsuffix="_ind"
    )
    cmp["r_dir"] = cmp["wBI1"].rank(ascending=False, method="min")
    cmp["r_ind"] = cmp["wBI1_ind"].rank(ascending=False, method="min")
    cmp["dR"] = cmp["r_dir"] - cmp["r_ind"]
    ind[d] = cmp
    R[f"rho_dir_ind_d{d}"] = float(spearmanr(cmp.wBI1, cmp.wBI1_ind).statistic)
    R[f"zero_ind_d{d}"] = int((cmp.wBI1_ind == 0).sum())
c2 = ind[2]
big = c2[c2.n_in >= N]
R["ind_rise"] = big.sort_values("dR", ascending=False).head(10).reset_index().to_dict("records")
R["ind_drop"] = big.sort_values("dR").head(6).reset_index().to_dict("records")
R["ind_small"] = c2[c2.n_in < N].sort_values("dR", ascending=False).reset_index()[["region", "n_in", "r_dir", "r_ind", "dR", "top1_ind"]].to_dict("records")
R["ind_top1_counts"] = c2["top1_ind"].value_counts().head(6).to_dict()
risers2 = set(big.sort_values("dR", ascending=False).head(10).index)
b3 = ind[3][ind[3].n_in >= N]
risers3 = set(b3.sort_values("dR", ascending=False).head(10).index)
R["risers_overlap_d2_d3"] = len(risers2 & risers3)
R["rho_ind_d2_d3"] = float(spearmanr(ind[2].wBI1_ind, ind[3].wBI1_ind.reindex(ind[2].index)).statistic)

# ---------- 5. Чувствительность ----------
sens = []
for a in [0.3, 0.4, 0.6, 0.7]:
    r = wbi(E, a, N, "rel").set_index("region")
    sens.append(dict(param="α", value=a, rho1=rho(base, r), rho2=rho(base, r, "wBI2"), ov=len(top(base) & top(r))))
for n in [3, 4, 6, 8, 10]:
    t0 = time.time()
    r = wbi(E, ALPHA, n, "rel").set_index("region")
    sens.append(dict(param="N", value=n, rho1=rho(base, r), rho2=rho(base, r, "wBI2"), ov=len(top(base) & top(r)), t=time.time() - t0))
t0 = time.time()
wbi(E, ALPHA, 15, "rel")
R["time_N15"] = time.time() - t0
yr = {}
for k, c in cl.items():
    Ey = edges_from(c[c.unit == "тонна"], "exporter", "importer")
    yr[k] = wbi(Ey, ALPHA, N, "rel").set_index("region")
sens.append(dict(param="срез", value="A↔B", rho1=rho(yr["2"], yr["3"]), rho2=rho(yr["2"], yr["3"], "wBI2"), ov=len(top(yr["2"]) & top(yr["3"]))))
nb = both.copy()
nb["share"] = nb["value"] / nb.groupby("product")["value"].transform("sum")
rn = wbi(edges_from(nb, "exporter", "importer", "share"), ALPHA, N, "rel").set_index("region")
sens.append(dict(param="веса", value="равные доли товаров", rho1=rho(base, rn), rho2=rho(base, rn, "wBI2"), ov=len(top(base) & top(rn))))
f = T[~T["product"].str.startswith(("Зерно", "Комбикорма"))]
rf = wbi(edges_from(f, "exporter", "importer"), ALPHA, N, "rel").set_index("region")
sens.append(dict(param="веса", value="без зерна и комбикормов", rho1=rho(base, rf), rho2=rho(base, rf, "wBI2"), ov=len(top(base) & top(rf))))
R["sens"] = sens
R["products_norm"] = int(nb["product"].nunique())
R["years_edges"] = {k: int(len(edges_from(c[c.unit == "тонна"], "exporter", "importer"))) for k, c in cl.items()}

pickle.dump(dict(base=base, ind=ind), open(ROOT / "results.pkl", "wb"))
json.dump(R, open(ROOT / "results.json", "w"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in R.items() if k not in ("direct_top", "direct_bottom", "wbi2_top", "few_suppliers", "ind_rise", "ind_drop", "ind_small", "altai_dep")}, ensure_ascii=False, indent=1, default=str))
