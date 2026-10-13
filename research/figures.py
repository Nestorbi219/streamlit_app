"""Рисунки для статьи (PNG, 300 dpi, светлый фон, пригодны для ч/б печати)."""
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fig"
OUT.mkdir(exist_ok=True)

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
S1 = "#2a78d6"  # серия 1 (синий)
S2 = "#eb6834"  # серия 2 (оранжевый)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

res = pickle.load(open(ROOT / "results.pkl", "rb"))
base = res["base"]
c2 = res["ind"][2]

# ---------- Рис. 1. Конвейер обработки ----------
fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")
row1 = [
    ("Таблицы Росстата\n(.xlsx, двухуровневая\nшапка)", "Загрузка"),
    ("Очистка: агрегаты,\nсоставные субъекты,\nпетли, нули, единицы", "Предобработка"),
    ("Long-формат\n(from, to, product,\nunit, value)", "Преобразование"),
    ("Ориентированный\nвзвешенный граф\nG = (V, E, W)", "Модель"),
]
row2 = [
    ("Квоты q_i = α·W_in(i);\nперебор подмножеств\nиз N крупнейших", "Критические группы"),
    ("wBI1, wBI1*, wBI2,\nk_min, доля\nкрупнейшего", "Прямые индексы"),
    ("Граф G_d (max–min)\nи пересчёт индексов\nс теми же q_i", "Непрямое влияние"),
    ("Таблицы, рейтинги,\nсравнение, экспорт\nCSV", "Визуализация"),
]
w, h = 23, 13.5
xs = [0.5, 25.8, 51.1, 76.4]
for row, y in ((row1, 23.5), (row2, 3)):
    for k, (txt, lab) in enumerate(row):
        x = xs[k]
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                             linewidth=0.9, edgecolor=INK2, facecolor="#f4f7fc" if row is row1 else "#fdf3ee")
        ax.add_patch(box)
        ax.text(x + w / 2, y + h - 2.3, lab, ha="center", va="center", fontsize=7.8, weight="bold", color=INK)
        ax.text(x + w / 2, y + (h - 3.5) / 2, txt, ha="center", va="center", fontsize=7.2, color=INK2, linespacing=1.25)
        if k < 3:
            ax.add_patch(FancyArrowPatch((x + w + 0.5, y + h / 2), (xs[k + 1] - 0.5, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=9, color=INK2, linewidth=0.9))
# переход со строки 1 на строку 2: ломаная линия в зазоре между строками
yb = 20.3
ax.plot([xs[3] + w / 2, xs[3] + w / 2, xs[0] + w / 2], [23.0, yb, yb], color=INK2, linewidth=0.9,
        linestyle=(0, (3, 2)))
ax.add_patch(FancyArrowPatch((xs[0] + w / 2, yb), (xs[0] + w / 2, 17.0), arrowstyle="-|>",
                             mutation_scale=9, color=INK2, linewidth=0.9))
fig.savefig(OUT / "fig1_pipeline.png")
plt.close(fig)

# ---------- Рис. 2. wBI1 и нормированный wBI1* против доли крупнейшего поставщика ----------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
few = base["n_in"] < 5
for ax, col, title in ((axes[0], "wBI1", "а) wBI1 (верхняя граница 2^(m−1) = 16 при m = 5)"),
                       (axes[1], "wBI1n", "б) нормированный wBI1* = wBI1 / 2^(m−1)")):
    ax.scatter(base.loc[~few, "top1_share"], base.loc[~few, col], s=22, color=S1,
               edgecolor="white", linewidth=0.6, label="5 и более поставщиков", zorder=3)
    ax.scatter(base.loc[few, "top1_share"], base.loc[few, col], s=30, color=S2, marker="D",
               edgecolor="white", linewidth=0.6, label="менее 5 поставщиков", zorder=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("Доля крупнейшего поставщика во ввозе региона")
axes[0].set_ylabel("Значение индекса")
axes[0].set_ylim(-0.5, 16.8)
axes[1].set_ylim(-0.03, 1.05)
for name, xy in (("Республика Алтай", (0.62, 5.2)), ("Ненецкий АО", (0.79, -0.1))):
    r = base.loc[name]
    axes[0].annotate(name, (r.top1_share, r.wBI1), xytext=xy, textcoords="data", va="center",
                     fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6,
                                                             shrinkA=1, shrinkB=3))
r = base.loc["г. Севастополь"]
axes[0].annotate("г. Севастополь", (r.top1_share, r.wBI1), xytext=(-70, -4), textcoords="offset points",
                 fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
axes[1].legend(loc="lower right", fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig2_bound.png")
plt.close(fig)

# ---------- Рис. 3. Ранги по wBI1: прямой граф и граф непрямого влияния (d = 2) ----------
big = c2[c2["n_in"] >= 5]
rise = big.sort_values("dR", ascending=False).head(8)
drop = big.sort_values("dR").head(6)
sel = list(rise.index) + list(drop.index)
fig, ax = plt.subplots(figsize=(7.2, 4.1))
ypos = list(range(len(sel)))[::-1]
for y, name in zip(ypos, sel):
    r = c2.loc[name]
    color = S2 if r.dR > 0 else S1
    ax.plot([r.r_dir, r.r_ind], [y, y], color=GRID, linewidth=2.2, zorder=1, solid_capstyle="round")
    ax.scatter([r.r_dir], [y], s=34, color="white", edgecolor=INK2, linewidth=1.1, zorder=3)
    ax.scatter([r.r_ind], [y], s=36, color=color, edgecolor="white", linewidth=0.6, zorder=4)
ax.set_yticks(ypos)
ax.set_yticklabels(sel, fontsize=8)
ax.set_xlim(0, 86)
ax.invert_xaxis()
ax.set_xlabel("Место в рейтинге уязвимости по wBI1 (1 — наиболее уязвимый регион)")
ax.grid(True, axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.axhline(len(drop) - 0.5, color=INK2, linewidth=0.5, linestyle=(0, (2, 2)))
ax.scatter([], [], s=34, color="white", edgecolor=INK2, linewidth=1.1, label="прямой граф G")
ax.scatter([], [], s=36, color=S2, label="граф G₂: место выше (рост уязвимости)")
ax.scatter([], [], s=36, color=S1, label="граф G₂: место ниже")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=7.2, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig3_indirect.png")
plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))
