"""Взвешенные bundle-индексы wBI1/wBI2 и граф непрямого влияния (max–min), как в v1.py."""
import itertools

import pandas as pd


def edges_from(df: pd.DataFrame, src: str, dst: str, val: str = "value") -> pd.DataFrame:
    e = df.groupby([src, dst], as_index=False)[val].sum()
    e.columns = ["u", "v", "w"]
    return e[e["w"] > 0].reset_index(drop=True)


def wbi(edges: pd.DataFrame, q, top_n: int = 5, q_mode: str = "abs", q_map=None) -> pd.DataFrame:
    """q_mode='abs' — квота в единицах веса; 'rel' — доля входящего потока вершины.
    q_map — словарь {вершина: квота} (переопределяет q_mode), например квоты прямого графа."""
    total = float(edges["w"].sum())
    nodes = sorted(set(edges["u"]) | set(edges["v"]))
    out = []
    for i in nodes:
        inc = edges[edges["v"] == i].sort_values("w", ascending=False)
        win = float(inc["w"].sum())
        if win <= 0:
            out.append(dict(region=i, wBI1=0.0, wBI2=0.0, in_w=0.0, n_in=0, n_used=0,
                            n_crit=0, top_share=0.0, hhi=0.0, top1="", top1_share=0.0,
                            min_group="", min_group_share=0.0))
            continue
        shares_all = inc["w"] / win
        hhi = float((shares_all ** 2).sum())
        top = inc.head(top_n)
        names = top["u"].tolist()
        ws = top["w"].astype(float).tolist()
        if q_map is not None:
            qi = q_map.get(i, float("inf"))
        else:
            qi = q * win if q_mode == "rel" else q
        s1 = s2 = 0.0
        crit = []
        for r in range(1, len(ws) + 1):
            for idx in itertools.combinations(range(len(ws)), r):
                s = sum(ws[t] for t in idx)
                if s >= qi:
                    s1 += s / win
                    s2 += s / total
                    crit.append((len(idx), -s, tuple(names[t] for t in idx), s))
        crit.sort()
        mg = crit[0] if crit else None
        out.append(dict(
            region=i, wBI1=s1, wBI2=s2, in_w=win, n_in=len(inc), n_used=len(ws),
            n_crit=len(crit), top_share=sum(ws) / win, hhi=hhi,
            top1=names[0], top1_share=ws[0] / win,
            min_group=", ".join(mg[2]) if mg else "",
            min_group_share=(mg[3] / win) if mg else 0.0,
        ))
    return pd.DataFrame(out)


def indirect(edges: pd.DataFrame, d: int = 2) -> pd.DataFrame:
    """w^(d)_{j->i} = max по путям длины d от j к i минимального веса ребра на пути."""
    base = edges[["u", "v", "w"]]
    paths = base.rename(columns={"u": "src", "v": "mid", "w": "bw"})
    for step in range(2, d + 1):
        nxt = paths.merge(base.rename(columns={"u": "mid", "v": "dst", "w": "w2"}), on="mid")
        nxt = nxt[nxt["src"] != nxt["dst"]]
        nxt["bw"] = nxt[["bw", "w2"]].min(axis=1)
        nxt = nxt.groupby(["src", "dst"], as_index=False)["bw"].max()
        paths = nxt.rename(columns={"dst": "mid"})
    return paths.rename(columns={"src": "u", "mid": "v", "bw": "w"})[["u", "v", "w"]]
