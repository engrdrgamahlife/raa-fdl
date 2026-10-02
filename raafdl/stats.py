"""Paired statistics over seeds, as specified in Section 4.2 of the revised manuscript.

For each (scenario, baseline): mean +/- SD (sample SD, ddof=1, used consistently in every
table), mean paired difference, 95% CI of the difference, paired t-test, exact Wilcoxon
signed-rank test, Holm-Bonferroni correction within a family, and Cohen's d_z.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def paired(a, b):
    """a = proposed, b = baseline, aligned by seed."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    sd = d.std(ddof=1)
    t_res = stats.ttest_rel(a, b)
    try:
        w_p = float(stats.wilcoxon(a, b, method="exact").pvalue) if np.any(d != 0) else 1.0
    except TypeError:                                        # older SciPy
        w_p = float(stats.wilcoxon(a, b, mode="exact").pvalue) if np.any(d != 0) else 1.0
    tc = stats.t.ppf(0.975, n - 1)
    half = tc * sd / np.sqrt(n)
    return {
        "n": n, "mean_delta": d.mean(), "ci_low": d.mean() - half, "ci_high": d.mean() + half,
        "t_p": float(t_res.pvalue), "wilcoxon_p": w_p,
        "d_z": d.mean() / sd if sd > 0 else np.inf, "wins": int((d > 0).sum()),
    }


def compare(df, proposed="raa", metric="test_macro_f1", family_cols=("scenario",)):
    """df: one row per (scenario, strategy, seed) with the final-round metric.

    Holm correction is applied within each family (default: all comparisons of one scenario;
    pass family_cols=() to pool the whole table into one family, as in the paper's tables).
    """
    rows = []
    for scen, g in df.groupby("scenario"):
        piv = g.pivot_table(index="seed", columns="strategy", values=metric)
        for base in [c for c in piv.columns if c != proposed]:
            ok = piv[[proposed, base]].dropna()
            r = paired(ok[proposed], ok[base])
            r.update(scenario=scen, comparison=f"{proposed} vs {base}")
            rows.append(r)
    out = pd.DataFrame(rows)
    keys = list(family_cols)
    if keys:
        out["holm_p"] = out.groupby(keys)["t_p"].transform(lambda s: holm(s.values))
    else:
        out["holm_p"] = holm(out["t_p"].values)
    return out


def summary(df, metric="test_macro_f1"):
    return (df.groupby(["scenario", "strategy"])[metric]
              .agg(mean="mean", sd=lambda s: s.std(ddof=1), n="count").reset_index())
