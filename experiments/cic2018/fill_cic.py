"""Turn the CSE-CIC-IDS2018 result files into the numbers and table rows used in the manuscript (ms_v4)."""
import os
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from raafdl.stats import paired, holm

NAMES = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fltrust": "FLTrust", "raa": "RAA-FDL",
         "fedavg_ft": "FedAvg + server fine-tuning", "raa_n1000": "RAA-FDL, 1,000-record anchor",
         "raa_n200": "RAA-FDL, 200-record anchor", "raa_norare": "RAA-FDL, anchor without the rarer half of the classes",
         "raa_noA": "without $A$", "raa_noS": "without $S$", "raa_noC": "without $C$", "raa_noD": "without $D$",
         "raa_gamma1": "RAA-FDL, $\\gamma = 1$"}
SCN = {"iid": "IID", "dir1.0": "Dirichlet $\\alpha = 1.0$", "dir0.5": "Dirichlet $\\alpha = 0.5$",
       "dir0.1": "Dirichlet $\\alpha = 0.1$", "temporal": "Temporal (capture day)"}
SC = ["iid", "dir1.0", "dir0.5", "dir0.1", "temporal"]; ST = ["fedavg", "fedprox", "scaffold", "fltrust", "raa"]
F = {}
ms = lambda s: f"{s.mean():.3f} ± {s.std(ddof=1):.3f}"
fp = lambda p: f"{p:.4f}" if p >= 0.0001 else "< 0.0001"
ci = lambda r: f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"

info = json.load(open("prep_info.json")); F["prep"] = {k: info[k] for k in ["rows_in", "dropped_nan_inf", "dropped_duplicates", "n_features", "rows_out", "cap_per_class"]}
F["prep"]["n_const"] = len(info["dropped_constant_columns"]); F["prep"]["class_counts"] = info["class_counts"]

m = pd.read_csv("results/main/final.csv")
# Table: main results
rows = []
for st in ST:
    cells = [ms(m[(m.scenario == sc) & (m.strategy == st)].test_macro_f1) for sc in SC]
    rows.append(f"| {NAMES[st]} | " + " | ".join(cells) + " |")
F["main_rows"] = "\n".join(rows)
F["means"] = {sc: {st: round(m[(m.scenario == sc) & (m.strategy == st)].test_macro_f1.mean(), 3) for st in ST} for sc in SC}
F["sds"] = {sc: {st: round(m[(m.scenario == sc) & (m.strategy == st)].test_macro_f1.std(ddof=1), 3) for st in ST} for sc in SC}
# significance
sig = pd.read_csv("results/sig_main.csv")
rows = []
for sc in SC:
    for r in sig[sig.scenario == sc].to_dict("records"):
        b = "**" if r["holm"] < 0.05 else ""
        rows.append(f"| {SCN[sc]} | vs. {NAMES[r['baseline']]} | {r['mean_delta']:+.3f} | {ci(r)} | {r['wins']}/5 | {fp(r['t_p'])} | {b}{fp(r['holm'])}{b} |")
F["sig_rows"] = "\n".join(rows)
F["sig"] = {f"{r['scenario']}|{r['baseline']}": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in sig.to_dict("records")}
# diagnostics
dg = pd.read_csv("results/diag_main.csv")
F["diag_rows"] = "\n".join(
    f"| {SCN[sc]} | {g.jsd.mean():.2f} | {g.shift_auc.mean():.2f} | {int(g.missing.max())} |" for sc, g in
    [(s, dg[dg.scenario == s]) for s in SC])
F["diag"] = {sc: {"jsd": round(dg[dg.scenario == sc].jsd.mean(), 2), "auc": round(dg[dg.scenario == sc].shift_auc.mean(), 2)} for sc in SC}
F["n_train"] = int(dg.n_train.iloc[0]); F["anchor_full"] = int(dg.anchor.iloc[0]); F["majority"] = round(dg.majority.iloc[0], 3)
# early rounds (validation macro-F1 at rounds 10, 30, 50, 100), Dirichlet 0.5 and 0.1
c = pd.read_csv("results/curves_main.csv")
early = []
for sc in ["dir0.5", "dir0.1"]:
    for rd in [10, 20, 30, 50]:
        p = c[(c.scenario == sc) & (c["round"] == rd)].pivot_table(index="seed", columns="tag", values="v")
        r = paired(p["raa"], p["fedavg"])
        early.append(f"| {SCN[sc]} | {rd} | {p['fedavg'].mean():.3f} | {p['raa'].mean():.3f} | {r['mean_delta']:+.3f} | {ci(r)} | {r['wins']}/5 | {fp(r['t_p'])} |")
F["early_rows"] = "\n".join(early)
# anchor / loo / temporal_abl / unreliable / central
m3 = m[m.seed.isin([42, 43, 44])]
def block(path, scen, base_tags):
    a = pd.read_csv(path); d = pd.concat([a, m3[(m3.scenario == scen) & (m3.strategy.isin(base_tags))]])
    return d.pivot_table(index="seed", columns="strategy", values="test_macro_f1")
pa = block("results/anchor/final.csv", "dir0.5", ["raa", "fedavg", "fltrust"])
rows, rs = [], []
for t in ["fedavg", "fedavg_ft", "fltrust", "raa_n1000", "raa_n200", "raa_norare"]:
    r = paired(pa[t], pa["raa"]); r["t"] = t; rs.append(r)
h = holm([r["t_p"] for r in rs])
rows.append(f"| RAA-FDL, full anchor ({F['anchor_full']:,} records) | {ms(pa['raa'])} | – | – |")
for r, hp in zip(rs, h):
    rows.append(f"| {NAMES[r['t']]} | {ms(pa[r['t']])} | {r['mean_delta']:+.3f} {ci(r)} | {fp(hp)} |")
F["anchor_rows"] = "\n".join(rows); F["anchor_means"] = {k: round(v, 3) for k, v in pa.mean().items()}
F["fedavg_ft_seeds"] = [round(v, 3) for v in pa["fedavg_ft"].values]
pl = block("results/loo/final.csv", "dir0.5", ["raa"])
rows, rs = [], []
for t in ["raa_noA", "raa_noS", "raa_noC", "raa_noD"]:
    r = paired(pl[t], pl["raa"]); r["t"] = t; rs.append(r)
h = holm([r["t_p"] for r in rs])
betas = {"raa_noA": "(0, 1/3, 1/3, 1/3)", "raa_noS": "(0.5, 0, 0.25, 0.25)", "raa_noC": "(0.5, 0.25, 0, 0.25)", "raa_noD": "(0.5, 0.25, 0.25, 0)"}
rows.append(f"| Full RAA-FDL | (0.4, 0.2, 0.2, 0.2) | {ms(pl['raa'])} | – | – |")
for r, hp in zip(rs, h):
    rows.append(f"| RAA-FDL {NAMES[r['t']]} | {betas[r['t']]} | {ms(pl[r['t']])} | {r['mean_delta']:+.3f} {ci(r)} | {fp(hp)} |")
F["loo_rows"] = "\n".join(rows)
pt = block("results/temporal_abl/final.csv", "temporal", ["raa", "fedavg", "scaffold"])
F["temporal_abl"] = {k: [round(v.mean(), 3), round(v.std(ddof=1), 3)] for k, v in pt.items()}
rows = []
for k in [3, 5, 7]:
    u = pd.read_csv(f"results/unrel{k}/final.csv"); p = u.pivot_table(index="seed", columns="strategy", values="test_macro_f1")
    rC = paired(p["raa"], p["raa_noC"])
    rows.append(f"| {k} of 10 | {ms(p['fedavg'])} | {ms(p['fedprox'])} | {ms(p['scaffold'])} | {ms(p['raa'])} | {ms(p['raa_noC'])} | {rC['mean_delta']:+.3f} {ci(rC)} |")
F["unrel_rows"] = "\n".join(rows)
ce = pd.read_csv("results/central/final.csv")
F["central"] = {mdl: [round(g.macro_f1.mean(), 3), round(g.macro_f1.std(ddof=1), 3), round(g.accuracy.mean(), 3), int(len(g))] for mdl, g in ce.groupby("model")}
pc = {mdl: pd.DataFrame([json.loads(s) for s in g.per_class_f1]).mean() for mdl, g in ce.groupby("model")}
fa = m[(m.scenario == "iid") & (m.strategy == "fedavg")]; pc["fedavg_iid"] = pd.DataFrame([json.loads(s) for s in fa.per_class_f1]).mean()
ra = m[(m.scenario == "iid") & (m.strategy == "raa")]; pc["raa_iid"] = pd.DataFrame([json.loads(s) for s in ra.per_class_f1]).mean()
pcd = pd.DataFrame(pc)[["fedavg_iid", "raa_iid", "central_cnn", "central_hgb"]]
F["perclass_rows"] = "\n".join(f"| {cls} | {info['class_counts'][cls]:,} | " + " | ".join(f"{v:.2f}" for v in row) + " |" for cls, row in pcd.iterrows())
json.dump(F, open("fill_cic.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in F.items() if not k.endswith("_rows")}, indent=1, default=str)[:3000])
