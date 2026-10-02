import os
import sys, json, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from raafdl.stats import paired, holm
f = pd.read_csv("results/main/final.csv")
SC = ["iid", "dir1.0", "dir0.5", "dir0.1", "temporal"]; ST = ["fedavg", "fedprox", "scaffold", "fltrust", "raa"]
out = {}
# paired RAA vs each baseline per scenario, Holm within scenario
rows = []
for sc in SC:
    piv = f[f.scenario == sc].pivot_table(index="seed", columns="strategy", values="test_macro_f1")
    rs = []
    for b in ["fedavg", "fedprox", "scaffold", "fltrust"]:
        r = paired(piv["raa"], piv[b]); r.update(scenario=sc, baseline=b); rs.append(r)
    h = holm([r["t_p"] for r in rs])
    for r, hp in zip(rs, h): r["holm"] = hp; rows.append(r)
sig = pd.DataFrame(rows); sig.to_csv("results/sig_main.csv", index=False)
print(sig[["scenario", "baseline", "mean_delta", "ci_low", "ci_high", "t_p", "holm", "wins", "d_z"]].round(4).to_string())
# curves: val macro-F1 per round; rounds to reach 90% of each scenario's best FedAvg final val
cur = {}
for l in open("results/main/rounds.jsonl"):
    r = json.loads(l)
    if "val_macro_f1" in r: cur[(r["scenario"], r["tag"], r["seed"], r["round"])] = r["val_macro_f1"]
c = pd.Series(cur).rename_axis(["scenario", "tag", "seed", "round"]).reset_index(name="v")
c.to_csv("results/curves_main.csv", index=False)
rt = []
for sc in SC:
    final_fa = c[(c.scenario == sc) & (c.tag == "fedavg") & (c["round"] == 100)].v.mean()
    target = 0.9 * final_fa
    for st in ST:
        for seed, g in c[(c.scenario == sc) & (c.tag == st)].groupby("seed"):
            g = g.sort_values("round"); hit = g[g.v >= target]
            rt.append(dict(scenario=sc, strategy=st, seed=seed, target=target, rounds=int(hit["round"].iloc[0]) if len(hit) else np.nan))
rt = pd.DataFrame(rt); rt.to_csv("results/rounds_to_target.csv", index=False)
print(rt.groupby(["scenario", "strategy"]).rounds.agg(["mean", "std", "count"]).round(1).unstack(0).to_string())
# diagnostics
d = []
for fn in glob.glob("results/main/diag_*.json"):
    j = json.load(open(fn)); sc, seed = fn.split("diag_")[1][:-5].rsplit("_", 1)
    d.append(dict(scenario=sc, seed=int(seed), jsd=j["js_distance_mean_pairwise"], shift_auc=j["feature_shift"]["mean_auc"],
                  missing=len(j["test_classes_missing_from_all_clients"]), majority=j["majority_test"]["macro_f1"], n_train=j["n_train"], anchor=j["anchor_size"]))
d = pd.DataFrame(d); d.to_csv("results/diag_main.csv", index=False)
print(d.groupby("scenario")[["jsd", "shift_auc", "missing", "majority", "n_train", "anchor"]].agg(["mean", "std"]).round(3).to_string())
