"""Build the 5GAD manuscript text (datasets paragraph + Section 4.7) from the experiment outputs."""
import os
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from raafdl.stats import paired, holm

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/5gad100"
f = pd.read_csv(f"{OUT}/final.csv")
piv = f.pivot_table(index="seed", columns="strategy", values="test_macro_f1")
cen = pd.read_csv("results/5gad_central.csv")
d42 = json.load(open("results/5gad/diag_42.json"))
cc = np.array(d42["class_counts"])
names = {"fedavg": "FedAvg", "fedprox": "FedProx", "scaffold": "SCAFFOLD", "fltrust": "FLTrust", "raa": "RAA-FDL",
         "raa_noD": "RAA-FDL without $D$", "raa_gamma1": "RAA-FDL with $\\gamma = 1$"}
iface_desc = {"eno1": "eno1 (host NIC)", "enp5s0": "enp5s0 (host NIC)", "lo": "lo (SBI, loopback)",
              "upfgtp": "upfgtp (GTP-U tunnel)"}


def ms(s, k):
    return f"{s[k].mean():.3f} ± {s[k].std(ddof=1):.3f}"


def fmt_p(p):
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.1e}"


# ---- weights of the lo client at the last round
w = {}
last = {}
for l in open(f"{OUT}/rounds.jsonl"):
    r = json.loads(l)
    if "raa" in r:
        last[(r.get("tag", r["strategy"]), r["seed"])] = (r["round"], r["raa"]["2"]["w"])
for (tag, seed), (rnd, wl) in last.items():
    w.setdefault(tag, []).append(wl)
lo_share = cc[2].sum() / cc.sum()

# ---- partition table
part_rows = "\n".join(
    f"| {iface_desc[i]} | {cc[k, 0]:,} | {cc[k, 1]:,} | {cc[k, 1] / cc[k].sum():.1%} | {cc[k].sum() / cc.sum():.1%} |"
    for k, i in enumerate(d42["interfaces"]))

# ---- centralised table
cmap = {"majority": "Majority class", "ports_protocol_LR": "Logistic regression, ports and protocol only (7 features)",
        "payload_LR": "Logistic regression, payload bytes only (128 features)",
        "all_HGB": "Gradient-boosted trees, all 135 features"}
cen_rows = "\n".join(
    f"| {cmap[mdl]} | {g.macro_f1.mean():.3f} ± {g.macro_f1.std(ddof=1):.3f} | {g.acc.mean():.3f} ± {g.acc.std(ddof=1):.3f} | {g.recall.mean():.3f} |"
    for mdl, g in cen.groupby("model", sort=False))

# ---- federated table
order = ["fedavg", "fedprox", "scaffold", "fltrust", "raa"]
fed_rows = []
for s in order:
    g = f[f.strategy == s]
    fed_rows.append(f"| {names[s]} | {ms(g, 'test_macro_f1')} | {ms(g, 'test_accuracy')} | {ms(g, 'attack_recall')} | {ms(g, 'attack_precision')} |")
fed_rows = "\n".join(fed_rows)

# ---- paired table
rows = []
for b in ["fedavg", "fedprox", "scaffold", "fltrust"]:
    r = paired(piv["raa"], piv[b]); r["b"] = b; rows.append(r)
t = pd.DataFrame(rows); t["holm"] = holm(t.t_p.values)
sig_rows = "\n".join(
    f"| vs. {names[r.b]} | {r.mean_delta:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.wins}/5 | {fmt_p(r.t_p)} | {r.wilcoxon_p:.4f} | {'**' if r.holm < 0.05 else ''}{fmt_p(r.holm)}{'**' if r.holm < 0.05 else ''} |"
    for r in t.itertuples())

# ---- ablation table
rows = []
for a in ["raa_noD", "raa_gamma1"]:
    for b in ["raa", "fedavg"]:
        r = paired(piv[a], piv[b]); r["a"] = a; r["b"] = b; rows.append(r)
ta = pd.DataFrame(rows); ta["holm"] = holm(ta.t_p.values)
abl_rows = [f"| RAA-FDL (default) | {ms(f[f.strategy=='raa'], 'test_macro_f1')} | {np.mean(w['raa']):.2f} ± {np.std(w['raa'], ddof=1):.2f} | – | – |"]
for a in ["raa_noD", "raa_gamma1"]:
    r1 = ta[(ta.a == a) & (ta.b == "raa")].iloc[0]; r2 = ta[(ta.a == a) & (ta.b == "fedavg")].iloc[0]
    abl_rows.append(f"| {names[a]} | {ms(f[f.strategy==a], 'test_macro_f1')} | {np.mean(w[a]):.2f} ± {np.std(w[a], ddof=1):.2f} | "
                    f"{r1.mean_delta:+.3f} [{r1.ci_low:+.3f}, {r1.ci_high:+.3f}], {r1.wins}/5, Holm {fmt_p(r1.holm)} | "
                    f"{r2.mean_delta:+.3f} [{r2.ci_low:+.3f}, {r2.ci_high:+.3f}], Holm {fmt_p(r2.holm)} |")
abl_rows = "\n".join(abl_rows)

m = {s: f[f.strategy == s].test_macro_f1 for s in f.strategy.unique()}
out = dict(part_rows=part_rows, cen_rows=cen_rows, fed_rows=fed_rows, sig_rows=sig_rows, abl_rows=abl_rows,
           lo_share=f"{lo_share:.0%}", w_raa=f"{np.mean(w['raa']):.2f}", w_noD=f"{np.mean(w['raa_noD']):.2f}",
           w_g1=f"{np.mean(w['raa_gamma1']):.2f}", raa_min=f"{m['raa'].min():.3f}", raa_max=f"{m['raa'].max():.3f}",
           fedavg=f"{m['fedavg'].mean():.3f}", raa=f"{m['raa'].mean():.3f}", t=t, ta=ta,
           n_runs=len(f))
json.dump({k: v for k, v in out.items() if isinstance(v, str) or isinstance(v, int)}, open(f"{OUT}/fill.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if isinstance(v, (str, int))}, indent=1))
print(t.round(4).to_string()); print(ta.round(4).to_string())
