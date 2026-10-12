"""Revision numbers (tuned protocol) -> /home/claude/rev/ana/fill_rev.json and CSVs. Tolerates partial data."""
import sys, os, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
from raafdl.stats import paired, holm
OUT = "/home/claude/rev/ana"
CIC, G5, NI = "/home/claude/cic/results", "/home/claude/g5/results", "/home/claude/nidd/results"
SC = ["iid", "dir1.0", "dir0.5", "dir0.1", "temporal"]; SETS = SC + ["5gad", "site"]
SET_N = {"iid": "IID", "dir1.0": "Dir. 1.0", "dir0.5": "Dir. 0.5", "dir0.1": "Dir. 0.1", "temporal": "Per-day", "5gad": "5GAD-c", "site": "5G-NIDD"}
ORDER = ["fedavg", "fedavgm", "fedadam", "fedprox", "fltrust", "fltrust_noscale", "fltrust_root200", "raa", "feddisco",
         "scaffold", "scaffold_n", "fedavg_la", "fedrs", "scaffold_n_la", "full"]
NM = {"fedavg": "FedAvg", "fedavgm": "FedAvgM", "fedadam": "FedAdam", "fedprox": "FedProx", "fltrust": "FLTrust",
      "fltrust_noscale": "FLTrust, no rescaling", "fltrust_root200": "FLTrust, 200-record root", "raa": "RAA-FDL",
      "feddisco": "FedDisco", "scaffold": "SCAFFOLD", "scaffold_n": "SCAFFOLD-N", "fedavg_la": "FedAvg + LA", "fedrs": "FedRS",
      "scaffold_n_la": "SCAFFOLD-N + LA", "full": "SCAFFOLD-N + LA + RW", "la_multi": "FedAvg + LA (multi-class clients only)"}
FAM = {"fedavgm": "S", "fedadam": "S", "fltrust": "S", "fltrust_noscale": "S", "fltrust_root200": "S", "raa": "S", "feddisco": "S",
       "fedprox": "C", "scaffold": "C", "scaffold_n": "C", "fedavg_la": "C", "fedrs": "C", "scaffold_n_la": "C", "full": "C+S"}
def rd(p):
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
def load_main():
    a = rd(f"{CIC}/t_main/final.csv"); b = rd(f"{G5}/5gadc_t/final.csv"); c = rd(f"{NI}/main/final.csv")
    d = pd.concat([x[["scenario", "strategy", "seed", "test_macro_f1"]] for x in (a, b, c) if len(x)])
    return d[d.seed.isin([42, 43, 44, 45, 46])].drop_duplicates(["scenario", "strategy", "seed"], keep="last")
F = {}
m = load_main()
ms = lambda v: f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) == 5 else (f"({v.mean():.3f}, n={len(v)})" if len(v) else "-")
rows = []
for s in ORDER:
    cells = [ms(m[(m.scenario == sc) & (m.strategy == s)].test_macro_f1) for sc in SETS]
    rows.append(f"| {NM[s]} | " + " | ".join(cells) + " |")
F["main_rows"] = "\n".join(rows)
F["means"] = {sc: {s: round(m[(m.scenario == sc) & (m.strategy == s)].test_macro_f1.mean(), 3) for s in ORDER} for sc in SETS}
# paired vs FedAvg, Holm per setting, global Holm, TOST(+-0.02)
def tost(d, margin=0.02):
    n = len(d); se = d.std(ddof=1) / np.sqrt(n); df = n - 1
    if se == 0: return 0.0
    p1 = 1 - stats.t.cdf((d.mean() + margin) / se, df); p2 = stats.t.cdf((d.mean() - margin) / se, df)
    return float(max(p1, p2))
T = []
for sc in SETS:
    p = m[m.scenario == sc].pivot_table(index="seed", columns="strategy", values="test_macro_f1")
    if "fedavg" not in p: continue
    rs = []
    for s in ORDER[1:]:
        if s in p and p[[s, "fedavg"]].dropna().shape[0] == 5:
            q = p[[s, "fedavg"]].dropna(); r = paired(q[s], q["fedavg"]); r.update(setting=sc, strategy=s, tost_p=tost(q[s] - q["fedavg"])); rs.append(r)
    for r, h in zip(rs, holm([r["t_p"] for r in rs])): r["holm"] = h
    T += rs
T = pd.DataFrame(T)
if len(T):
    T["holm_global"] = holm(T.t_p.values); T.to_csv(f"{OUT}/paired_vs_fedavg.csv", index=False)
    mark = lambda r: ("\\*" if r.holm < 0.05 else "") + ("‡" if r.tost_p < 0.05 else "")
    vs = []
    for s in ORDER[1:]:
        cells = []
        for sc in SETS:
            q = T[(T.setting == sc) & (T.strategy == s)]
            cells.append("-" if not len(q) else f"{q.iloc[0].mean_delta:+.3f} ({q.iloc[0].wins}/5){mark(q.iloc[0])}")
        vs.append(f"| {NM[s]} | " + " | ".join(cells) + " |")
    F["vs_rows"] = "\n".join(vs)
    F["n_tests"] = int(len(T)); F["n_sig_holm"] = int((T.holm < 0.05).sum()); F["n_sig_global"] = int((T.holm_global < 0.05).sum())
    F["sig_list"] = T[T.holm < 0.05][["setting", "strategy", "mean_delta", "holm", "holm_global"]].round(4).to_dict("records")
    F["tost_list"] = T[T.tost_p < 0.05][["setting", "strategy", "mean_delta", "tost_p"]].round(4).to_dict("records")
json.dump(F, open(f"{OUT}/fill_rev.json", "w"), indent=1, default=str)
print(F["main_rows"]); print(F.get("vs_rows", ""))
print("tests", F.get("n_tests"), "holm sig", F.get("n_sig_holm"), "global", F.get("n_sig_global"))

# ---------------- per-setting robust summaries (median, collapsed seeds) for the 5G datasets
COL = {"5gad": 0.5, "site": 0.7}
rows = []
for s in ORDER:
    cells = []
    for sc in ["5gad", "site"]:
        v = m[(m.scenario == sc) & (m.strategy == s)].test_macro_f1
        cells.append("-" if not len(v) else f"{v.median():.3f} | {int((v < COL[sc]).sum())}/{len(v)}")
    rows.append(f"| {NM[s]} | " + " | ".join(cells) + " |")
F["g5_robust_rows"] = "\n".join(rows)
# ---------------- CCVR and operating-point metrics
def ph(path, sc_override=None):
    d = rd(path)
    if not len(d): return d
    if "round" in d: d = d[d["round"].fillna(100) == 100]
    if sc_override: d["scenario"] = sc_override
    return d
P = pd.concat([x for x in [ph(f"{CIC}/t_main/posthoc.csv"), ph(f"{G5}/5gadc_t/posthoc.csv"), ph(f"{NI}/main/posthoc.csv")] if len(x)])
P = P[P.seed.isin([42, 43, 44, 45, 46])].drop_duplicates(["scenario", "tag", "seed"], keep="last")
CC = ["fedavg", "fedavgm", "fedadam", "fedprox", "fltrust", "raa", "feddisco", "scaffold", "fedavg_la", "fedrs", "scaffold_n_la", "full"]
rows = []
for s in CC:
    cells = []
    for sc in SETS:
        q = P[(P.scenario == sc) & (P.tag == s)]
        cells.append("-" if len(q) < 5 else f"{q.test_macro_f1.mean():.3f} → {q.ccvr_test_macro_f1.mean():.3f}")
    rows.append(f"| {NM[s]} | " + " | ".join(cells) + " |")
F["ccvr_rows"] = "\n".join(rows)
ccv = []
for sc in SETS:
    for s in CC:
        q = P[(P.scenario == sc) & (P.tag == s)]
        if len(q) == 5:
            r = paired(q.ccvr_test_macro_f1.values, q.test_macro_f1.values); r.update(setting=sc, strategy=s); ccv.append(r)
ccv = pd.DataFrame(ccv)
if len(ccv):
    ccv["holm"] = holm(ccv.t_p.values); ccv.to_csv(f"{OUT}/ccvr_paired.csv", index=False)
    F["ccvr_fedavg"] = {sc: ccv[(ccv.setting == sc) & (ccv.strategy == "fedavg")][["mean_delta", "ci_low", "ci_high", "wins", "holm"]].round(4).to_dict("records") for sc in SETS}
op = []
for s in ["fedavg", "fedavgm", "fedadam", "raa", "fedavg_la", "scaffold"]:
    for sc in SC:
        q = P[(P.scenario == sc) & (P.tag == s)]
        if len(q) == 5:
            op.append(dict(strategy=s, scenario=sc, fpr=q.benign_fpr.mean(), dr=q.detection_rate.mean(), f1nat=q.macro_f1_nat.mean(),
                           c_fpr=q.ccvr_benign_fpr.mean(), c_dr=q.ccvr_detection_rate.mean(), c_f1nat=q.ccvr_macro_f1_nat.mean()))
op = pd.DataFrame(op); op.to_csv(f"{OUT}/operating.csv", index=False)
F["op"] = op.round(3).to_dict("records")
# ---------------- learnable-class macro-F1 (12 CIC classes: excludes the three classes with < 100 records)
EXC = ["DoS attacks-SlowHTTPTest", "FTP-BruteForce", "SQL Injection"]
a = rd(f"{CIC}/t_main/final.csv")
if len(a):
    a["f1_12"] = [np.mean([v for k, v in json.loads(x).items() if k not in EXC]) for x in a.per_class_f1]
    p12 = a.pivot_table(index="strategy", columns="scenario", values="f1_12")
    F["f1_12"] = p12.round(3).to_dict()
# ---------------- robust evaluation choices: mean test macro-F1 over rounds 85-100, and test at best-validation round
def robust(path):
    if not os.path.exists(path): return pd.DataFrame()
    R = [json.loads(l) for l in open(path)]
    d = pd.DataFrame([r for r in R if "test_macro_f1" in r and r.get("round", 0) % 5 == 0])
    d = d.drop_duplicates(["scenario", "tag", "seed", "round"], keep="last")
    out = []
    for (sc, t, sd), g in d.groupby(["scenario", "tag", "seed"]):
        if g["round"].max() < 100: continue
        g = g[g["round"] <= 100]
        out.append(dict(scenario=sc, strategy=t, seed=sd, last=g[g["round"] == 100].test_macro_f1.iloc[0],
                        last4=g[g["round"] >= 85].test_macro_f1.mean(), bestval=g.loc[g.val_macro_f1.idxmax()].test_macro_f1))
    return pd.DataFrame(out)
RB = pd.concat([x for x in [robust(f"{CIC}/t_main/rounds.jsonl"), robust(f"{G5}/5gadc_t/rounds.jsonl"), robust(f"{NI}/main/rounds.jsonl")] if len(x)])
RB = RB[RB.seed.isin([42, 43, 44, 45, 46])]
RB.to_csv(f"{OUT}/robust_eval.csv", index=False)
# do conclusions change? rank correlation of strategy means across evaluation choices, per setting
rc = {}
for sc in SETS:
    g = RB[(RB.scenario == sc) & (RB.strategy != "la_multi")].groupby("strategy")[["last", "last4", "bestval"]].mean()
    if len(g) > 3: rc[sc] = {"last4": round(stats.spearmanr(g["last"], g["last4"])[0], 3), "bestval": round(stats.spearmanr(g["last"], g["bestval"])[0], 3),
                              "top_last": g["last"].idxmax(), "top_last4": g["last4"].idxmax(), "top_bestval": g["bestval"].idxmax()}
F["robust_eval"] = rc
json.dump(F, open(f"{OUT}/fill_rev.json", "w"), indent=1, default=str)
print(F["ccvr_rows"]); print(json.dumps(rc, indent=0)); print(op.round(3).to_string())

# ======================= robustness and auxiliary tables
def ms5(v): return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) >= 2 else ("-" if not len(v) else f"{v.mean():.3f}")
mainc = rd(f"{CIC}/t_main/final.csv")
def base(sc, s): return mainc[(mainc.scenario == sc) & (mainc.strategy == s)].sort_values("seed")
def pdiff(a, b):
    a = a.sort_values("seed"); b = b.sort_values("seed"); common = sorted(set(a.seed) & set(b.seed))
    x = a.set_index("seed").loc[common].test_macro_f1.values; y = b.set_index("seed").loc[common].test_macro_f1.values
    return paired(x, y) if len(common) >= 3 else None
def fmt_d(r): return "-" if r is None else f"{r['mean_delta']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
ROB = {}
# Table 2 (partitions)
diag = rd(f"{CIC}/diag_main.csv"); bs = rd(f"{OUT}/benign_shift.csv")
prow = []
for sc in SC:
    g = diag[diag.scenario == sc]; b = bs[bs.scenario == sc]
    prow.append(f"| CSE-CIC-IDS2018, {SET_N[sc]} | 10 | {g.jsd.mean():.2f} | {g.shift_auc.mean():.2f} | {b.benign_auc.mean():.2f} |")
for sc, path, nm, k in [("5gad", f"{G5}/5gadc_t", "5GAD-c (interfaces)", 4), ("site", f"{NI}/main", "5G-NIDD (base stations)", 2)]:
    js, au = [], []
    for seed in [42, 43, 44, 45, 46]:
        f = f"{path}/diag_{seed}.json" if sc == "5gad" else f"{path}/diag_site_{seed}.json"
        if os.path.exists(f):
            dj = json.load(open(f)); cc = np.array(dj["class_counts"], float)
            P_ = cc / cc.sum(1, keepdims=True)
            from scipy.spatial.distance import jensenshannon
            js.append(np.mean([jensenshannon(P_[i] + 1e-12, P_[j] + 1e-12, base=2) for i in range(len(P_)) for j in range(i + 1, len(P_))]))
            if "feature_shift" in dj: au.append(dj["feature_shift"]["mean_auc"])
    prow.append(f"| {nm} | {k} | {np.mean(js):.2f} | {('%.2f' % np.mean(au)) if au else 'n/a'} | n/a |")
F["part_rows"] = "\n".join(prow)
# Table 3: submitted protocol vs tuned
old = pd.concat([rd(f"{CIC}/main/final.csv"), rd(f"{CIC}/v5/final.csv")])
trows = []
for s in ["fedavg", "fedprox", "scaffold", "fltrust", "raa", "fedavg_la"]:
    for lab, d in [("lr 0.01", old), ("tuned", mainc)]:
        trows.append(f"| {NM[s]} | {lab} | " + " | ".join(f"{d[(d.scenario == sc) & (d.strategy == s)].test_macro_f1.mean():.3f}" for sc in SC) + " |")
F["tune_rows"] = "\n".join(trows)
F["tune"] = {"fedavg_temporal_old": round(old[(old.scenario == "temporal") & (old.strategy == "fedavg")].test_macro_f1.mean(), 3),
             "fedavg_temporal_new": round(base("temporal", "fedavg").test_macro_f1.mean(), 3)}
ct = rd(f"{CIC}/central_tuned/final.csv")
if len(ct):
    ct["model"] = ct.strategy.str.extract(r"central_(\w+)_lr")[0]
    sel = {}
    for mdl, g in ct.groupby("model"):
        v = g.groupby("strategy").val_macro_f1.mean(); best = v.idxmax(); t = g[g.strategy == best]
        sel[mdl] = dict(config=best, test=round(t.test_macro_f1.mean(), 3), sd=round(t.test_macro_f1.std(ddof=1), 3), fpr=round(t.benign_fpr.mean(), 3))
    F["central"] = sel
# GN, MLP
for name, path, sets, tags in [("gn", f"{CIC}/t_gn/final.csv", ["dir0.1", "temporal"], ["fedavg", "fedavg_la", "scaffold"]),
                               ("mlp", f"{CIC}/t_mlp/final.csv", ["iid", "dir0.1", "temporal"], ["fedavg", "fedavgm_tuned", "fedadam_tuned", "raa", "feddisco", "fedavg_la", "scaffold", "fedavgm", "fedadam"])]:
    d = rd(path); rows = []
    for t in tags:
        cells = []
        for sc in sets:
            v = d[(d.scenario == sc) & (d.strategy == t)].test_macro_f1
            bt = t.replace("_tuned", "")
            cells.append(f"{ms5(v)} (CNN {base(sc, bt).test_macro_f1.mean():.3f})" if len(v) else "-")
        rows.append(f"| {NM.get(t.replace('_tuned', ''), t)}{' (server lr re-tuned for MLP)' if 'tuned' in t else (' (CNN server settings)' if t in ('fedavgm', 'fedadam') and name == 'mlp' else '')} | " + " | ".join(cells) + " |")
    F[f"{name}_rows"] = "\n".join(rows)
    if name == "gn":
        F["gn_vs_bn"] = {f"{sc}|{t}": {k: round(v, 4) if isinstance(v, float) else v for k, v in pdiff(d[(d.scenario == sc) & (d.strategy == t)], base(sc, t)).items()} for sc in sets for t in tags}
# LA sensitivity (per-day)
d = rd(f"{CIC}/t_las/final.csv")
F["las"] = {t: ms5(d[d.strategy == t].test_macro_f1) for t in d.strategy.unique()} if len(d) else {}
F["las"]["la_tau1_eps1"] = ms5(base("temporal", "fedavg_la").test_macro_f1)
# participation, K, poisoning
for name in ["t_part0.5", "t_K20", "t_K50", "t_poison1", "t_poison3", "t_chrono", "t_chrono_noport"]:
    d = rd(f"{CIC}/{name}/final.csv")
    if len(d):
        F[name] = {f"{sc}|{t}": ms5(g.test_macro_f1) for (sc, t), g in d.groupby(["scenario", "strategy"])}
        if name.startswith("t_chrono"):
            rs = {}
            for sc in d.scenario.unique():
                for t in d.strategy.unique():
                    if t != "fedavg":
                        r = pdiff(d[(d.scenario == sc) & (d.strategy == t)], d[(d.scenario == sc) & (d.strategy == "fedavg")])
                        if r: rs[f"{sc}|{t}"] = fmt_d(r) + f" {r['wins']}/{r['n']}"
            F[name + "_vsfedavg"] = rs
for k in [1, 3]:
    p = ph(f"{CIC}/t_poison{k}/posthoc.csv")
    if len(p): F[f"poison{k}_op"] = p.groupby("tag")[["benign_fpr", "detection_rate"]].mean().round(3).to_dict("index")
# extended training
d = rd(f"{CIC}/t_ext300/final.csv")
if len(d):
    R = [json.loads(l) for l in open(f"{CIC}/t_ext300/rounds.jsonl")]
    e = pd.DataFrame([r for r in R if "test_macro_f1" in r]).drop_duplicates(["scenario", "tag", "seed", "round"], keep="last")
    F["ext300"] = {f"{sc}|{t}": {str(rr): round(g[g["round"] == rr].test_macro_f1.mean(), 3) for rr in (100, 200, 300)} for (sc, t), g in e.groupby(["scenario", "tag"])}
json.dump(F, open(f"{OUT}/fill_rev.json", "w"), indent=1, default=str)
for k in ["part_rows", "tune_rows", "gn_rows", "mlp_rows"]: print(F.get(k)); print()
for k in ["central", "las", "t_part0.5", "t_K20", "t_K50", "t_poison1", "t_poison3", "poison1_op", "poison3_op", "gn_vs_bn", "tune"]: print(k, json.dumps(F.get(k), default=str)[:900])
