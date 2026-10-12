import json, re, numpy as np, pandas as pd
F = json.load(open("/home/claude/rev/ana/fill_rev.json")); C = json.load(open("/home/claude/cic/fill_cic.json"))
conf = json.load(open("/home/claude/cic/conflicts.json")); ni = json.load(open("/home/claude/nidd/prep_info.json"))
bs = pd.read_csv("/home/claude/rev/ana/benign_shift.csv")
s = open("/home/claude/jnsm6/ms6_template.md").read()
pr = C["prep"]
NM = {"fedavg": "FedAvg", "fedavgm": "FedAvgM", "fedadam": "FedAdam", "raa": "RAA-FDL", "fedavg_la": "FedAvg + LA", "scaffold": "SCAFFOLD",
      "fltrust": "FLTrust", "feddisco": "FedDisco", "fedprox": "FedProx"}
SN = {"iid": "IID", "dir1.0": "Dir. 1.0", "dir0.5": "Dir. 0.5", "dir0.1": "Dir. 0.1", "temporal": "Per-day"}
rep = {"{prep[rows_in]:,}": f"{pr['rows_in']:,}", "{prep[dropped_nan_inf]:,}": f"{pr['dropped_nan_inf']:,}",
       "{prep[dropped_duplicates]:,}": f"{pr['dropped_duplicates']:,}", "{prep[n_const]}": str(pr["n_const"]),
       "{prep[n_features]}": str(pr["n_features"]), "{prep[cap_per_class]:,}": f"{pr['cap_per_class']:,}", "{prep[rows_out]:,}": f"{pr['rows_out']:,}",
       "{conf[n_feature_vectors_with_conflicting_labels]:,}": f"{conf['n_feature_vectors_with_conflicting_labels']:,}",
       "{conf[n_records_involved]:,}": f"{conf['n_records_involved']:,}", "{conf[benign_infil]:,}": f"{conf['class_pairs']['Benign | Infilteration']:,}",
       "{conf[slow_ftp]}": str(conf["class_pairs"]["DoS attacks-SlowHTTPTest | FTP-BruteForce"]),
       "{conf[kept_records_with_conflicting_labels]}": str(conf["kept_records_with_conflicting_labels"]),
       "{conf[kept_ftp]}": str(conf["kept_by_class"]["FTP-BruteForce"]), "{conf[kept_slow]}": str(conf["kept_by_class"]["DoS attacks-SlowHTTPTest"]),
       "{shift[temporal]}": f"{bs[bs.scenario == 'temporal'].benign_auc.mean():.2f}",
       "{tune[fedavg_temporal_old]}": f"{F['tune']['fedavg_temporal_old']:.3f}", "{tune[fedavg_temporal_new]}": f"{F['tune']['fedavg_temporal_new']:.3f}",
       "{central[cnn]}": f"{F['central']['cnn']['test']:.3f}", "{central[mlp]}": f"{F['central']['mlp']['test']:.3f}",
       "{central[hgb]}": f"{C['central']['central_hgb'][0]:.3f}", "{central[cnn_fpr]}": f"{F['central']['cnn']['fpr']:.2f}"}
sc_ = ni["site_class"]; tot = [sum(sc_[c][str(k)] for c in sc_) for k in (0, 1)]
for k in (0, 1):
    rep[f"{{nidd[b{k}]}}"] = f"{sc_['Benign'][str(k)] / tot[k]:.0%}"; rep[f"{{nidd[u{k}]}}"] = f"{sc_['UDPFlood'][str(k)] / tot[k]:.0%}"
for k, v in rep.items():
    assert k in s, k; s = s.replace(k, v)
s = s.replace("{T:part_rows}", F["part_rows"]).replace("{T:tune_rows}", F["tune_rows"])
for k in ["main_rows", "vs_rows", "g5_robust_rows", "ccvr_rows"]: s = s.replace("{R:%s}" % k, F[k])
# operating-point table
op = pd.DataFrame(F["op"]); rows = []
for t in ["fedavg", "fedavgm", "fedadam", "raa", "fedavg_la", "scaffold"]:
    for sc in ["iid", "dir0.5", "temporal"]:
        r = op[(op.strategy == t) & (op.scenario == sc)].iloc[0]
        rows.append(f"| {NM[t]} | {SN[sc]} | {r.fpr:.3f} | {r.dr:.3f} | {r.f1nat:.3f} | {r.c_fpr:.3f} / {r.c_dr:.3f} / {r.c_f1nat:.3f} |")
s = s.replace("{R:op_rows}", "\n".join(rows))
# model / normalisation table
mlp = pd.read_csv("/home/claude/cic/results/t_mlp/final.csv"); gn = pd.read_csv("/home/claude/cic/results/t_gn/final.csv"); mc = pd.read_csv("/home/claude/cic/results/t_main/final.csv")
ms = lambda v: f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
def cell(d, sc, t, bt=None):
    v = d[(d.scenario == sc) & (d.strategy == t)].test_macro_f1
    if not len(v): return "-"
    b = mc[(mc.scenario == sc) & (mc.strategy == (bt or t))].test_macro_f1.mean()
    return f"{ms(v)} ({b:.3f})"
rows = []
for t in ["fedavg", "fedavg_la", "scaffold"]:
    rows.append(f"| GroupNorm network | {NM[t]} | - | {cell(gn, 'dir0.1', t)} | {cell(gn, 'temporal', t)} |")
for t, lab in [("fedavg", "FedAvg"), ("fedavgm_tuned", "FedAvgM, server lr 0.3"), ("fedadam_tuned", "FedAdam, server lr 0.01"), ("raa", "RAA-FDL"),
               ("feddisco", "FedDisco"), ("fedavg_la", "FedAvg + LA"), ("scaffold", "SCAFFOLD"), ("fedavgm", "FedAvgM, network's server lr (1.0)"), ("fedadam", "FedAdam, network's server lr (0.1)")]:
    bt = t.replace("_tuned", "")
    rows.append(f"| MLP | {lab} | {cell(mlp, 'iid', t, bt)} | {cell(mlp, 'dir0.1', t, bt)} | {cell(mlp, 'temporal', t, bt)} |")
s = s.replace("{R:gn_mlp_rows}", "\n".join(rows))
s = s.replace("| Strategy | Setting | Result |\n|:--|:--|:--|", "| Model | Strategy | IID | Dir. 0.1 | Per-day |\n|:--|:--|:--:|:--:|:--:|")
# poisoning table
rows = []
for t in ["fedavg", "fedavgm", "fedavg_la", "fltrust", "raa"]:
    base = mc[(mc.scenario == "dir0.5") & (mc.strategy == t)].test_macro_f1
    dr = " / ".join(f"{F[f'poison{k}_op'][t]['detection_rate']:.3f}" for k in (1, 3))
    fp = " / ".join(f"{F[f'poison{k}_op'][t]['benign_fpr']:.3f}" for k in (1, 3))
    rows.append(f"| {NM[t]} | {ms(base)} | {F['t_poison1']['dir0.5|' + t]} | {F['t_poison3']['dir0.5|' + t]} | {fp} | {dr} |")
s = s.replace("{R:poison_rows}", "\n".join(rows))
# tuning tables
tc = pd.read_csv("/home/claude/rev/ana/tuning_cic.csv").set_index("tag"); t1 = pd.read_csv("/home/claude/rev/ana/tuning_cic_lr001.csv").set_index("tag")
L = ["0.003", "0.01", "0.03", "0.1", "0.3", "1.0"]
def g(t, lr):
    if lr == "0.01": return f"{t1.loc[t, 'avg']:.3f}"
    k = f"{t}_lr{lr}"; return f"{tc.loc[k, 'avg']:.3f}" if k in tc.index else "-"
tt = ["| Strategy | " + " | ".join(f"lr {x}" for x in L) + " |", "|:--|" + ":--:|" * len(L)]
for t in ["fedavg", "fedavg_la", "scaffold"]: tt.append(f"| {NM[t]} | " + " | ".join(g(t, x) for x in L) + " |")
tt.append("| Mean of the three | " + " | ".join(f"{np.mean([float(g(t, x)) for t in ['fedavg', 'fedavg_la', 'scaffold']]):.3f}" for x in L) + " |")
srv = ["", "Table: Table A2. CSE-CIC-IDS2018, server settings and FedProx at local learning rate 0.3 (validation macro-F1, mean of the Dirichlet α = 0.5 and per-day partitions).", "",
       "| Setting | Validation macro-F1 |", "|:--|:--:|"]
for k, lab in [("fedavgm_slr0.1_l0.3", "FedAvgM, server lr 0.1"), ("fedavgm_slr0.3_l0.3", "FedAvgM, server lr 0.3"), ("fedavgm_slr1.0_l0.3", "FedAvgM, server lr 1.0 (selected)"),
               ("fedadam_slr0.01_l0.3", "FedAdam, server lr 0.01"), ("fedadam_slr0.03_l0.3", "FedAdam, server lr 0.03"), ("fedadam_slr0.1_l0.3", "FedAdam, server lr 0.1 (selected)"),
               ("fedprox_mu0.001_l0.3", "FedProx, μ = 0.001 (selected)"), ("fedprox_mu0.01_l0.3", "FedProx, μ = 0.01"), ("fedprox_mu0.1_l0.3", "FedProx, μ = 0.1 (per-day: seeds 42 and 43 only)")]:
    srv.append(f"| {lab} | {tc.loc[k, 'avg']:.3f} |")
o = []
for fn, lab, cap in [("tuning_nidd.csv", "5G-NIDD", "A3"), ("tuning_5gadc.csv", "5GAD-c", "A4")]:
    d = pd.read_csv(f"/home/claude/rev/ana/{fn}").set_index("st"); cols = list(d.columns)
    o += ["", f"Table: Table {cap}. {lab}, local learning rate (validation macro-F1, mean over seeds 42 to 44).", "",
          "| Strategy | " + " | ".join(f"lr {c}" for c in cols) + " |", "|:--|" + ":--:|" * len(cols)]
    for t in d.index: o.append(f"| {NM[t]} | " + " | ".join(f"{d.loc[t, c]:.3f}" for c in cols) + " |")
    o.append("| Mean of the three | " + " | ".join(f"{d[c].mean():.3f}" for c in cols) + " |")
d = pd.read_csv("/home/claude/rev/ana/tuning_mlp_server.csv").set_index("tag")
o += ["", "Table: Table A5. MLP on CSE-CIC-IDS2018, server learning rates (local learning rate 0.3; validation macro-F1, mean of the Dirichlet α = 0.5 and per-day partitions).", "",
      "| Setting | Validation macro-F1 |", "|:--|:--:|"] + [f"| {k.replace('fedavgm_slr', 'FedAvgM, server lr ').replace('fedadam_slr', 'FedAdam, server lr ')} | {d.loc[k, 'avg']:.3f} |" for k in d.index]
s = s.replace("{R:tuning_tables}", "\n".join(tt + srv + o))
# appendix B
rb = F["robust_eval"]; f12 = pd.DataFrame(F["f1_12"])
NM2 = {"fedavg": "FedAvg", "fedavgm": "FedAvgM", "fedadam": "FedAdam", "fedprox": "FedProx", "fltrust": "FLTrust", "fltrust_noscale": "FLTrust, no rescaling",
       "fltrust_root200": "FLTrust, 200-record root", "raa": "RAA-FDL", "feddisco": "FedDisco", "scaffold": "SCAFFOLD", "scaffold_n": "SCAFFOLD-N",
       "fedavg_la": "FedAvg + LA", "fedrs": "FedRS", "scaffold_n_la": "SCAFFOLD-N + LA", "full": "SCAFFOLD-N + LA + RW"}
SETN = dict(SN, **{"5gad": "5GAD-c", "site": "5G-NIDD"})
b = ["Table B1 asks whether the ranking of strategies depends on when the test score is taken. For each setting it gives the Spearman correlation between the strategies' mean test macro-F1 after the last round and (a) their mean over rounds 85 to 100, (b) their test macro-F1 at the round with the best validation score, together with the top strategy under each choice.", "",
     "Table: Table B1. Sensitivity of the ranking to the evaluation round (Spearman correlation with the last-round ranking; top strategy under each choice).", "",
     "| Setting | Rounds 85 to 100 | Best validation round | Top: last round | Top: rounds 85 to 100 | Top: best validation round |", "|:--|:--:|:--:|:--|:--|:--|"]
for k, v in rb.items():
    b.append(f"| {SETN[k]} | {v['last4']:.2f} | {v['bestval']:.2f} | {NM2[v['top_last']]} | {NM2[v['top_last4']]} | {NM2[v['top_bestval']]} |")
b += ["", "Table: Table B2. Learnable-class macro-F1 on CSE-CIC-IDS2018 (12 classes; mean over five seeds).", "",
      "| Strategy | IID | Dir. 1.0 | Dir. 0.5 | Dir. 0.1 | Per-day |", "|:--|:--:|:--:|:--:|:--:|:--:|"]
for t in NM2:
    if t in f12.index: b.append(f"| {NM2[t]} | " + " | ".join(f"{f12.loc[t, sc]:.3f}" for sc in ["iid", "dir1.0", "dir0.5", "dir0.1", "temporal"]) + " |")
s = s.replace("{R:robust_eval prose and learnable-class table}", "\n".join(b))

# chrono prose
ch, cv = F["t_chrono"], F["t_chrono_vsfedavg"]; nh, nv = F["t_chrono_noport"], F["t_chrono_noport_vsfedavg"]
m = lambda d, k: d[k].split(" ±")[0]
def ci(x):
    d, rest = x.split(" [", 1); c, w = rest.split("] "); n = w.split("/")[0]
    return d, "[" + c + "]", {"5": "all five", "4": "four of five", "3": "three of five", "2": "two of five", "1": "one of five", "0": "none of five"}[n]
a = ci(cv["iid|fedavgm"]); b = ci(cv["temporal|fedavgm"]); c = ci(nv["iid|fedavgm"]); d = ci(nv["temporal|fedavgm"])
F["chrono prose"] = (f"With the conflicting-label records removed and the per-class chronological split (Section 4.1), FedAvgM's lead over FedAvg held. "
  f"Under IID data FedAvg reached {ch['iid|fedavg']}, FedAvgM {ch['iid|fedavgm']} ({a[0]}, 95% CI {a[1]}, {a[2]} seeds), RAA-FDL {m(ch,'iid|raa')} and FedAvg + LA {m(ch,'iid|fedavg_la')}. "
  f"Under the per-day partition the corresponding values were {m(ch,'temporal|fedavg')}, {m(ch,'temporal|fedavgm')} ({b[0]}, 95% CI {b[1]}, {b[2]} seeds), {m(ch,'temporal|raa')} and {m(ch,'temporal|fedavg_la')}. "
  f"Removing the destination port as well left the picture unchanged: FedAvgM led FedAvg by {c[0]} {c[1]} under IID data ({m(nh,'iid|fedavgm')} against {m(nh,'iid|fedavg')}) and by {d[0]} {d[1]} under the per-day partition ({m(nh,'temporal|fedavgm')} against {m(nh,'temporal|fedavg')}), "
  f"and logit adjustment stayed slightly below FedAvg ({ci(nv['iid|fedavg_la'])[0]} and {ci(nv['temporal|fedavg_la'])[0]}). "
  f"RAA-FDL moved slightly above FedAvg ({cv['iid|raa'].split(' [')[0]} and {cv['temporal|raa'].split(' [')[0]}, both confidence intervals including zero), where it had been below it in the main runs ({F['means']['iid']['raa'] - F['means']['iid']['fedavg']:+.3f} and {F['means']['temporal']['raa'] - F['means']['temporal']['fedavg']:+.3f}), so the order of strategies that tie with FedAvg can change, as expected for differences within noise. "
  "The comparison between families therefore does not rest on flows from one session falling on both sides of a random split, or on the port feature.")
for k in ["{R:ext300 prose}", "{R:chrono prose}"]:
    if k in s and k[3:-1] in F: s = s.replace(k, F[k[3:-1]])
s = s.replace("| – |", "| - |").replace("–", "-").replace("—", ", ")
left = re.findall(r"\{(?:R:|T:|prep|conf|shift|nidd|central|tune)[^}]*\}", s)
open("/home/claude/jnsm6/ms6.md", "w").write(s)
print("left:", left); print("abstract words:", len(re.search(r"\*\*Abstract\.\*\* (.*?)\n", s).group(1).split()))

# ---------- ext300 (training length) ----------
import scipy.stats as st
E = [json.loads(l) for l in open("/home/claude/cic/results/t_ext300/rounds.jsonl")]
E = pd.DataFrame([r for r in E if "test_macro_f1" in r]).drop_duplicates(["tag", "seed", "round"], keep="last")
done = E[E["round"] == 300].groupby("tag").seed.nunique()
ext_ok = len(done) == 4 and (done == 5).all()
def pci(x):
    x = np.asarray(x); mu = x.mean(); h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return mu, mu - h, mu + h
ex = {}
for t in ["fedavg", "fedavgm", "fedavg_la", "scaffold"]:
    g = E[E.tag == t].pivot(index="seed", columns="round", values="test_macro_f1")
    if 300 not in g.columns: ex[t] = dict(n=0); continue
    g = g[g[300].notna()]
    late = g[[c for c in g.columns if c >= 205]].mean(axis=1)
    ex[t] = dict(n=len(g), r100=g[100].mean(), r200=g[200].mean(), r300=g[300].mean(), d=pci(g[300] - g[100]), late=late.mean(), best=g[[c for c in g.columns if c > 100]].max(axis=1).mean())
f3 = lambda v: f"{v:.3f}"; sg = lambda v: f"{v:+.3f}"
PH = pd.read_csv("/home/claude/cic/results/t_main/posthoc.csv"); ccv_pd = PH[(PH.scenario == "temporal") & (PH.tag == "fedavg")].ccvr_test_macro_f1.mean()
if all(ex[t]["n"] for t in ex):
    ordr = sorted(ex, key=lambda t: -ex[t]["r300"])
    txt = ("To test whether the per-day collapse is a matter of training length, we continued the round-100 checkpoints of FedAvg, FedAvgM, FedAvg + LA and SCAFFOLD to 300 rounds (Fig. 3a). "
           "Longer training did not recover the setting, although it helped some strategies a little. Mean test macro-F1 at rounds 100, 200 and 300 was " +
           "; ".join(f"{NM[t]} {f3(ex[t]['r100'])}, {f3(ex[t]['r200'])} and {f3(ex[t]['r300'])}" for t in ex) + ". "
           "The change from round 100 to round 300 (95% CI over seeds) was " + ", ".join(f"{sg(ex[t]['d'][0])} [{sg(ex[t]['d'][1])}, {sg(ex[t]['d'][2])}] for {NM[t]}" for t in ex) + "; only the gain for FedAvg + LA excludes zero, and it still leaves that strategy at " + f3(ex["fedavg_la"]["r300"]) + ". "
           f"Even the best test score any run reached after round 100, which no deployed system could pick, averaged at most {f3(max(ex[t]['best'] for t in ex))}, far below the {ccv_pd:.3f} that CCVR gives FedAvg at round 100. "
           "The curves oscillate rather than rise: label-disjoint clients pull the shared classifier in turn towards their own classes, and more rounds repeat the cycle.")
    F["ext300 prose"] = txt; s = open("/home/claude/jnsm6/ms6.md").read().replace("{R:ext300 prose}", txt); open("/home/claude/jnsm6/ms6.md", "w").write(s)
    F["ext300 response"] = ("We continued the per-day runs of FedAvg, FedAvgM, FedAvg + LA and SCAFFOLD to 300 rounds (Section 5.5, Fig. 3a). The curves oscillate rather than rise. From round 100 to 300 the change was " +
           ", ".join(f"{sg(ex[t]['d'][0])} for {NM[t]}" for t in ex) + "; only the FedAvg + LA gain has a 95% confidence interval excluding zero, and every strategy stayed between " + f3(min(ex[t]['r300'] for t in ex)) + " and " + f3(max(ex[t]['r300'] for t in ex)) + ", against " + f"{ccv_pd:.3f}" + " for FedAvg with CCVR at round 100. The per-day failure is therefore not a matter of training length; server-side calibration (CCVR) is what recovers it.")
print("ext300 complete:", ext_ok, {t: ex[t]["n"] for t in ex}, "ccvr per-day fedavg", round(ccv_pd, 3))

# ---------- response letter ----------
r = open("/home/claude/jnsm6/response_template.md").read()
NEW = ["cic/results/central_tuned", "cic/results/t_K20", "cic/results/t_K50", "cic/results/t_chrono", "cic/results/t_chrono_noport", "cic/results/t_ext300", "cic/results/t_gn",
       "cic/results/t_las", "cic/results/t_main", "cic/results/t_mlp", "cic/results/t_mlp_tune", "cic/results/t_part0.5", "cic/results/t_poison1", "cic/results/t_poison3", "cic/results/tune",
       "g5/results/5gadc_t", "g5/results/5gadc_tune", "nidd/results/main", "nidd/results/tune"]
nrun = sum(len(pd.read_csv(f"/home/claude/{d}/final.csv")) for d in NEW)
r = r.replace("1,016 new training runs", f"{nrun:,} new training runs")
for k, v in rep.items(): r = r.replace(k, v)
r = r.replace("{abs_words}", str(len(re.search(r"\*\*Abstract\.\*\* (.*?)\n", s).group(1).split())))
ch_ = F["t_chrono"]; cv_ = F["t_chrono_vsfedavg"]
F["chrono response"] = ("We added a per-class chronological split (earliest 70% of each class's flows for training, next 15% for validation, latest 15% for testing) with all conflicting-label records removed, and repeated it without the destination port (Sections 4.1 and 5.5). "
    f"FedAvgM's lead over FedAvg held: under IID data FedAvgM led FedAvg by {cv_['iid|fedavgm'].split(' [')[0]} ({ch_['iid|fedavgm'].split(' ±')[0]} against {ch_['iid|fedavg'].split(' ±')[0]}), "
    f"and under the per-day partition by {cv_['temporal|fedavgm'].split(' [')[0]} ({ch_['temporal|fedavgm'].split(' ±')[0]} against {ch_['temporal|fedavg'].split(' ±')[0]}). Without the port the lead was {F['t_chrono_noport_vsfedavg']['iid|fedavgm'].split(' [')[0]} and {F['t_chrono_noport_vsfedavg']['temporal|fedavgm'].split(' [')[0]}. "
    f"Logit adjustment and RAA-FDL stayed within noise of FedAvg ({cv_['iid|fedavg_la'].split(' [')[0]} and {cv_['iid|raa'].split(' [')[0]} under IID data), as in the main runs, although RAA-FDL now lay slightly above FedAvg rather than below it.")
for k in ["ext300 response", "chrono response"]:
    if k in F: r = r.replace("{R:%s}" % k, F[k])
r = r.replace("—", ", ").replace("–", "-")
open("/home/claude/jnsm6/response.md", "w").write(r)
print("runs:", nrun, "response left:", re.findall(r"\{(?:R:|T:|prep|conf|shift|nidd|central|tune|abs)[^}]*\}", r))
open("/home/claude/jnsm6/cover.md", "w").write(open("/home/claude/jnsm6/cover_rev.md").read().replace("{nrun}", f"{nrun:,}"))
