import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
T = pd.read_csv("/home/claude/rev/ana/paired_vs_fedavg.csv")
SC = ["iid", "dir1.0", "dir0.5", "dir0.1", "temporal"]; LAB = {"iid": "IID", "dir1.0": "Dir. α=1.0", "dir0.5": "Dir. α=0.5", "dir0.1": "Dir. α=0.1", "temporal": "Per-day"}
ST = [("fedavgm", "FedAvgM", "S"), ("fedadam", "FedAdam", "S"), ("feddisco", "FedDisco", "R"), ("raa", "RAA-FDL", "R"), ("fltrust", "FLTrust", "R"),
      ("fedprox", "FedProx", "C"), ("scaffold", "SCAFFOLD", "C"), ("fedavg_la", "FedAvg + LA", "C"), ("fedrs", "FedRS", "C")]
COL = {"S": "#2a78d6", "R": "#eb6834", "C": "#1baf7a"}; MK = {"S": "o", "R": "s", "C": "^"}
fig, axs = plt.subplots(1, 5, figsize=(7.4, 3.0), sharey=True)
for ax, sc in zip(axs, SC):
    for k, (s, name, fam) in enumerate(ST):
        r = T[(T.setting == sc) & (T.strategy == s)].iloc[0]
        ax.plot([r.ci_low, r.ci_high], [k, k], color=COL[fam], lw=1.8, solid_capstyle="round")
        ax.plot(r.mean_delta, k, MK[fam], ms=5.5, mfc=COL[fam] if r.holm < 0.05 else "white", mec=COL[fam], mew=1.4)
    ax.axvline(0, color="#444", lw=0.8); ax.axvspan(-0.02, 0.02, color="#999", alpha=0.12, lw=0)
    ax.set_title(LAB[sc], fontsize=8, loc="left"); ax.set_xlim(-0.45, 0.25); ax.set_xticks([-0.4, -0.2, 0, 0.2]); ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(ST))); axs[0].set_yticklabels([n for _, n, _ in ST]); axs[0].invert_yaxis()
fig.text(0.5, 0.075, "Difference in test macro-F1 from FedAvg (mean over five paired seeds, 95% CI)", ha="center", fontsize=7.5)
fig.text(0.01, 0.01, "Filled: Holm p < 0.05. Grey band: ±0.02 equivalence margin. Circles: server optimisers; squares: server reweighting; triangles: client-side corrections.", fontsize=6.5, color="#555")
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig("/home/claude/jnsm6/fig2_forest.png", dpi=300)
# Fig 3: per-day validation curves to 300 rounds + CCVR
R = [json.loads(l) for l in open("/home/claude/cic/results/t_ext300/rounds.jsonl")]
d = pd.DataFrame([r for r in R if "val_macro_f1" in r and r["scenario"] == "temporal"]).drop_duplicates(["tag", "seed", "round"], keep="last")
S3 = [("fedavg", "FedAvg", "#2a78d6", "-"), ("fedavgm", "FedAvgM", "#eb6834", "--"), ("fedavg_la", "FedAvg + LA", "#1baf7a", ":"), ("scaffold", "SCAFFOLD", "#eda100", "-.")]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
for tag, name, c, ls in S3:
    g = d[d.tag == tag].groupby("round").val_macro_f1; m, s = g.mean(), g.std(ddof=1)
    axs[0].plot(m.index, m.values, color=c, ls=ls, lw=1.5, label=name); axs[0].fill_between(m.index, m - s, m + s, color=c, alpha=0.12, lw=0)
axs[0].axvline(100, color="#999", lw=0.8, ls=":"); axs[0].set_xlabel("Round"); axs[0].set_ylabel("Validation macro-F1"); axs[0].set_title("(a) Per-day partition, 300 rounds", fontsize=8, loc="left")
axs[0].legend(frameon=False, fontsize=7, loc="upper left")
P = pd.read_csv("/home/claude/cic/results/t_main/posthoc.csv"); P = P[(P.scenario == "temporal") & (P["round"] == 100)]
tags = [("fedavg", "FedAvg"), ("fedavgm", "FedAvgM"), ("fedadam", "FedAdam"), ("raa", "RAA-FDL"), ("fedavg_la", "FedAvg+LA"), ("scaffold", "SCAFFOLD")]
x = np.arange(len(tags))
b = [P[P.tag == t].test_macro_f1.mean() for t, _ in tags]; a = [P[P.tag == t].ccvr_test_macro_f1.mean() for t, _ in tags]
axs[1].bar(x - 0.19, b, 0.36, color="#9aa7b8", label="trained model"); axs[1].bar(x + 0.19, a, 0.36, color="#2a78d6", label="after CCVR")
axs[1].set_xticks(x); axs[1].set_xticklabels([n for _, n in tags], rotation=40, ha="right", fontsize=7); axs[1].set_ylim(0, 0.9)
axs[1].set_title("(b) Per-day partition: test macro-F1", fontsize=8, loc="left"); axs[1].legend(frameon=False, fontsize=7, loc="upper center", ncol=2); axs[1].grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("/home/claude/jnsm6/fig3_perday.png", dpi=300)
print("ok")
