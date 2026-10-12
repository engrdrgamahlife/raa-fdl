"""Post-hoc CCVR + benign-dominated evaluation for CSE-CIC-IDS2018 checkpoints of any run family.
Usage: python posthoc_cic.py <results_dir relative to /home/claude/cic> <tag> [--model cnn] [--data full] [--split random]
       [--drop-feat ''] [--K 10] [--scenarios iid,...] [--seeds 42,...]"""
import sys, os, json, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")); sys.path.insert(0, "/home/claude/rev")
from raafdl import data as D
from raafdl.models import build_model, split_head
from raafdl.metrics import evaluate
from ccvrlib import ccvr, predict, prior_metrics
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("tag")
ap.add_argument("--model", default="cnn"); ap.add_argument("--data", default="full"); ap.add_argument("--split", default="random")
ap.add_argument("--drop-feat", default=""); ap.add_argument("--K", type=int, default=10)
ap.add_argument("--scenarios", default="iid,dir1.0,dir0.5,dir0.1,temporal"); ap.add_argument("--seeds", default="42,43,44,45,46")
a = ap.parse_args(); os.chdir("/home/claude/cic")
X, y, day = np.load("X.npy"), np.load("y.npy"), np.load("day.npy"); info = json.load(open("prep_info.json")); t = np.load("t.npy")
if a.data == "clean":
    k = ~np.load("conflict_mask.npy"); X, y, day, t = X[k], y[k], day[k], t[k]
if a.drop_feat:
    X = X[:, [i for i, f in enumerate(info["features"]) if f not in a.drop_feat.split(",")]]
names = info["label_names"]; C = len(names); labels = list(range(C)); B = names.index("Benign")
ds = D.Dataset(X, y, day, names); rows = []
for sc in a.scenarios.split(","):
    for seed in [int(s) for s in a.seeds.split(",")]:
        ck = f"{a.out}/ckpt_{sc}_{a.tag}_{seed}.pt"
        if not os.path.exists(ck): continue
        sp = D.split_and_scale(ds, seed) if a.split == "random" else D.split_chrono(ds, t, seed); ytr = sp.train[1]
        if sc == "iid": parts = D.partition(ytr, sp.g_train, "iid", a.K, seed)
        elif sc.startswith("dir"): parts = D.partition(ytr, sp.g_train, "dirichlet", a.K, seed, float(sc[3:]))
        else: parts = D.partition(ytr, sp.g_train, "group", len(np.unique(day)), seed)
        clients = [(sp.train[0][p], ytr[p]) for p in parts if len(p) > 0]
        ckd = torch.load(ck, weights_only=False); m = build_model(a.model, X.shape[1], C); m.load_state_dict(ckd["model"]); m.eval()
        fe, head = split_head(m); yp = predict(fe, head, sp.test[0])
        r = dict(scenario=sc, tag=a.tag, seed=seed, round=ckd["round"], test_macro_f1=evaluate(sp.test[1], yp, labels)["macro_f1"],
                 **prior_metrics(sp.test[1], yp, labels, B, 0.83))
        fe2, h2 = ccvr(m, clients, seed); yp2 = predict(fe2, h2, sp.test[0])
        r.update(ccvr_val_macro_f1=evaluate(sp.val[1], predict(fe2, h2, sp.val[0]), labels)["macro_f1"],
                 ccvr_test_macro_f1=evaluate(sp.test[1], yp2, labels)["macro_f1"],
                 ccvr_per_class=json.dumps({names[k]: round(v["f1"], 4) for k, v in evaluate(sp.test[1], yp2, labels)["per_class"].items()}),
                 **{"ccvr_" + k: v for k, v in prior_metrics(sp.test[1], yp2, labels, B, 0.83).items()})
        rows.append(r); print(sc, seed, round(r["test_macro_f1"], 4), round(r["ccvr_test_macro_f1"], 4), flush=True)
fp = f"{a.out}/posthoc.csv"
if rows: pd.DataFrame(rows).to_csv(fp, mode="a", header=not os.path.exists(fp), index=False)
