"""Post-hoc CCVR + normal-dominated evaluation for 5GAD checkpoints. Usage: python posthoc_g5.py <out> <tag> [--exclude FakeAMFDelete] [--model cnn]"""
import sys, os, json, argparse, numpy as np, pandas as pd, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")); sys.path.insert(0, "/home/claude/rev")
from raafdl import data as D
from raafdl.models import build_model, split_head
from raafdl.metrics import evaluate
from ccvrlib import ccvr, predict, prior_metrics
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("tag"); ap.add_argument("--exclude", default="FakeAMFDelete")
ap.add_argument("--model", default="cnn"); ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
a = ap.parse_args(); os.chdir("/home/claude/g5")
X = np.load("work/X.npy"); meta = pd.read_csv("work/meta.csv"); rng = np.random.default_rng(0)
norm = np.where(meta.label == 0)[0]; att = np.where((meta.label == 1) & ~meta.source.isin([e for e in a.exclude.split(",") if e]))[0]
keep = np.sort(np.concatenate([att, rng.choice(norm, min(20000, len(norm)), replace=False)]))
ifaces = sorted(meta.interface.unique()); g = meta.interface.map({k: i for i, k in enumerate(ifaces)}).to_numpy()
ds = D.Dataset(X[keep], meta.label.to_numpy()[keep].astype(np.int64), g[keep], ["normal", "attack"]); rows = []
for seed in [int(s) for s in a.seeds.split(",")]:
    ck = f"{a.out}/ckpt_{a.tag}_{seed}.pt"
    if not os.path.exists(ck): continue
    sp = D.split_and_scale(ds, seed); parts = D.partition(sp.train[1], sp.g_train, "group", 4, seed)
    clients = [(sp.train[0][p], sp.train[1][p]) for p in parts if len(p) > 0]
    m = build_model(a.model, X.shape[1], 2); m.load_state_dict(torch.load(ck, weights_only=False)["model"]); m.eval()
    fe, head = split_head(m); yp = predict(fe, head, sp.test[0]); fe2, h2 = ccvr(m, clients, seed); yp2 = predict(fe2, h2, sp.test[0])
    r = dict(scenario="5gad", tag=a.tag, seed=seed, test_macro_f1=evaluate(sp.test[1], yp, [0, 1])["macro_f1"],
             ccvr_val_macro_f1=evaluate(sp.val[1], predict(fe2, h2, sp.val[0]), [0, 1])["macro_f1"],
             ccvr_test_macro_f1=evaluate(sp.test[1], yp2, [0, 1])["macro_f1"],
             **prior_metrics(sp.test[1], yp, [0, 1], 0, 0.99), **{"ccvr_" + k: v for k, v in prior_metrics(sp.test[1], yp2, [0, 1], 0, 0.99).items()})
    rows.append(r); print(seed, round(r["test_macro_f1"], 4), round(r["ccvr_test_macro_f1"], 4), flush=True)
fp = f"{a.out}/posthoc.csv"
if rows: pd.DataFrame(rows).to_csv(fp, mode="a", header=not os.path.exists(fp), index=False)
