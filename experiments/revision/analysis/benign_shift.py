"""Feature-shift AUC computed on benign records only (class-conditional), per partition and seed."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..")); sys.path.insert(0, "/home/claude/cic")
from raafdl import data as D
X, y, day = np.load("X.npy"), np.load("y.npy"), np.load("day.npy"); info = json.load(open("prep_info.json"))
B = info["label_names"].index("Benign"); ds = D.Dataset(X, y, day, info["label_names"]); rows = []
for seed in [42, 43, 44, 45, 46]:
    sp = D.split_and_scale(ds, seed); ytr = sp.train[1]
    for sc in ["iid", "dir1.0", "dir0.5", "dir0.1", "temporal"]:
        if sc == "iid": parts = D.partition(ytr, sp.g_train, "iid", 10, seed)
        elif sc.startswith("dir"): parts = D.partition(ytr, sp.g_train, "dirichlet", 10, seed, float(sc[3:]))
        else: parts = D.partition(ytr, sp.g_train, "group", len(np.unique(day)), seed)
        parts = [p for p in parts if len(p) > 0]
        bp = [p[ytr[p] == B] for p in parts]
        r = D.feature_shift_auc(sp.train[0], bp, seed)
        rows.append(dict(scenario=sc, seed=seed, benign_auc=r["mean_auc"], n_clients_with_benign=int(sum(len(b) >= 20 for b in bp)),
                         K=len(parts)))
d = pd.DataFrame(rows); d.to_csv("/home/claude/rev/ana/benign_shift.csv", index=False)
print(d.groupby("scenario")[["benign_auc", "n_clients_with_benign", "K"]].mean().round(3))
