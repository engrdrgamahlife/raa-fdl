"""One CSE-CIC-IDS2018 federated run (strategy x scenario x seed), resumable from checkpoints."""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from raafdl import data as D
from raafdl.fl import FLConfig, run_federated
from raafdl.models import build_model
from raafdl.metrics import majority_baseline

ap = argparse.ArgumentParser()
ap.add_argument("--strategy", required=True)
ap.add_argument("--scenario", required=True, help="iid | dir0.1 | dir0.5 | dir1.0 | temporal")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--rounds", type=int, default=100)
ap.add_argument("--beta", default="0.4,0.2,0.2,0.2")
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--anchor", default="full", help="full | n<size> | norare")
ap.add_argument("--unreliable", type=int, default=0)
ap.add_argument("--drop", type=float, default=0.5)
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="results/main")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tag = a.tag or a.strategy
K = 10
X, y, day = np.load("X.npy"), np.load("y.npy"), np.load("day.npy")
info = json.load(open("prep_info.json"))
labels = list(range(len(info["label_names"])))
ds = D.Dataset(X, y, day, info["label_names"])
sp = D.split_and_scale(ds, a.seed)
ytr = sp.train[1]
if a.scenario == "iid":
    parts = D.partition(ytr, sp.g_train, "iid", K, a.seed)
elif a.scenario.startswith("dir"):
    parts = D.partition(ytr, sp.g_train, "dirichlet", K, a.seed, float(a.scenario[3:]))
else:
    parts = D.partition(ytr, sp.g_train, "group", len(np.unique(day)), a.seed)
parts = [p for p in parts if len(p) > 0]
K = len(parts)
clients = [(sp.train[0][p], ytr[p]) for p in parts]

# anchor set: always from the validation split
Xv, yv = sp.val
rng = np.random.default_rng(a.seed + 17)
if a.anchor == "full":
    anchor = (Xv, yv)
elif a.anchor[0] == "n" and a.anchor[1:].isdigit():
    i = rng.choice(len(yv), min(int(a.anchor[1:]), len(yv)), replace=False); anchor = (Xv[i], yv[i])
elif a.anchor == "norare":
    cnt = np.bincount(yv, minlength=len(labels)); present = [c for c in np.argsort(cnt) if cnt[c] > 0]
    rare = present[: len(present) // 2]; i = np.where(~np.isin(yv, rare))[0]; anchor = (Xv[i], yv[i])

drop = [0.0] * K
if a.unreliable:
    for i in np.random.default_rng(12345).choice(K, a.unreliable, replace=False):
        drop[i] = a.drop

diag_path = f"{a.out}/diag_{a.scenario}_{a.seed}.json"
if not os.path.exists(diag_path):
    json.dump({"class_counts": D.class_counts(ytr, parts, len(labels)).tolist(),
               "jsd": D.jsd_report(ytr, parts, len(labels)),
               "js_distance_mean_pairwise": float(np.mean([
                   __import__("scipy").spatial.distance.jensenshannon(
                       (np.bincount(ytr[p], minlength=len(labels)) + 1e-12) / (len(p) + 1e-9),
                       (np.bincount(ytr[q], minlength=len(labels)) + 1e-12) / (len(q) + 1e-9), base=2)
                   for i, p in enumerate(parts) for q in parts[i + 1:]])),
               "feature_shift": D.feature_shift_auc(sp.train[0], parts, a.seed),
               "test_classes_missing_from_all_clients": D.test_classes_missing_from_training(sp.test[1], [ytr[p] for p in parts], len(labels)),
               "majority_test": majority_baseline(ytr, sp.test[1], labels),
               "anchor_size": int(len(anchor[1])), "n_train": int(len(ytr))}, open(diag_path, "w"), indent=1)

cfg = FLConfig(rounds=a.rounds, local_epochs=3, batch=256, lr=0.01, eval_every=5, threads=1,
               beta=tuple(float(v) for v in a.beta.split(",")), gamma=a.gamma, drop_probs=drop)
nf = X.shape[1]
recs = run_federated(a.strategy, lambda: build_model("cnn", nf, len(labels)), clients, sp.val, sp.test, labels,
                     cfg, a.seed, anchor=anchor, ckpt_path=f"{a.out}/ckpt_{a.scenario}_{tag}_{a.seed}.pt")
with open(f"{a.out}/rounds.jsonl", "a") as fr:
    for r in recs:
        fr.write(json.dumps({"scenario": a.scenario, "tag": tag, **r}) + "\n")
L = recs[-1]
row = dict(scenario=a.scenario, strategy=tag, seed=a.seed, rounds=a.rounds, test_macro_f1=L["test_macro_f1"],
           test_accuracy=L["test_accuracy"], best_val_macro_f1=max(r.get("val_macro_f1", 0) for r in recs),
           per_class_f1=json.dumps({info["label_names"][int(k)]: round(v["f1"], 4) for k, v in L["test_per_class"].items()}),
           elapsed_s=L["elapsed_s"])
fp = f"{a.out}/final.csv"
pd.DataFrame([row]).to_csv(fp, mode="a", header=not os.path.exists(fp), index=False)
print(row)
