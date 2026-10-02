"""5GAD federated experiment: K = 4 clients = capture interfaces (E1 of the revision plan)."""
import sys, json, os, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from raafdl import data as D
from raafdl.fl import FLConfig, run_federated
from raafdl.models import build_model
from raafdl.metrics import majority_baseline, evaluate
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--strategies", default="fedavg,fedprox,scaffold,fltrust,raa")
ap.add_argument("--seeds", default="42,43,44,45,46")
ap.add_argument("--rounds", type=int, default=100)
ap.add_argument("--normal-cap", type=int, default=60000)
ap.add_argument("--out", default="results/5gad")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--beta", default="0.4,0.2,0.2,0.2")
ap.add_argument("--gamma", type=float, default=0.5)
ap.add_argument("--tag", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
X = np.load("work/X.npy"); meta = pd.read_csv("work/meta.csv")
rng = np.random.default_rng(0)
norm = np.where(meta.label == 0)[0]; att = np.where(meta.label == 1)[0]
keep = np.sort(np.concatenate([att, rng.choice(norm, min(a.normal_cap, len(norm)), replace=False)]))
ifaces = sorted(meta.interface.unique())
g = meta.interface.map({k: i for i, k in enumerate(ifaces)}).to_numpy()
ds = D.Dataset(X[keep], meta.label.to_numpy()[keep].astype(np.int64), g[keep], ["normal", "attack"])
cfg = FLConfig(rounds=a.rounds, local_epochs=3, batch=256, lr=0.01, eval_every=5, threads=a.threads, beta=tuple(float(x) for x in a.beta.split(",")), gamma=a.gamma)
fr = open(f"{a.out}/rounds.jsonl", "a")
for seed in [int(s) for s in a.seeds.split(",")]:
    sp = D.split_and_scale(ds, seed)
    parts = D.partition(sp.train[1], sp.g_train, "group", 4, seed)
    clients = [(sp.train[0][p], sp.train[1][p]) for p in parts]
    diag = {"interfaces": ifaces, "class_counts": D.class_counts(sp.train[1], parts, 2).tolist(),
            "majority_test": majority_baseline(sp.train[1], sp.test[1], [0, 1])}
    json.dump(diag, open(f"{a.out}/diag_{seed}.json", "w"), indent=1)
    for s in a.strategies.split(","):
        recs = run_federated(s, lambda: build_model("cnn", X.shape[1], 2), clients, sp.val, sp.test, [0, 1],
                             cfg, seed, anchor=sp.val,
                             log=None, ckpt_path=f"{a.out}/ckpt_{a.tag or s}_{seed}.pt")
        for r in recs: fr.write(json.dumps({"scenario": "5gad", "tag": a.tag or s, **r}) + "\n")
        fr.flush()
        L = recs[-1]; pc = L["test_per_class"]
        row = dict(scenario="5gad", strategy=(a.tag or s), seed=seed, test_macro_f1=L["test_macro_f1"],
                   test_accuracy=L["test_accuracy"], attack_recall=pc[1]["recall"], attack_precision=pc[1]["precision"],
                   elapsed_s=L["elapsed_s"])
        fp = f"{a.out}/final.csv"
        pd.DataFrame([row]).to_csv(fp, mode="a", header=not os.path.exists(fp), index=False)
        print(row, flush=True)
