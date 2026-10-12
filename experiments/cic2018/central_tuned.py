"""Centralised reference with a tuned learning rate: same network trained on the pooled training split for 300
epochs (the sample passes of 100 rounds x 3 local epochs), SGD at lr in {0.01, 0.03, 0.1, 0.3, 1.0}; the
learning rate is selected on validation macro-F1. Usage: python central_tuned.py <seed> <lr> <model>"""
import sys, json, os, numpy as np, pandas as pd, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
from raafdl import data as D
from raafdl.models import build_model
from raafdl.fl import FLConfig, local_train, predict
from raafdl.metrics import evaluate
seed, lr, model = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
out = "results/central_tuned"; os.makedirs(out, exist_ok=True); torch.set_num_threads(1)
X, y, day = np.load("X.npy"), np.load("y.npy"), np.load("day.npy"); info = json.load(open("prep_info.json"))
labels = list(range(len(info["label_names"]))); B = info["label_names"].index("Benign")
sp = D.split_and_scale(D.Dataset(X, y, day, info["label_names"]), seed)
torch.manual_seed(seed); m = build_model(model, X.shape[1], len(labels))
local_train(m, *sp.train, FLConfig(local_epochs=300, batch=256, lr=lr), seed)
ev, et = evaluate(sp.val[1], predict(m, sp.val[0]), labels), evaluate(sp.test[1], yp := predict(m, sp.test[0]), labels)
yt = sp.test[1]
r = dict(scenario="central", strategy=f"central_{model}_lr{lr}", seed=seed, val_macro_f1=ev["macro_f1"], test_macro_f1=et["macro_f1"],
         test_accuracy=et["accuracy"], benign_fpr=float(((yp != B) & (yt == B)).sum() / (yt == B).sum()),
         per_class_f1=json.dumps({info["label_names"][k]: round(v["f1"], 4) for k, v in et["per_class"].items()}))
f = f"{out}/final.csv"; pd.DataFrame([r]).to_csv(f, mode="a", header=not os.path.exists(f), index=False); print(r)
