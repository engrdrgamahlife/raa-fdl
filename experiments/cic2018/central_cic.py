"""Centralised references on the pooled training split (Sec. 4.10).
CNN: same model/optimiser as the clients, 300 epochs = the sample passes of 100 rounds x 3 local epochs.
HGB: gradient-boosted trees as a strong non-neural reference. Also the majority baseline."""
import sys, json, os, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from raafdl import data as D
from raafdl.models import build_model
from raafdl.fl import FLConfig, local_train, predict
from raafdl.metrics import evaluate, majority_baseline
from sklearn.ensemble import HistGradientBoostingClassifier
seed = int(sys.argv[1]); out = "results/central"; os.makedirs(out, exist_ok=True)
torch.set_num_threads(1)
X, y, day = np.load("X.npy"), np.load("y.npy"), np.load("day.npy"); info = json.load(open("prep_info.json"))
labels = list(range(len(info["label_names"])))
sp = D.split_and_scale(D.Dataset(X, y, day, info["label_names"]), seed)
rows = []
mb = majority_baseline(sp.train[1], sp.test[1], labels); rows.append(("majority", mb))
torch.manual_seed(seed); m = build_model("cnn", X.shape[1], len(labels))
local_train(m, *sp.train, FLConfig(local_epochs=300, batch=256, lr=0.01), seed)
rows.append(("central_cnn", evaluate(sp.test[1], predict(m, sp.test[0]), labels)))
h = HistGradientBoostingClassifier(random_state=seed).fit(*sp.train)
rows.append(("central_hgb", evaluate(sp.test[1], h.predict(sp.test[0]), labels)))
for name, e in rows:
    r = dict(model=name, seed=seed, macro_f1=e["macro_f1"], accuracy=e["accuracy"],
             per_class_f1=json.dumps({info["label_names"][k]: round(v["f1"], 4) for k, v in e["per_class"].items()}))
    f = f"{out}/final.csv"; pd.DataFrame([r]).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
    print(name, round(e["macro_f1"], 4), round(e["accuracy"], 4))
