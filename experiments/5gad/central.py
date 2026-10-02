import numpy as np, pandas as pd, sys, json, os
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from raafdl import data as D
from raafdl.metrics import evaluate, majority_baseline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
X=np.load("work/X.npy"); m=pd.read_csv("work/meta.csv")
rng=np.random.default_rng(0); norm=np.where(m.label==0)[0]; att=np.where(m.label==1)[0]
keep=np.sort(np.concatenate([att,rng.choice(norm,20000,replace=False)]))
ifaces=sorted(m.interface.unique()); g=m.interface.map({k:i for i,k in enumerate(ifaces)}).to_numpy()
ds=D.Dataset(X[keep],m.label.to_numpy()[keep].astype(np.int64),g[keep],["normal","attack"])
rows=[]
for seed in [42,43,44,45,46]:
    sp=D.split_and_scale(ds,seed); Xtr,ytr=sp.train; Xte,yte=sp.test
    mb=majority_baseline(ytr,yte,[0,1]); rows.append(dict(model="majority",seed=seed,macro_f1=mb["macro_f1"],acc=mb["accuracy"],recall=mb["per_class"][1]["recall"]))
    for name,cols,mdl in [("ports_protocol_LR",list(range(128,135)),LogisticRegression(max_iter=3000)),
                          ("payload_LR",list(range(128)),LogisticRegression(max_iter=3000)),
                          ("all_HGB",list(range(135)),HistGradientBoostingClassifier(random_state=seed))]:
        e=evaluate(yte,mdl.fit(Xtr[:,cols],ytr).predict(Xte[:,cols]),[0,1])
        rows.append(dict(model=name,seed=seed,macro_f1=e["macro_f1"],acc=e["accuracy"],recall=e["per_class"][1]["recall"]))
df=pd.DataFrame(rows); df.to_csv("results/5gad_central.csv",index=False)
print(df.groupby("model")[["macro_f1","acc","recall"]].agg(["mean","std"]).round(4))
