import os
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from raafdl.stats import paired, holm
f=pd.read_csv("results/5gad/final.csv")
print("runs:",len(f)); print(f.groupby("strategy").seed.count().to_dict())
agg=f.groupby("strategy").agg(mf1=("test_macro_f1","mean"),mf1_sd=("test_macro_f1","std"),acc=("test_accuracy","mean"),acc_sd=("test_accuracy","std"),
    rec=("attack_recall","mean"),rec_sd=("attack_recall","std"),prec=("attack_precision","mean"),n=("seed","count"))
print(agg.round(3).to_string())
piv=f.pivot_table(index="seed",columns="strategy",values="test_macro_f1")
def table(prop,bases):
    rows=[]
    for b in bases:
        if b not in piv or prop not in piv: continue
        ok=piv[[prop,b]].dropna(); r=paired(ok[prop],ok[b]); r["comparison"]=f"{prop} vs {b}"; rows.append(r)
    t=pd.DataFrame(rows); t["holm_p"]=holm(t.t_p.values); return t
cols=["comparison","n","mean_delta","ci_low","ci_high","t_p","wilcoxon_p","holm_p","d_z","wins"]
print(table("raa",["fedavg","fedprox","scaffold","fltrust"])[cols].round(4).to_string())
print(table("raa",["raa_noD","raa_gamma1"])[cols].round(4).to_string())
# weight given to lo client by raa variants, final round, and fedavg share
w={}
for l in open("results/5gad/rounds.jsonl"):
    r=json.loads(l)
    if r["round"]==50 and "raa" in r: w.setdefault(r.get("tag",r["strategy"]),[]).append(r["raa"]["2"]["w"])
print({k:(round(np.mean(v),3),round(np.std(v,ddof=1),3) if len(v)>1 else None) for k,v in w.items()})
