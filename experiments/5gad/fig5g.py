import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8.5,"axes.spines.top":False,"axes.spines.right":False})
recs={}
for l in open("results/5gad100/rounds.jsonl"):
    r=json.loads(l)
    if "val_macro_f1" in r: recs[(r.get("tag",r["strategy"]),r["seed"],r["round"])]=r["val_macro_f1"]
c=pd.Series(recs).rename_axis(["tag","seed","round"]).reset_index(name="v")
C={"fedavg":"#2a78d6","fedprox":"#eb6834","scaffold":"#1baf7a","fltrust":"#4a3aa7","raa":"#e34948","raa_noD":"#eda100","raa_gamma1":"#e87ba4"}
L={"fedavg":"FedAvg","fedprox":"FedProx","scaffold":"SCAFFOLD","fltrust":"FLTrust","raa":"RAA-FDL","raa_noD":"RAA-FDL without D","raa_gamma1":"RAA-FDL, γ = 1"}
S={"fedprox":(0,(4,2)),"scaffold":(0,(1,1.5)),"fltrust":(0,(6,2,1,2)),"raa_noD":(0,(4,2)),"raa_gamma1":(0,(1,1.5))}
fig,axs=plt.subplots(1,2,figsize=(4.9,2.3),sharey=True)
for ax,tags,title in [(axs[0],["fedavg","fedprox","scaffold","fltrust","raa"],"(a) Aggregation strategies"),(axs[1],["fedavg","raa","raa_noD","raa_gamma1"],"(b) RAA-FDL ablations")]:
    for t in tags:
        g=c[c.tag==t].groupby("round").v; m=g.mean(); s=g.std(ddof=1)
        ax.plot(m.index,m.values,color=C[t],lw=1.6,ls=S.get(t,"-"),label=L[t])
        ax.fill_between(m.index,m-s,m+s,color=C[t],alpha=0.12,lw=0)
    ax.set_title(title,fontsize=8.5); ax.set_xlabel("Communication round"); ax.set_ylim(0.25,1.0); ax.set_xlim(0,100)
    ax.yaxis.grid(True,color="#ddd",lw=0.5); ax.legend(frameon=False,fontsize=6.5,loc="lower right")
axs[0].set_ylabel("Validation macro-F1")
fig.tight_layout(); fig.savefig("fig6_5gad.png",dpi=300)
print(c.groupby("tag").seed.nunique().to_dict())
