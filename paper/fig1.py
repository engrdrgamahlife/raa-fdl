import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
plt.rcParams.update({"font.family":"serif","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
  "axes.edgecolor":"#555","axes.labelcolor":"#222","xtick.color":"#333","ytick.color":"#333"})
S=["FedAvg","FedProx","SCAFFOLD","RAA-FDL"]
C=["#2a78d6","#eb6834","#1baf7a","#eda100"]; H=["","//","..","xx"]
def bars(ax,cats,vals,errs=None):
    n=len(S); w=0.8/n; x=np.arange(len(cats))
    for k in range(n):
        ax.bar(x+(k-(n-1)/2)*w, vals[k], w-0.02, yerr=None if errs is None else errs[k],
               color=C[k], hatch=H[k], edgecolor="white", linewidth=0.6, label=S[k],
               error_kw=dict(ecolor="#222",capsize=2,lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.yaxis.grid(True,color="#ddd",lw=0.5); ax.set_axisbelow(True)
# Fig1 pipeline
f,ax=plt.subplots(figsize=(4.9,5.2)); ax.set_xlim(0,10); ax.set_ylim(0,11); ax.axis("off")
def box(x,y,w,h,t,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.15",fc=fc,ec="#444",lw=0.8))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=7.5)
def arr(x1,y1,x2,y2): ax.annotate("",(x2,y2),(x1,y1),arrowprops=dict(arrowstyle="-|>",color="#444",lw=0.8))
box(0.3,9.8,4.4,0.9,"CSE-CIC-IDS2018 (15 classes)\n5GAD-c (2) · 5G-NIDD (9)","#dbe8f8")
box(5.3,9.8,4.4,0.9,"70/15/15 split per seed;\nhyperparameters tuned on validation","#dbe8f8")
box(1.5,8.2,7,0.9,"Clients: IID / Dirichlet α / per-day (K = 10),\ninterfaces (K = 4), base stations (K = 2)","#e3f1e0")
arr(2.5,9.8,3.5,9.1); arr(7.5,9.8,6.5,9.1)
xs=[0.3,2.75,5.2,7.65]
for i,x in enumerate(xs):
    box(x,6.0,2.05,1.4,["Client 1","Client 2","  ...  ","Client K"][i]+"\nlocal SGD;\noptional local\ncorrection","#fdf0c8")
    arr(5,8.2,x+1.02,7.35)
    arr(x+1.02,6.0,5,5.35)
box(0.5,3.9,9.0,1.45,"Server, round t: client weights\nsize, uniform or cosine trust\n(FedAvg, SCAFFOLD, SCAFFOLD-N, FLTrust),\nreliability scores (RAA-FDL), label discrepancy (FedDisco)\nor a server optimiser (FedAvgM, FedAdam)","#f8dfe0")
arr(5,3.9,5,3.55)
box(1.3,2.6,7.4,0.95,"Aggregation of models or updates (Table 1)\nafter round 100: optional classifier calibration (CCVR)","#ebe3f5")
arr(5,2.6,5,2.05)
box(1.3,1.1,7.4,0.95,"Global model θ(t+1): validation macro-F1 every 5 rounds;\ntest macro-F1 logged every 5 rounds;\nreported results: model after the last round","#eeeeee")
ax.annotate("",(9.75,6.9),(8.7,1.55),arrowprops=dict(arrowstyle="-|>",color="#777",lw=0.8,connectionstyle="angle,angleA=0,angleB=90,rad=5"))
ax.text(9.85,4.2,"next round",rotation=90,fontsize=7,color="#555",va="center")
f.tight_layout(); f.savefig("/home/claude/jnsm6/fig1_pipeline.png",dpi=300)
