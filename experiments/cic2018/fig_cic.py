import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.spines.top":False,"axes.spines.right":False})
c=pd.read_csv("results/curves_main.csv")
C={"fedavg":"#2a78d6","fedprox":"#eb6834","scaffold":"#1baf7a","fltrust":"#4a3aa7","raa":"#e34948"}
L={"fedavg":"FedAvg","fedprox":"FedProx","scaffold":"SCAFFOLD","fltrust":"FLTrust","raa":"RAA-FDL"}
S={"fedprox":(0,(4,2)),"scaffold":(0,(1,1.5)),"fltrust":(0,(6,2,1,2))}
T={"dir0.5":"(a) Dirichlet α = 0.5","dir0.1":"(b) Dirichlet α = 0.1","temporal":"(c) Temporal (capture day)"}
fig,axs=plt.subplots(1,3,figsize=(4.9,2.1),sharey=True)
for ax,sc in zip(axs,T):
    for t in C:
        g=c[(c.scenario==sc)&(c.tag==t)].groupby("round").v; mu=g.mean(); sd=g.std(ddof=1)
        ax.plot(mu.index,mu.values,color=C[t],lw=1.3,ls=S.get(t,"-"),label=L[t]); ax.fill_between(mu.index,mu-sd,mu+sd,color=C[t],alpha=0.12,lw=0)
    ax.set_title(T[sc],fontsize=7.5); ax.set_xlabel("Round"); ax.set_xlim(0,100); ax.set_ylim(0,0.6); ax.yaxis.grid(True,color="#ddd",lw=0.5)
axs[0].set_ylabel("Validation macro-F1"); axs[0].legend(frameon=False,fontsize=6,loc="upper left")
fig.tight_layout(); fig.savefig("fig2_cic_curves.png",dpi=300)
