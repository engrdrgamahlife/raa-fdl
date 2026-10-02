"""Resumable job queue: skips runs already in final.csv; in-progress runs resume from checkpoints."""
import pandas as pd, subprocess, os
OUT=os.environ.get('OUT','results/5gad'); ROUNDS=os.environ.get('ROUNDS','50')
from concurrent.futures import ThreadPoolExecutor
def pending():
    done=set()
    if os.path.exists(OUT+"/final.csv"):
        f=pd.read_csv(OUT+"/final.csv"); done={(s,int(d)) for s,d in zip(f.strategy,f.seed)}
    return [l.split() for l in open("jobs_all.txt") if l.strip() and (l.split()[4],int(l.split()[1])) not in done]
def run(j):
    s,seed,beta,gamma,tag=j
    subprocess.run(f"python3 run5gad.py --strategies {s} --seeds {seed} --beta {beta} --gamma {gamma} --tag {tag} --rounds {ROUNDS} --normal-cap 20000 --out {OUT} > logs/{tag}_{seed}_{ROUNDS}.log 2>&1",shell=True)
jobs=pending(); print(len(jobs),"pending",flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(run,jobs))
