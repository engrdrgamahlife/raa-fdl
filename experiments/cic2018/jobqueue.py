"""Resumable 2-worker queue: a job is done when its (scenario, tag, seed) row exists in its out/final.csv."""
import os, shlex, subprocess, pandas as pd
from concurrent.futures import ThreadPoolExecutor
def key(j):
    a = shlex.split(j); g = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    return g("--out"), g("--scenario"), g("--tag", g("--strategy")), int(g("--seed"))
def done(j):
    if j.startswith("CENTRAL"):
        f="results/central/final.csv"
        return os.path.exists(f) and (pd.read_csv(f).query("model=='central_cnn'").seed == int(j.split()[1])).any()
    out, sc, tag, seed = key(j); f = f"{out}/final.csv"
    if not os.path.exists(f): return False
    d = pd.read_csv(f); return ((d.scenario == sc) & (d.strategy == tag) & (d.seed == seed)).any()
def run(j):
    if j.startswith("CENTRAL"):
        seed = j.split()[1]
        if not (os.path.exists("results/central/final.csv") and (pd.read_csv("results/central/final.csv").query("model=='central_cnn'").seed == int(seed)).any()):
            subprocess.run(f"python3 central_cic.py {seed} > logs/central_{seed}.log 2>&1", shell=True)
        return
    out, sc, tag, seed = key(j); os.makedirs("logs", exist_ok=True)
    subprocess.run(f"python3 runcic.py {j} > logs/{os.path.basename(out)}_{sc}_{tag}_{seed}.log 2>&1", shell=True)
jobs = [j.strip() for j in open("jobs.txt") if j.strip() and not done(j.strip())]
print(len(jobs), "pending", flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(run, jobs))
