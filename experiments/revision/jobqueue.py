"""Resumable 2-worker queue for v5; a job is done when its row exists in <workdir>/<out>/final.csv."""
import os, subprocess, pandas as pd
from concurrent.futures import ThreadPoolExecutor
def done(wd, out, sc, tag, seed):
    f = f"{wd}/{out}/final.csv"
    if not os.path.exists(f): return False
    d = pd.read_csv(f); return ((d.scenario == sc) & (d.strategy == tag) & (d.seed == int(seed))).any()
def run(j):
    wd, out, sc, tag, seed, cmd = j
    os.makedirs(f"{wd}/logs", exist_ok=True)
    subprocess.run(f"cd {wd} && {cmd} > logs/rev_{os.path.basename(out)}_{sc}_{tag}_{seed}.log 2>&1", shell=True)
jobs = [l.rstrip("\n").split("\t") for l in open("/home/claude/rev/jobs.txt") if l.strip()]
jobs = [j for j in jobs if not done(*j[:5])]
print(len(jobs), "pending", flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(run, jobs))
