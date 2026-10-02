"""Core grid agreed with the author (one line per run: args for runcic.py)."""
S5 = [42, 43, 44, 45, 46]; S3 = [42, 43, 44]
jobs = []
# main: priority order so the key comparisons finish first
for sc in ["dir0.5", "temporal", "iid", "dir0.1", "dir1.0"]:
    for seed in S5:
        for s in ["fedavg", "fedprox", "scaffold", "fltrust", "raa"]:
            jobs.append(f"--strategy {s} --scenario {sc} --seed {seed} --rounds 100 --out results/main")
for seed in S3:  # server-side data fairness (Table 9)
    jobs.append(f"--strategy fedavg_ft --scenario dir0.5 --seed {seed} --rounds 100 --out results/anchor")
    for an in ["n1000", "n200", "norare"]:
        jobs.append(f"--strategy raa --scenario dir0.5 --seed {seed} --rounds 100 --anchor {an} --tag raa_{an} --out results/anchor")
for seed in S3:  # leave-one-out of each reliability term (Table 18)
    for t, b in [("noA", "0,0.3333,0.3333,0.3334"), ("noS", "0.5,0,0.25,0.25"), ("noC", "0.5,0.25,0,0.25"), ("noD", "0.5,0.25,0.25,0")]:
        jobs.append(f"--strategy raa --scenario dir0.5 --seed {seed} --rounds 100 --beta {b} --tag raa_{t} --out results/loo")
for seed in S3:  # 5GAD mechanism check on the temporal partition
    jobs.append(f"--strategy raa --scenario temporal --seed {seed} --rounds 100 --beta 0.5,0.25,0.25,0 --tag raa_noD --out results/temporal_abl")
    jobs.append(f"--strategy raa --scenario temporal --seed {seed} --rounds 100 --gamma 1.0 --tag raa_gamma1 --out results/temporal_abl")
for m in [3, 5, 7]:  # communication reliability (Table 17), 50 rounds
    for seed in S3:
        for s, extra in [("fedavg", ""), ("fedprox", ""), ("scaffold", ""), ("raa", ""), ("raa", "--beta 0.5,0.25,0,0.25 --tag raa_noC")]:
            tag = "raa_noC" if "noC" in extra else s
            jobs.append(f"--strategy {s} --scenario dir0.5 --seed {seed} --rounds 50 --unreliable {m} --drop 0.5 {extra} --out results/unrel{m}".replace(" --out", f"{'' if extra else ''} --out"))
open("jobs.txt", "w").write("\n".join(jobs) + "\n")
print(len(jobs), "jobs")
