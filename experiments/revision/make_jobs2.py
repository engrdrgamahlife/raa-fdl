"""Phase 2: main comparison and robustness at the tuned settings (selected on the validation split).
CSE-CIC-IDS2018: common local lr 0.3; FedAvgM server lr 1.0 (momentum 0.9); FedAdam server lr 0.1; FedProx mu 0.001.
5GAD: local lr LR5 (selected on the 5GAD validation split), server settings as for CSE-CIC-IDS2018."""
import sys
CIC, G5 = "/home/claude/cic", "/home/claude/g5"
LR, LR5 = "0.3", (sys.argv[1] if len(sys.argv) > 1 else None)
LRN = sys.argv[2] if len(sys.argv) > 2 else None
NIDD = "/home/claude/nidd"
S5 = [42, 43, 44, 45, 46]; SX = [47, 48, 49, 50, 51]
SC = ["iid", "dir1.0", "dir0.5", "dir0.1", "temporal"]
FULL = "--la --beta 0.5,0.25,0.25,0 --gamma 1.0"
ST = {"fedavg": ("fedavg", ""), "fedprox": ("fedprox", "--mu 0.001"), "scaffold": ("scaffold", ""),
      "scaffold_n": ("scaffold_n", ""), "fltrust": ("fltrust", ""), "fltrust_noscale": ("fltrust", "--fltrust-noscale"),
      "fltrust_root200": ("fltrust", "--anchor n200"), "raa": ("raa", ""), "feddisco": ("feddisco", ""),
      "fedavgm": ("fedavgm", "--server-lr 1.0"), "fedadam": ("fedadam", "--server-lr 0.1"),
      "fedavg_la": ("fedavg", "--la"), "fedrs": ("fedavg", "--fedrs 0.5"), "scaffold_n_la": ("scaffold_n", "--la"), "full": ("raa_scaffold", FULL)}
J = []
def cic(out, sc, tag, seed, st, ex, rounds=100, lr=LR):
    J.append((CIC, out, sc, tag, seed, f"python3 runcic.py --strategy {st} --scenario {sc} --seed {seed} --rounds {rounds} --lr {lr} {ex} --tag {tag} --out {out}"))
def g5(out, tag, seed, st, ex):
    J.append((G5, out, "5gad", tag, seed, f"python3 run5gad.py --strategies {st} --seeds {seed} --lr {LR5} {ex} --tag {tag} --rounds 100 --normal-cap 20000 --exclude FakeAMFDelete --out {out}"))
for l in open("jobs_phase1.txt"):                            # unfinished 5GAD tuning first
    if "5gadc_tune" in l: J.append(tuple(l.rstrip("\n").split("\t")))
for seed in [42, 43, 44]:
    for t, (st, ex) in {"fedavg": ("fedavg", ""), "fedavg_la": ("fedavg", "--la"), "scaffold": ("scaffold", "")}.items():
        J.append((G5, "results/5gadc_tune", "5gad", f"{t}_lr0.003", seed, f"python3 run5gad.py --strategies {st} --seeds {seed} {ex} --lr 0.003 --tag {t}_lr0.003 --rounds 100 --normal-cap 20000 --exclude FakeAMFDelete --out results/5gadc_tune"))
for seed in [42, 43, 44]:                                    # 5G-NIDD: local learning rate on validation
    for lr in ["0.01", "0.03", "0.1", "0.3"]:
        for t in ["fedavg", "fedavg_la", "scaffold"]:
            st, ex = ST[t]
            J.append((NIDD, "results/tune", "site", f"{t}_lr{lr}", seed, f"python3 runcic.py --strategy {st} --scenario site --seed {seed} --rounds 100 --lr {lr} {ex} --tag {t}_lr{lr} --out results/tune"))
for seed in S5:                                              # main comparison, CSE-CIC-IDS2018
    for sc in SC:
        for t, (st, ex) in ST.items(): cic("results/t_main", sc, t, seed, st, ex)
if LRN:                                                      # main comparison, 5G-NIDD (K = 2 base stations)
    for seed in S5:
        for t, (st, ex) in ST.items():
            if t in ("fltrust_root200", "fltrust_noscale", "scaffold_n"): continue
            J.append((NIDD, "results/main", "site", t, seed, f"python3 runcic.py --strategy {st} --scenario site --seed {seed} --rounds 100 --lr {LRN} {ex} --tag {t} --out results/main"))
for seed in S5:                                              # tuned centralised references
    for model in ["cnn", "mlp"]:
        for lr in ["0.01", "0.03", "0.1", "0.3", "1.0"]:
            J.append((CIC, "results/central_tuned", "central", f"central_{model}_lr{float(lr)}", seed, f"python3 central_tuned.py {seed} {lr} {model}"))
if LR5:
    for seed in S5:
        for t, (st, ex) in ST.items(): g5("results/5gadc_t", t, seed, st, ex)
        g5("results/5gadc_t", "la_multi", seed, "fedavg", "--la --la-multi")
# robustness (tuned): normalisation and model
for sc in ["temporal", "dir0.1"]:
    for seed in S5:
        for t in ["fedavg", "fedavg_la", "scaffold"]: st, ex = ST[t]; cic("results/t_gn", sc, t, seed, st, f"{ex} --model cnn_gn")
for sc in ["iid", "dir0.1", "temporal"]:
    for seed in S5:
        for t in ["fedavg", "fedavg_la", "scaffold"]: st, ex = ST[t]; cic("results/t_mlp", sc, t, seed, st, f"{ex} --model mlp")
MLPS = {"fedavgm": sys.argv[3] if len(sys.argv) > 3 else None, "fedadam": sys.argv[4] if len(sys.argv) > 4 else None}
for sc in ["dir0.5", "temporal"]:                            # MLP: server learning rates re-tuned on validation
    for seed in [42, 43, 44]:
        for slr in ["0.1", "0.3"]: cic("results/t_mlp_tune", sc, f"fedavgm_slr{slr}", seed, "fedavgm", f"--server-lr {slr} --model mlp")
        for slr in ["0.01", "0.03"]: cic("results/t_mlp_tune", sc, f"fedadam_slr{slr}", seed, "fedadam", f"--server-lr {slr} --model mlp")
        for slr in ["1.0"]: cic("results/t_mlp_tune", sc, f"fedavgm_slr{slr}", seed, "fedavgm", f"--server-lr {slr} --model mlp")
        for slr in ["0.1"]: cic("results/t_mlp_tune", sc, f"fedadam_slr{slr}", seed, "fedadam", f"--server-lr {slr} --model mlp")
for t, slr in MLPS.items():
    if slr:
        for sc in ["iid", "dir0.1", "temporal"]:
            for seed in S5: cic("results/t_mlp", sc, f"{t}_tuned", seed, t, f"--server-lr {slr} --model mlp")
for sc in ["iid", "dir0.1", "temporal"]:                     # MLP: server-side strategies too
    for seed in S5:
        for t in ["fedavgm", "fedadam", "raa", "feddisco"]: st, ex = ST[t]; cic("results/t_mlp", sc, t, seed, st, f"{ex} --model mlp")
# LA sensitivity
for sc in ["temporal"]:
    for seed in S5:
        for t, ex in {"la_tau0.5": "--la --la-tau 0.5", "la_tau2": "--la --la-tau 2", "la_eps0.1": "--la --la-eps 0.1", "la_eps10": "--la --la-eps 10"}.items():
            cic("results/t_las", sc, t, seed, "fedavg", ex)
# poisoning, participation, number of clients
for k in [1, 3]:
    for seed in S5:
        for t, (st, ex) in {"fedavg": ST["fedavg"], "fedavgm": ST["fedavgm"], "fedavg_la": ST["fedavg_la"], "fltrust": ST["fltrust"],
                             "raa": ST["raa"]}.items():
            cic(f"results/t_poison{k}", "dir0.5", t, seed, st, f"{ex} --poison {k}")
for p in ["0.5"]:
    for sc in ["dir0.5", "temporal"]:
        for seed in S5:
            for t in ["fedavg", "fedavgm", "fedavg_la", "scaffold"]: st, ex = ST[t]; cic(f"results/t_part{p}", sc, t, seed, st, f"{ex} --participation {p}")
for K in [20, 50]:
    for seed in S5:
        for t in ["fedavg", "fedavgm", "fedavg_la", "scaffold"]: st, ex = ST[t]; cic(f"results/t_K{K}", "dir0.5", t, seed, st, f"{ex} --K {K}")
# data robustness
for out, ex2 in [("results/t_chrono", "--data clean --split chrono"), ("results/t_chrono_noport", "--data clean --split chrono --drop-feat 'Dst Port'")]:
    for sc in ["iid", "temporal"]:
        for seed in S5:
            for t in (["fedavg", "fedavgm", "fedavg_la", "raa"] if out == "results/t_chrono" else ["fedavg", "fedavgm", "fedavg_la"]): st, ex = ST[t]; cic(out, sc, t, seed, st, f"{ex} {ex2}")
# 300 rounds (continued from the round-100 checkpoints of t_main, copied into t_ext300)
for sc in ["temporal"]:
    for seed in S5:
        for t in ["fedavg", "fedavgm", "fedavg_la", "scaffold"]:
            st, ex = ST[t]; cic("results/t_ext300", sc, t, seed, st, ex, rounds=300)
            w, o, s_, tg, sd, cmd = J[-1]
            J[-1] = (w, o, s_, tg, sd, f"mkdir -p results/t_ext300 && (test -f results/t_ext300/ckpt_{sc}_{t}_{seed}.pt || cp results/t_main/ckpt_{sc}_{t}_{seed}.pt results/t_ext300/) && {cmd}")
# five more seeds (only if time allows; last in the queue)
for seed in (SX if "--seeds10" in sys.argv else []):
    for sc in SC:
        for t in ["fedavg", "fedavgm", "raa", "fedavg_la"]: st, ex = ST[t]; cic("results/t_seeds10", sc, t, seed, st, ex)
    if LR5:
        for t in ["fedavg", "fedavgm", "raa", "fedavg_la"]: st, ex = ST[t]; g5("results/5gadc_t", t, seed, st, ex)
open("jobs.txt", "w").write("\n".join("\t".join(map(str, j)) for j in J) + "\n")
import collections; print(len(J), collections.Counter(j[1] for j in J))
