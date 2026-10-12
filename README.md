# Heterogeneity remedies for federated intrusion detection (revision)

Code, configurations and per-seed results for the paper *Which Heterogeneity Remedy Helps Federated Intrusion Detection? A Tuned Comparison of Server-Side and Client-Side Methods on Enterprise and 5G Traffic* (A. G. Gbaden, revised version submitted to the Journal of Network and Systems Management, 2026).

Every number in the revised paper is computed from the files in `results/` by the scripts in `experiments/revision/analysis/` and `paper/`. The results of the submitted version are kept unchanged in `results_submitted/`.

## Contents

| Path | What it holds |
| --- | --- |
| `raafdl/` | Federated learning library: 15 strategies (FedAvg, FedAvgM, FedAdam, FedProx, SCAFFOLD, SCAFFOLD-N, FLTrust and variants, FedDisco, RAA-FDL, logit adjustment, FedRS and combinations), partitioning, metrics, paired statistics, models (1-D CNN with BatchNorm or GroupNorm, MLP) |
| `experiments/cic2018/` | CSE-CIC-IDS2018 preprocessing, timestamps and conflicting-label check (`prep_time.py`), run script, tuned centralised baselines |
| `experiments/5gad/` | 5GAD packet feature extraction and run script (FakeAMFDelete excluded with `--exclude FakeAMFDelete`) |
| `experiments/5g-nidd/` | 5G-NIDD preprocessing (two base stations as two sites) and run script |
| `experiments/revision/` | Job list of all revision runs (`jobs.txt`), resumable queue, CCVR post-hoc calibration, analysis scripts and tuning tables |
| `paper/` | Manuscript template and the scripts that fill every table and draw every figure |
| `results/` | `final.csv` (final metrics per run), `rounds.jsonl.gz` (validation and test metrics every five rounds), `posthoc.csv` (operating point and CCVR), partition diagnostics |

## Installation

```
pip install -r requirements.txt
```

Everything runs on CPU. A 100-round CSE-CIC-IDS2018 run takes about 10 to 15 minutes on one core.

## Reproducing

1. Prepare the data as described in the paper (Section 4.1): `experiments/cic2018/prep.py` and `prep_time.py`, `experiments/5gad/extract.py`, `experiments/5g-nidd/prep_nidd.py` (input: `Combined.csv` of 5G-NIDD). Data paths are set at the top of each script.
2. `cd experiments/revision && python make_jobs2.py 0.003 0.3 0.3 0.01` writes `jobs.txt` with the tuned learning rates; `python jobqueue.py` runs it. Runs checkpoint every five rounds and resume after an interruption.
3. `python posthoc_cic.py`, `posthoc_g5.py` and `posthoc_nidd.py` compute operating-point metrics and CCVR.
4. `python analysis/fill_rev.py`, then `python ../../paper/make_ms6.py` and `figs6.py`, produce all tables and figures.

## Main result

Test macro-F1 on CSE-CIC-IDS2018 after 100 rounds, tuned protocol (mean over five seeds; per-day = each client holds whole capture days):

| Strategy | IID | Dirichlet 1.0 | Dirichlet 0.5 | Dirichlet 0.1 | Per-day |
| --- | --- | --- | --- | --- | --- |
| FedAvg | 0.697 | 0.706 | 0.699 | 0.655 | 0.266 |
| FedAvgM | 0.787 | 0.786 | 0.756 | 0.684 | 0.284 |
| FedAdam | 0.788 | 0.777 | 0.772 | 0.712 | 0.298 |
| FedProx | 0.682 | 0.690 | 0.691 | 0.645 | 0.253 |
| FLTrust | 0.448 | 0.602 | 0.564 | 0.485 | 0.110 |
| FLTrust, no rescaling | 0.573 | 0.632 | 0.623 | 0.565 | 0.221 |
| FLTrust, 200-record root | 0.517 | 0.562 | 0.499 | 0.438 | 0.105 |
| RAA-FDL | 0.691 | 0.702 | 0.703 | 0.654 | 0.184 |
| FedDisco | 0.686 | 0.690 | 0.686 | 0.622 | 0.230 |
| SCAFFOLD | 0.688 | 0.699 | 0.689 | 0.582 | 0.199 |
| SCAFFOLD-N | 0.704 | 0.705 | 0.712 | 0.608 | 0.190 |
| FedAvg + LA | 0.685 | 0.708 | 0.701 | 0.678 | 0.235 |
| FedRS | 0.698 | 0.713 | 0.719 | 0.625 | 0.255 |
| SCAFFOLD-N + LA | 0.690 | 0.705 | 0.707 | 0.580 | 0.209 |
| SCAFFOLD-N + LA + RW | 0.708 | 0.709 | 0.711 | 0.633 | 0.146 |

Server-side classifier calibration (CCVR) raises FedAvg on the per-day partition to 0.657. See the paper for 5GAD, 5G-NIDD and all robustness analyses.

## Licence

MIT, see `LICENSE`.
