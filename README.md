# RAA-FDL: reliability-aware aggregation for federated intrusion detection

Code, configurations and results for the paper *Reliability-Aware Aggregation for Federated Intrusion Detection:
An Empirical Evaluation on Enterprise and 5G Core Traffic* (A. G. Gbaden, 2026).

The repository compares RAA-FDL with FedAvg, FedProx, SCAFFOLD and FLTrust on CSE-CIC-IDS2018 and 5GAD. Every
number in the paper comes from the files in `results/`.

## Contents

| Path | What it holds |
| --- | --- |
| `raafdl/` | The federated learning library: the five strategies, data partitioning, metrics, paired statistics, models |
| `experiments/cic2018/` | Sampling, preprocessing, the run script, the job list (200 runs), analysis and the figure |
| `experiments/5gad/` | Packet feature extraction, the run script, the job list (35 runs), analysis and the figure |
| `results/cic2018/` | Final metrics of every run (`final.csv`), per-round logs (`rounds.jsonl.gz`), partition diagnostics |
| `results/5gad/` | The same for 5GAD, at 50 and 100 rounds, plus the centralised baselines |

## Installation

```
pip install -r requirements.txt
```

The code runs on CPU. The paper's runs used one CPU core per run, about 10 minutes per CSE-CIC-IDS2018 run and
20 minutes per 5GAD run.

## Reproducing the CSE-CIC-IDS2018 results

1. Download the ten CSV files of "Processed Traffic Data for ML Algorithms" (6.4 GB):
   `aws s3 sync --no-sign-request --region ca-central-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" cic`
2. `cd experiments/cic2018 && python make_sample.py` writes `cic2018_sample.csv.gz` (uniform sample of up to
   20,000 records per class per day).
3. `python prep.py cic2018_sample.csv.gz 5000` removes identifiers, non-finite values, duplicates and constant
   columns, caps each class at 5,000 records, and writes `X.npy`, `y.npy`, `day.npy` and `prep_info.json`.
4. `python jobqueue.py` runs every line of `jobs.txt` (two at a time). Runs checkpoint every five rounds and resume
   after an interruption.
5. `python analyse_main.py`, `python fill_cic.py` and `python fig_cic.py` produce the tables and Fig. 2.

## Reproducing the 5GAD results

1. Get the attack-isolated captures (`Attacks/<attack>/Attacks_<attack>.pcapng`) and
   `Normal-1UE/allcap_00006_20220607091008.pcapng` from https://github.com/IdahoLabResearch/5GAD (Git LFS). Place
   them as `att/Attacks/...` and `norm/allcap_00006.pcapng` inside `experiments/5gad/`.
2. `python extract.py` writes the 135-feature packet table and its metadata.
3. `python central.py` gives the centralised baselines; `OUT=results/5gad100 ROUNDS=100 python jobqueue.py` runs the
   35 federated runs; `python fill5g.py results/5gad100` and `python fig5g.py` produce the tables and Fig. 3.

## Main results

Test macro-F1 on CSE-CIC-IDS2018 after 100 rounds (mean over five seeds):

| Strategy | IID | Dirichlet 1.0 | Dirichlet 0.5 | Dirichlet 0.1 | Temporal |
| --- | --- | --- | --- | --- | --- |
| FedAvg | 0.518 | 0.507 | 0.476 | 0.356 | 0.060 |
| FedProx | 0.517 | 0.506 | 0.475 | 0.354 | 0.052 |
| SCAFFOLD | 0.517 | 0.489 | 0.474 | 0.279 | 0.132 |
| FLTrust | 0.435 | 0.266 | 0.223 | 0.145 | 0.027 |
| RAA-FDL | 0.518 | 0.508 | 0.478 | 0.312 | 0.061 |

On 5GAD after 100 rounds, FedAvg reaches 0.938 and RAA-FDL 0.883 macro-F1. The paper explains both results.

## Licence

MIT. The datasets keep their own licences: CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity) and 5GAD (MIT).
