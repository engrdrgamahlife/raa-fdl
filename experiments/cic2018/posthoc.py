"""Post-hoc analyses from the round-100 checkpoints (no retraining of the federation):
  * CCVR (Luo et al., NeurIPS 2021): server-side classifier calibration with virtual features sampled from
    client-reported class means/covariances of the global model's penultimate features;
  * evaluation at a benign-dominated class prior (benign = 83% of traffic, attacks keep their relative shares):
    benign false-positive rate, attack detection rate, attack precision, prior-weighted macro-F1.
Usage: python posthoc.py <results_dir> <tag> [scenario ...]   -> appends to <results_dir>/posthoc.csv"""
import sys, os, json, numpy as np, pandas as pd, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
from raafdl import data as D
from raafdl.models import build_model, split_head
from raafdl.metrics import evaluate
torch.set_num_threads(1)
OUT, TAG = sys.argv[1], sys.argv[2]
SCN = sys.argv[3:] or ["iid", "dir1.0", "dir0.5", "dir0.1", "temporal"]
X, y, day = np.load("X.npy"), np.load("y.npy"), np.load("day.npy")
info = json.load(open("prep_info.json")); names = info["label_names"]; C = len(names); labels = list(range(C))
B = names.index("Benign"); PB = 0.83
ds = D.Dataset(X, y, day, names)

def build(seed, sc):
    sp = D.split_and_scale(ds, seed); ytr = sp.train[1]
    if sc == "iid": parts = D.partition(ytr, sp.g_train, "iid", 10, seed)
    elif sc.startswith("dir"): parts = D.partition(ytr, sp.g_train, "dirichlet", 10, seed, float(sc[3:]))
    else: parts = D.partition(ytr, sp.g_train, "group", len(np.unique(day)), seed)
    parts = [p for p in parts if len(p) > 0]
    return sp, [(sp.train[0][p], ytr[p]) for p in parts]

def predict(fe, head, Xn):
    with torch.no_grad():
        return head(fe(torch.from_numpy(Xn))).argmax(1).numpy()

def feats(fe, Xn):
    with torch.no_grad():
        return fe(torch.from_numpy(Xn)).numpy().astype(np.float64)

def ccvr(model, clients, seed, M=2000, epochs=50):
    fe, head = split_head(model)
    stats = {}
    for Xc, yc in clients:                                   # what each client would report
        F = feats(fe, Xc)
        for c in np.unique(yc):
            f = F[yc == c]; n = len(f); mu = f.mean(0)
            cov = np.cov(f, rowvar=False) if n > 1 else np.zeros((F.shape[1],) * 2)
            stats.setdefault(c, []).append((n, mu, cov))
    rng = np.random.default_rng(seed + 31); Vs, Vy = [], []
    for c, lst in stats.items():                             # server: global moments per class
        N = sum(n for n, _, _ in lst); mu = sum(n * m for n, m, _ in lst) / N
        if N > 1:
            S = sum((n - 1) / (N - 1) * cv + n / (N - 1) * np.outer(m, m) for n, m, cv in lst) - N / (N - 1) * np.outer(mu, mu)
        else:
            S = np.zeros((len(mu), len(mu)))
        S = (S + S.T) / 2 + 1e-6 * np.eye(len(mu))
        Vs.append(rng.multivariate_normal(mu, S, size=M, method="eigh")); Vy.append(np.full(M, c))
    V = torch.tensor(np.concatenate(Vs), dtype=torch.float32); Vy = torch.tensor(np.concatenate(Vy))
    h = torch.nn.Linear(head.in_features, head.out_features); h.load_state_dict(head.state_dict())
    opt = torch.optim.SGD(h.parameters(), lr=0.01, momentum=0.9); g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Vy), generator=g)
        for s in range(0, len(perm), 256):
            i = perm[s:s + 256]; opt.zero_grad()
            torch.nn.functional.cross_entropy(h(V[i]), Vy[i]).backward(); opt.step()
    return fe, h

def prior_metrics(yt, yp):
    w = np.where(yt == B, PB / (yt == B).mean(), (1 - PB) / (yt != B).mean())
    fpr = float(((yp != B) & (yt == B)).sum() / (yt == B).sum())
    dr = float(((yp != B) & (yt != B)).sum() / (yt != B).sum())
    att_prec = float((w * ((yp != B) & (yt != B))).sum() / max((w * (yp != B)).sum(), 1e-12))
    f1s = []
    for c in labels:
        tp = (w * ((yp == c) & (yt == c))).sum(); fp = (w * ((yp == c) & (yt != c))).sum(); fn = (w * ((yp != c) & (yt == c))).sum()
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return dict(benign_fpr=fpr, detection_rate=dr, attack_precision_nat=att_prec, macro_f1_nat=float(np.mean(f1s)))

rows = []
for sc in SCN:
    for seed in [42, 43, 44, 45, 46]:
        ck = f"{OUT}/ckpt_{sc}_{TAG}_{seed}.pt"
        if not os.path.exists(ck): continue
        sp, clients = build(seed, sc)
        m = build_model("cnn", X.shape[1], C); m.load_state_dict(torch.load(ck, weights_only=False)["model"]); m.eval()
        fe, head = split_head(m)
        yp = predict(fe, head, sp.test[0]); ev = evaluate(sp.test[1], yp, labels)
        r = dict(scenario=sc, tag=TAG, seed=seed, test_macro_f1=ev["macro_f1"], **prior_metrics(sp.test[1], yp))
        fe2, h2 = ccvr(m, clients, seed)
        yv2 = predict(fe2, h2, sp.val[0]); yp2 = predict(fe2, h2, sp.test[0])
        r.update(ccvr_val_macro_f1=evaluate(sp.val[1], yv2, labels)["macro_f1"], ccvr_test_macro_f1=evaluate(sp.test[1], yp2, labels)["macro_f1"],
                 **{"ccvr_" + k: v for k, v in prior_metrics(sp.test[1], yp2).items()})
        rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
fp = f"{OUT}/posthoc.csv"
pd.DataFrame(rows).to_csv(fp, mode="a", header=not os.path.exists(fp), index=False)
