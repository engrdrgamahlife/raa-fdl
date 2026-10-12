"""CCVR (Luo et al., NeurIPS 2021) and prior-shifted evaluation, shared by the post-hoc scripts.
CCVR hyperparameters fixed before use: 2,000 virtual features per class, 50 epochs of SGD (lr 0.01, momentum 0.9,
batch 256) on the linear classifier only, starting from the global classifier."""
import sys, numpy as np, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", ".."))
from raafdl.models import split_head
torch.set_num_threads(1)

def predict(fe, head, Xn):
    with torch.no_grad():
        return head(fe(torch.from_numpy(Xn))).argmax(1).numpy()

def feats(fe, Xn):
    with torch.no_grad():
        return fe(torch.from_numpy(Xn)).numpy().astype(np.float64)

def ccvr(model, clients, seed, M=2000, epochs=50):
    fe, head = split_head(model)
    stats = {}
    for Xc, yc in clients:
        F = feats(fe, Xc)
        for c in np.unique(yc):
            f = F[yc == c]; n = len(f); mu = f.mean(0)
            cov = np.cov(f, rowvar=False) if n > 1 else np.zeros((F.shape[1],) * 2)
            stats.setdefault(int(c), []).append((n, mu, cov))
    rng = np.random.default_rng(seed + 31); Vs, Vy = [], []
    for c, lst in sorted(stats.items()):
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

def prior_metrics(yt, yp, labels, B, PB):
    """B = index of the benign/normal class; PB = its share of traffic in the reweighted test distribution."""
    w = np.where(yt == B, PB / (yt == B).mean(), (1 - PB) / (yt != B).mean())
    fpr = float(((yp != B) & (yt == B)).sum() / (yt == B).sum())
    dr = float(((yp != B) & (yt != B)).sum() / (yt != B).sum())
    att_prec = float((w * ((yp != B) & (yt != B))).sum() / max((w * (yp != B)).sum(), 1e-12))
    f1s = []
    for c in labels:
        tp = (w * ((yp == c) & (yt == c))).sum(); fp = (w * ((yp == c) & (yt != c))).sum(); fn = (w * ((yp != c) & (yt == c))).sum()
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return dict(benign_fpr=fpr, detection_rate=dr, attack_precision_nat=att_prec, macro_f1_nat=float(np.mean(f1s)))
