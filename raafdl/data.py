"""Data loading, splitting, federated partitioning and partition diagnostics.

Review issues addressed:
  * preprocessing must be fully specified (duplicates, NaN/inf, scaling fitted on train only);
  * the Jensen-Shannon divergence of each partition must be defined and reported;
  * the temporal (capture-day) collapse needs per-client class counts and a feature-shift
    measure, to tell label shift from covariate shift and to rule out per-day scaling artefacts.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    X: np.ndarray                 # float32 features
    y: np.ndarray                 # int64 labels 0..C-1
    group: np.ndarray             # capture day (CIC2018) or interface (5GAD); -1 if unknown
    label_names: list = field(default_factory=list)

    @property
    def labels(self):
        return list(range(len(self.label_names)))


# ---------------------------------------------------------------- loaders
CIC_DROP = ["Flow ID", "Src IP", "Dst IP", "Src Port", "Timestamp"]


def load_cic2018(path: str, max_rows_per_file: int | None = None) -> Dataset:
    """Load CSE-CIC-IDS2018 CSVs (original release or the corrected release of Liu et al. 2022).

    Steps, all reported in the paper's preprocessing paragraph:
      1. read every CSV, harmonise column names, keep the columns common to all files;
      2. derive the capture day from Timestamp (used only for the temporal partition), then drop
         identifiers (Flow ID, IPs, source port, Timestamp) that allow trivial shortcuts;
      3. drop repeated header rows, coerce to numeric, replace +/-inf by NaN, drop NaN rows;
      4. drop exact duplicate rows (feature vector + label);
      5. encode labels in a fixed, sorted order.
    Scaling is NOT done here; it is fitted on the training split only (see `split_and_scale`).
    """
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSV files in {path}")
    frames = []
    for f in files:
        df = pd.read_csv(f, nrows=max_rows_per_file, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df[df["Label"] != "Label"]                       # repeated header rows
        day = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce").dt.date
        df["__day"] = day.astype(str)
        frames.append(df)
    common = set.intersection(*(set(f.columns) for f in frames))
    df = pd.concat([f[[c for c in frames[0].columns if c in common]] for f in frames], ignore_index=True)
    df = df.drop(columns=[c for c in CIC_DROP if c in df.columns])
    feats = [c for c in df.columns if c not in ("Label", "__day")]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates(subset=feats + ["Label"])
    names = sorted(df["Label"].unique())
    y = df["Label"].map({n: i for i, n in enumerate(names)}).to_numpy(np.int64)
    days = sorted(df["__day"].unique())
    g = df["__day"].map({d: i for i, d in enumerate(days)}).to_numpy(np.int64)
    return Dataset(df[feats].to_numpy(np.float32), y, g, names)


def load_feature_csv(path: str, label_col: str = "label", group_col: str | None = None,
                     positive_values=("attack", "Attack", 1, "1")) -> Dataset:
    """Generic loader for a feature table (e.g. 5GAD features you extracted from the pcaps).

    For 5GAD the label must be binary (normal vs attack) and `group_col` the capture interface.
    """
    df = pd.read_csv(path)
    y = df[label_col].isin(positive_values).astype(np.int64).to_numpy()
    g = df[group_col].astype("category").cat.codes.to_numpy(np.int64) if group_col else np.full(len(df), -1)
    X = df.drop(columns=[c for c in (label_col, group_col) if c]).apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(X.to_numpy(np.float64)).all(axis=1)
    return Dataset(X.to_numpy(np.float32)[ok], y[ok], g[ok], ["normal", "attack"])


def make_synthetic(n=12000, n_features=20, n_classes=5, n_groups=5, shift=1.5, seed=0) -> Dataset:
    """Synthetic data with both label shift and covariate shift across groups, for smoke tests."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.0, (n_classes, n_features))
    g = rng.integers(0, n_groups, n)
    # each group over-represents two classes (label shift)
    y = np.empty(n, np.int64)
    for k in range(n_groups):
        idx = np.where(g == k)[0]
        p = np.full(n_classes, 0.05)
        p[[k % n_classes, (k + 1) % n_classes]] += 0.4
        y[idx] = rng.choice(n_classes, len(idx), p=p / p.sum())
    offsets = rng.normal(0, shift, (n_groups, n_features))          # covariate shift
    X = centers[y] + offsets[g] + rng.normal(0, 1.5, (n, n_features))
    return Dataset(X.astype(np.float32), y, g, [f"c{i}" for i in range(n_classes)])


# ---------------------------------------------------------------- splitting
@dataclass
class Splits:
    train: tuple
    val: tuple
    test: tuple
    g_train: np.ndarray
    scaler: StandardScaler


def split_and_scale(ds: Dataset, seed: int, val=0.15, test=0.15) -> Splits:
    """Stratified 70/15/15 split; the scaler is fitted on the training split only."""
    idx = np.arange(len(ds.y))
    tr, tmp = train_test_split(idx, test_size=val + test, stratify=ds.y, random_state=seed)
    va, te = train_test_split(tmp, test_size=test / (val + test), stratify=ds.y[tmp], random_state=seed)
    sc = StandardScaler().fit(ds.X[tr])
    f = lambda i: (sc.transform(ds.X[i]).astype(np.float32), ds.y[i])
    return Splits(f(tr), f(va), f(te), ds.group[tr], sc)


# ---------------------------------------------------------------- partitions
def partition(y: np.ndarray, groups: np.ndarray, scheme: str, K: int, seed: int, alpha: float = 0.5):
    """Return a list of K index arrays into the training split.

    scheme: 'iid' | 'dirichlet' | 'group' (temporal by capture day, or 5GAD interface).
    For 'group', groups are mapped to clients round-robin, so K = number of groups gives one
    group per client.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if scheme == "iid":
        return [np.sort(p) for p in np.array_split(rng.permutation(n), K)]
    if scheme == "dirichlet":
        parts = [[] for _ in range(K)]
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            props = rng.dirichlet(np.full(K, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        return [np.sort(np.array(p, dtype=np.int64)) for p in parts]
    if scheme == "group":
        ug = np.unique(groups)
        parts = [[] for _ in range(K)]
        for j, gval in enumerate(ug):
            parts[j % K].extend(np.where(groups == gval)[0].tolist())
        return [np.sort(np.array(p, dtype=np.int64)) for p in parts]
    raise ValueError(scheme)


# ---------------------------------------------------------------- diagnostics
def class_counts(y, parts, n_classes) -> np.ndarray:
    """K x C matrix of per-client class counts (report it for every partition)."""
    return np.stack([np.bincount(y[p], minlength=n_classes) for p in parts])


def jsd_report(y, parts, n_classes) -> dict:
    """Two standard definitions of partition heterogeneity (base-2 JSD, in [0, 1]).

    mean_pairwise: average JSD between the label distributions of every pair of clients.
    mean_to_global: average JSD between each client's label distribution and the pooled one.
    State in the paper which one is reported.
    """
    cc = class_counts(y, parts, n_classes).astype(float) + 1e-12
    P = cc / cc.sum(1, keepdims=True)
    glob_ = np.bincount(y, minlength=n_classes) + 1e-12
    glob_ = glob_ / glob_.sum()
    K = len(parts)
    pair = [jensenshannon(P[i], P[j], base=2) ** 2 for i in range(K) for j in range(i + 1, K)]
    to_g = [jensenshannon(P[i], glob_, base=2) ** 2 for i in range(K)]
    return {"mean_pairwise": float(np.mean(pair)) if pair else 0.0, "mean_to_global": float(np.mean(to_g))}


def feature_shift_auc(X, parts, seed=0, max_per_client=3000) -> dict:
    """Covariate shift: for each client, AUC of a logistic classifier separating that client's
    features from everyone else's. 0.5 = no shift; near 1.0 = the client is trivially separable.
    Labels are not used, so this isolates feature (covariate) shift from label shift.
    """
    rng = np.random.default_rng(seed)
    samp = [rng.choice(p, min(len(p), max_per_client), replace=False) for p in parts]
    aucs = []
    for k in range(len(parts)):
        pos = samp[k]
        neg = np.concatenate([s for j, s in enumerate(samp) if j != k])
        if len(pos) < 20 or len(neg) < 20:
            aucs.append(float("nan"))
            continue
        Xk = np.concatenate([X[pos], X[neg]])
        yk = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        Xa, Xb, ya, yb = train_test_split(Xk, yk, test_size=0.3, stratify=yk, random_state=seed)
        clf = LogisticRegression(max_iter=500).fit(Xa, ya)
        aucs.append(float(roc_auc_score(yb, clf.predict_proba(Xb)[:, 1])))
    return {"per_client_auc": aucs, "mean_auc": float(np.nanmean(aucs))}


def test_classes_missing_from_training(y_test, y_train_parts, n_classes) -> list:
    """Classes present in the test split but held by no client (an artefact check for the temporal case)."""
    held = set(np.concatenate(y_train_parts).tolist())
    return [c for c in range(n_classes) if c not in held and (y_test == c).any()]
