"""Federated learning simulator: FedAvg, FedProx, SCAFFOLD, RAA-FDL, FLTrust, FedAvg+server fine-tuning,
SCAFFOLD-N (size-weighted SCAFFOLD updates), raa_scaffold (SCAFFOLD local training + RAA-FDL weights on the
updates), FedAvgM and FedAdam (server optimisers on the size-weighted mean update) and FedDisco (discrepancy-aware
weights). Local-loss options: logit adjustment (la_tau, la_eps, la_multi_only) and FedRS restricted softmax. FLConfig.logit_adjust=True adds the client's add-one smoothed log label prior to the logits during local
training (logit adjustment / balanced softmax) for any strategy.

Design choices that the review asked to make explicit (all configurable, all logged):
  * identical global initialisation per seed for every strategy;
  * identical, seeded client-dropout pattern for every strategy (participation gate: a dropped
    client contributes neither weight nor update);
  * BatchNorm running statistics are aggregated together with the parameters;
  * FedProx's proximal term uses live local parameters against a constant copy of the global model;
  * SCAFFOLD supports Option I and Option II control variates;
  * RAA-FDL implements Eqs. (1)-(5) of the revised manuscript, with S = 1 / (1 + CV) over a
    trailing window and D = (1 + cos) / 2;
  * FLTrust and FedAvg+FT receive the same server-held anchor data as RAA-FDL, so the
    comparison is fair with respect to server-side information.
"""
from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .metrics import evaluate

STRATEGIES = ("fedavg", "fedprox", "scaffold", "raa", "fltrust", "fedavg_ft", "scaffold_n", "raa_scaffold",
              "fedavgm", "fedadam", "feddisco")
SERVER_OPT = ("fedavgm", "fedadam")
SCAFFOLD_TYPE = ("scaffold", "scaffold_n", "raa_scaffold")
RAA_TYPE = ("raa", "raa_scaffold")


@dataclass
class FLConfig:
    rounds: int = 100
    local_epochs: int = 3
    batch: int = 256
    lr: float = 0.01
    optimizer: str = "sgd"              # 'sgd' | 'adam'
    mu: float = 0.1                     # FedProx
    scaffold_option: str = "II"         # 'I' | 'II'
    drop_probs: list | None = None      # per-client probability of failing a round
    eval_every: int = 1
    # RAA-FDL (Eqs. 1-5)
    beta: tuple = (0.4, 0.2, 0.2, 0.2)  # (A, S, C, D); set a term to 0 for leave-one-out
    lam: float = 0.5
    tau: float = 0.15
    gamma: float = 0.5
    wmin: float = 0.02
    window: int = 5
    r0: float | None = None             # initial smoothed reliability; None -> 1/K (as in the paper)
    # server-side training for fltrust / fedavg_ft
    server_epochs: int = 1
    # logit adjustment (balanced softmax) in local training: add log of the client's smoothed label prior
    logit_adjust: bool = False
    la_tau: float = 1.0                 # scale of the log-prior offset (Menon et al.'s tau)
    la_eps: float = 1.0                 # additive smoothing of the client label counts
    la_multi_only: bool = False         # apply LA only on clients that hold >= 2 classes
    fedrs_alpha: float | None = None    # FedRS: scale logits of locally absent classes by alpha
    # server optimisers (FedAvgM, FedAdam; Hsu et al. 2019, Reddi et al. 2021)
    server_lr: float = 1.0
    server_momentum: float = 0.9
    adam_b1: float = 0.9
    adam_b2: float = 0.99
    adam_tau: float = 1e-3
    # FedDisco (Ye et al. 2023): p_k ~ ReLU(n_k - a d_k + b)
    disco_a: float = 0.5
    disco_b: float = 0.1
    fltrust_rescale: bool = True        # FLTrust's norm rescaling of client updates
    threads: int = 0                    # torch threads; 0 = library default

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ helpers
def _batches(n, batch, gen):
    perm = torch.randperm(n, generator=gen)
    for s in range(0, n, batch):
        yield perm[s:s + batch]


def _make_opt(params, cfg):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    return torch.optim.SGD(params, lr=cfg.lr)


def predict(model, X, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[s:s + batch])).argmax(1).numpy())
    return np.concatenate(out)


def local_train(model, X, y, cfg, seed, prox_ref=None, scaffold_c=None, scaffold_ci=None):
    """Train in place; returns the number of optimiser steps taken."""
    model.train()
    params = [p for p in model.parameters()]
    opt = _make_opt(params, cfg)
    gen = torch.Generator().manual_seed(int(seed))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    lossf = nn.CrossEntropyLoss()
    steps = 0
    la = rs = None
    if cfg.logit_adjust or cfg.fedrs_alpha is not None:
        model.eval()
        with torch.no_grad():
            C = model(Xt[:2]).shape[1]
        cnt = torch.bincount(yt, minlength=C).float()
        model.train()
        if cfg.logit_adjust and not (cfg.la_multi_only and int((cnt > 0).sum()) < 2):
            la = cfg.la_tau * torch.log((cnt + cfg.la_eps) / (cnt.sum() + C * cfg.la_eps))
        if cfg.fedrs_alpha is not None:
            rs = torch.where(cnt > 0, torch.ones(C), torch.full((C,), float(cfg.fedrs_alpha)))
    for _ in range(cfg.local_epochs):
        for idx in _batches(len(yt), cfg.batch, gen):
            if len(idx) < 2:
                continue                       # BatchNorm needs >1 sample
            out = model(Xt[idx])
            if rs is not None:
                out = out * rs
            if la is not None:
                out = out + la
            loss = lossf(out, yt[idx])
            if prox_ref is not None:           # live params vs constant global copy
                loss = loss + 0.5 * cfg.mu * sum(((p - r) ** 2).sum() for p, r in zip(params, prox_ref))
            opt.zero_grad()
            loss.backward()
            if scaffold_c is not None:
                for p, c, ci in zip(params, scaffold_c, scaffold_ci):
                    p.grad.add_(c - ci)
            opt.step()
            steps += 1
    return steps


def full_gradient(model, X, y, max_batches=10, batch=1024, seed=0):
    """Average gradient at the current parameters (SCAFFOLD Option I)."""
    model.train()
    gen = torch.Generator().manual_seed(int(seed))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    grads = [torch.zeros_like(p) for p in model.parameters()]
    nb = 0
    for idx in _batches(len(yt), batch, gen):
        if len(idx) < 2:
            continue
        model.zero_grad()
        nn.CrossEntropyLoss()(model(Xt[idx]), yt[idx]).backward()
        for g, p in zip(grads, model.parameters()):
            g.add_(p.grad)
        nb += 1
        if nb >= max_batches:
            break
    return [g / max(nb, 1) for g in grads]


def weighted_state(states, weights):
    """Weighted average of full state dicts (parameters AND BatchNorm buffers)."""
    out = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            out[k] = sum(w * s[k] for s, w in zip(states, weights))
        else:                                   # e.g. num_batches_tracked
            out[k] = states[0][k].clone()
    return out


# ------------------------------------------------------------------ RAA-FDL
class RAATracker:
    """Server-side reliability state for RAA-FDL (Eqs. 1-4)."""

    def __init__(self, K, cfg: FLConfig):
        self.cfg = cfg
        self.R = np.full(K, cfg.r0 if cfg.r0 is not None else 1.0 / K)
        self.norms = [[] for _ in range(K)]
        self.hist = [[] for _ in range(K)]      # 1 = delivered update, 0 = failed round

    def record(self, i, success):
        self.hist[i].append(1 if success else 0)

    def weights(self, part, deltas, anchor_f1, sizes):
        c = self.cfg
        W = c.window
        mean_delta = torch.stack([deltas[i] for i in part]).mean(0)
        comps = {}
        for i in part:
            self.norms[i].append(float(deltas[i].norm()))
            h = np.array(self.norms[i][-W:])
            cv = float(h.std(ddof=0) / h.mean()) if len(h) >= 2 and h.mean() > 0 else 0.0
            S = 1.0 / (1.0 + cv)
            C = float(np.mean(self.hist[i][-W:])) if self.hist[i] else 1.0
            cos = float(torch.nn.functional.cosine_similarity(deltas[i], mean_delta, dim=0))
            D = (1.0 + cos) / 2.0
            A = anchor_f1[i]
            r = c.beta[0] * A + c.beta[1] * S + c.beta[2] * C + c.beta[3] * D          # Eq. 1
            self.R[i] = c.lam * r + (1 - c.lam) * self.R[i]                             # Eq. 2
            comps[i] = dict(A=A, S=S, C=C, D=D, R=r, R_s=float(self.R[i]))
        logits = np.array([c.gamma * math.log(sizes[i]) + self.R[i] / c.tau for i in part])
        w = np.exp(logits - logits.max())
        w = w / w.sum()                                                                  # Eq. 3
        w = np.maximum(w, c.wmin)
        w = w / w.sum()                                                                  # Eq. 4
        for i, wi in zip(part, w):
            comps[i]["w"] = float(wi)
        return w, comps


# ------------------------------------------------------------------ main loop
def run_federated(strategy, model_fn, clients, val, test, labels, cfg: FLConfig, seed,
                  anchor=None, log=None, ckpt_path=None, ckpt_every=5):
    """Run one strategy for one seed. Returns a list of per-round records.

    clients: list of (X, y) numpy arrays. val/test/anchor: (X, y). labels: the dataset's classes.
    """
    assert strategy in STRATEGIES, strategy
    if strategy in ("raa", "raa_scaffold", "fltrust", "fedavg_ft"):
        assert anchor is not None, f"{strategy} needs server-held anchor data"
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    K = len(clients)
    sizes = np.array([len(c[1]) for c in clients], dtype=float)
    torch.manual_seed(seed)
    gmodel = model_fn()                                    # identical init per seed
    drop = np.array(cfg.drop_probs if cfg.drop_probs is not None else [0.0] * K)
    part_rng = np.random.default_rng([seed, 7919])         # identical dropout pattern per seed
    tracker = RAATracker(K, cfg) if strategy in RAA_TYPE else None
    pnames = [n for n, _ in gmodel.named_parameters()]
    if strategy in SCAFFOLD_TYPE:
        c_glob = [torch.zeros_like(p) for p in gmodel.parameters()]
        c_loc = [[torch.zeros_like(p) for p in gmodel.parameters()] for _ in range(K)]
    sopt = {"m": None, "v": None}                          # server optimiser state
    if strategy == "feddisco":
        C_ = len(labels)
        cnts = np.stack([np.bincount(c[1], minlength=C_) for c in clients]).astype(float)
        # Ye et al. (2023) defaults: d_k = KL(D_k || uniform), rescaled to sum to 1 over clients (official code)
        Dk = cnts / np.maximum(cnts.sum(1, keepdims=True), 1)
        disco_d = np.array([float(np.sum(np.where(Dk[i] > 0, Dk[i] * np.log(Dk[i] * C_), 0.0))) for i in range(K)])
        disco_d = disco_d / disco_d.sum() if disco_d.sum() > 0 else disco_d
    records = []
    t0 = time.time()
    start = 1
    if ckpt_path and os.path.exists(ckpt_path):            # exact resume after a restart
        ck = torch.load(ckpt_path, weights_only=False)
        gmodel.load_state_dict(ck["model"])
        part_rng.bit_generator.state = ck["rng"]
        records = ck["records"]
        t0 = time.time() - ck["elapsed"]
        start = ck["round"] + 1
        if tracker:
            tracker.R, tracker.norms, tracker.hist = ck["tracker"]
        if strategy in SCAFFOLD_TYPE:
            c_glob, c_loc = ck["scaffold"]
        if ck.get("sopt") is not None:
            sopt = ck["sopt"]
    for t in range(start, cfg.rounds + 1):
        u = part_rng.random(K)
        part = [i for i in range(K) if u[i] >= drop[i]]
        if tracker:
            for i in range(K):
                if i not in part:
                    tracker.record(i, False)
        rec = {"round": t, "strategy": strategy, "seed": seed, "n_participating": len(part)}
        if part:
            g_state = copy.deepcopy(gmodel.state_dict())
            g_vec = parameters_to_vector(gmodel.parameters()).detach().clone()
            states, deltas, steps = {}, {}, {}
            for i in part:
                m = model_fn()
                m.load_state_dict(g_state)
                kw = {}
                if strategy == "fedprox":
                    kw["prox_ref"] = [p.detach().clone() for p in gmodel.parameters()]
                if strategy in SCAFFOLD_TYPE:
                    kw["scaffold_c"], kw["scaffold_ci"] = c_glob, c_loc[i]
                steps[i] = local_train(m, *clients[i], cfg, seed * 100003 + t * 1009 + i, **kw)
                states[i] = copy.deepcopy(m.state_dict())
                deltas[i] = parameters_to_vector(m.parameters()).detach() - g_vec
                if strategy in SCAFFOLD_TYPE:
                    if cfg.scaffold_option == "I":
                        m.load_state_dict(g_state)
                        new_ci = full_gradient(m, *clients[i], seed=seed + t)
                    else:
                        denom = max(steps[i], 1) * cfg.lr
                        pv = dict(zip(pnames, [p.detach() for p in m.parameters()]))
                        new_ci = [ci - c - (pv[n] - g_state[n]) / denom
                                  for ci, c, n in zip(c_loc[i], c_glob, pnames)]
                    states[i]["__dc"] = [n - o for n, o in zip(new_ci, c_loc[i])]
                    c_loc[i] = new_ci
                if tracker:
                    tracker.record(i, True)
            n_w = sizes[part] / sizes[part].sum()
            plist = [states[i] for i in part]
            if strategy in SERVER_OPT:
                buf = weighted_state([{k: v for k, v in s.items() if k != "__dc"} for s in plist], n_w)
                gmodel.load_state_dict(buf)                              # BatchNorm buffers size-weighted
                d = sum(float(wi) * deltas[i] for wi, i in zip(n_w, part))
                if strategy == "fedavgm":
                    sopt["m"] = d if sopt["m"] is None else cfg.server_momentum * sopt["m"] + d
                    step = cfg.server_lr * sopt["m"]
                else:
                    if sopt["m"] is None:
                        sopt["m"] = torch.zeros_like(d); sopt["v"] = torch.full_like(d, cfg.adam_tau ** 2)
                    sopt["m"] = cfg.adam_b1 * sopt["m"] + (1 - cfg.adam_b1) * d
                    sopt["v"] = cfg.adam_b2 * sopt["v"] + (1 - cfg.adam_b2) * d * d
                    step = cfg.server_lr * sopt["m"] / (sopt["v"].sqrt() + cfg.adam_tau)
                vector_to_parameters(g_vec + step, gmodel.parameters())
            elif strategy == "feddisco":
                nn_ = sizes[part] / sizes[part].sum()
                pw = np.maximum(nn_ - cfg.disco_a * disco_d[part] + cfg.disco_b, 0.0)
                pw = pw / pw.sum() if pw.sum() > 0 else nn_
                gmodel.load_state_dict(weighted_state([{k: v for k, v in s.items() if k != "__dc"} for s in plist], pw))
                rec["disco_w"] = [round(float(x), 4) for x in pw]
            elif strategy in ("fedavg", "fedprox", "fedavg_ft"):
                new = weighted_state([{k: v for k, v in s.items() if k != "__dc"} for s in plist], n_w)
                gmodel.load_state_dict(new)
            elif strategy in SCAFFOLD_TYPE:
                if strategy == "scaffold":
                    agg_w = np.full(len(part), 1.0 / len(part))         # original SCAFFOLD: uniform mean
                elif strategy == "scaffold_n":
                    agg_w = n_w                                          # data-size weighted deltas
                else:                                                    # raa_scaffold: reliability weights
                    a_f1 = {}
                    for i in part:
                        m = model_fn()
                        m.load_state_dict({k: v for k, v in states[i].items() if k != "__dc"})
                        a_f1[i] = evaluate(anchor[1], predict(m, anchor[0]), labels)["macro_f1"]
                    agg_w, comps = tracker.weights(part, deltas, a_f1, sizes)
                    rec["raa"] = {int(i): comps[i] for i in part}
                buf = weighted_state([{k: v for k, v in s.items() if k != "__dc"} for s in plist],
                                     n_w if strategy == "scaffold" else agg_w)   # original: buffers N-weighted
                gmodel.load_state_dict(buf)
                vector_to_parameters(g_vec + sum(float(wi) * deltas[i] for wi, i in zip(agg_w, part)),
                                     gmodel.parameters())                # x <- x + sum_i w_i (y_i - x)
                dcs = [states[i]["__dc"] for i in part]
                for j in range(len(c_glob)):
                    c_glob[j] = c_glob[j] + (len(part) / K) * sum(d[j] for d in dcs) / len(part)
            elif strategy == "raa":
                a_f1 = {}
                for i in part:
                    m = model_fn()
                    m.load_state_dict(states[i])
                    a_f1[i] = evaluate(anchor[1], predict(m, anchor[0]), labels)["macro_f1"]
                w, comps = tracker.weights(part, deltas, a_f1, sizes)
                gmodel.load_state_dict(weighted_state(plist, w))
                rec["raa"] = {int(i): comps[i] for i in part}
            elif strategy == "fltrust":
                m = model_fn()
                m.load_state_dict(g_state)
                ecfg = copy.copy(cfg)
                ecfg.local_epochs = cfg.server_epochs
                local_train(m, *anchor, ecfg, seed * 7 + t)
                g0 = parameters_to_vector(m.parameters()).detach() - g_vec
                ts, upd = [], torch.zeros_like(g_vec)
                for i in part:
                    cs = float(torch.nn.functional.cosine_similarity(deltas[i], g0, dim=0))
                    tsi = max(cs, 0.0)
                    ts.append(tsi)
                    scale = (g0.norm() / (deltas[i].norm() + 1e-12)) if cfg.fltrust_rescale else 1.0
                    upd += tsi * deltas[i] * scale
                buf = weighted_state(plist, n_w)
                gmodel.load_state_dict(buf)
                vector_to_parameters(g_vec + (upd / sum(ts) if sum(ts) > 0 else 0 * upd), gmodel.parameters())
                rec["fltrust_ts_mean"] = float(np.mean(ts))
            if strategy == "fedavg_ft":
                ecfg = copy.copy(cfg)
                ecfg.local_epochs = cfg.server_epochs
                local_train(gmodel, *anchor, ecfg, seed * 11 + t)
        if t % cfg.eval_every == 0 or t == cfg.rounds:
            rec["val_macro_f1"] = evaluate(val[1], predict(gmodel, val[0]), labels)["macro_f1"]
            ev = evaluate(test[1], predict(gmodel, test[0]), labels)
            rec["test_macro_f1"], rec["test_accuracy"] = ev["macro_f1"], ev["accuracy"]
            if t == cfg.rounds:
                rec["test_per_class"] = ev["per_class"]
        rec["elapsed_s"] = round(time.time() - t0, 2)
        records.append(rec)
        if log:
            log(rec)
        if ckpt_path and (t % ckpt_every == 0 or t == cfg.rounds):
            ck = {"model": gmodel.state_dict(), "rng": part_rng.bit_generator.state, "records": records,
                  "elapsed": time.time() - t0, "round": t,
                  "tracker": (tracker.R, tracker.norms, tracker.hist) if tracker else None,
                  "scaffold": (c_glob, c_loc) if strategy in SCAFFOLD_TYPE else None,
                  "sopt": sopt if strategy in SERVER_OPT else None}
            torch.save(ck, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    return records


def rounds_to_target(records, target, key="val_macro_f1"):
    """First round at which `key` reaches `target` (None if never): the convergence-speed metric."""
    for r in records:
        if r.get(key) is not None and r[key] >= target:
            return r["round"]
    return None
