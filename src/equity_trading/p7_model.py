"""P7 training in rank space (PyTorch). Reuses the P6 CNN + Transformer and factor neutralisation.

mode 'paper': gross mean-variance (gamma) on non-tradable rank returns.
mode 'net':   Sharpe of the tradable rank-mapped stock book, net of measured costs on |change in stock weights|.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from .p6_model import CNNTransformer, _scores, _weights


def to_tensors(d: dict, rows: np.ndarray, device):
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a[rows]), dtype=dt, device=device)  # noqa: E731
    return {"x": t(np.nan_to_num(d["x"])), "mask": t(d["ok"], torch.bool), "X": t(np.nan_to_num(d["X"])),
            "ret": t(np.nan_to_num(d["ret"])), "tret": t(np.nan_to_num(d["tret"])),
            "smap": t(np.clip(d["smap"], 0, None), torch.long), "cost": t(np.nan_to_num(d["cost_stock"]))}


def returns(model, T, a: int, b: int, mode: str, prev_s=None):
    w = _weights(_scores(model, T["x"][a:b], T["mask"][a:b]), T["mask"][a:b], T["X"][a:b])
    if mode == "paper":
        return (w * T["ret"][a:b]).sum(1), None
    s = torch.zeros(b - a, T["cost"].shape[1], device=w.device).scatter_add(1, T["smap"][a:b], w)
    prev = torch.cat([prev_s[None] if prev_s is not None else torch.zeros_like(s[:1]), s[:-1]])
    c = ((s - prev).abs() * T["cost"][a:b]).sum(1)
    return (w * T["tret"][a:b]).sum(1) - c, s[-1]


def objective(r, mode: str, gamma: float):
    if mode == "paper":
        return 252 * r.mean() - 0.5 * gamma * 252 * r.var()
    return math.sqrt(252) * r.mean() / r.std().clamp(min=1e-12)


def evaluate_obj(model, T, mode, gamma, chunk):
    model.eval()
    out, prev = [], None
    with torch.no_grad():
        D = T["x"].shape[0]
        for a in range(0, D, chunk):
            b = min(a + chunk, D)
            r, prev = returns(model, T, a, b, mode, prev)
            out.append(r)
    return float(objective(torch.cat(out), mode, gamma))


def fit(d: dict, train_rows, valid_rows, seed: int, mode: str, gamma: float = 2.0, epochs: int = 100,
        patience: int = 10, lr: float = 1e-3, chunk: int = 50, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNNTransformer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Tt, Tv = to_tensors(d, train_rows, device), to_tensors(d, valid_rows, device)
    D = Tt["x"].shape[0]
    rng = np.random.default_rng(seed)
    best, best_state, bad, n = -np.inf, None, 0, 0
    for ep in range(epochs):
        model.train()
        for a in rng.permutation(np.arange(0, D, chunk)):
            b = min(a + chunk, D)
            if b - a < 5:
                continue
            r, _ = returns(model, Tt, a, b, mode)
            loss = -objective(r, mode, gamma)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate_obj(model, Tv, mode, gamma, chunk)
        n += 1
        if v > best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_valid": best, "epochs": n, "train": evaluate_obj(model, Tt, mode, gamma, chunk)}


def predict_scores(model, d: dict, rows: np.ndarray, chunk: int = 50) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    out = np.full((len(rows), d["x"].shape[1]), np.nan, np.float32)
    with torch.no_grad():
        for a in range(0, len(rows), chunk):
            r = rows[a:a + chunk]
            T = to_tensors(d, r, device)
            out[a:a + len(r)] = np.where(d["ok"][r], _scores(model, T["x"], T["mask"]).cpu().numpy(), np.nan)
    return out
