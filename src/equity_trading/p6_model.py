"""P6 network (PyTorch): CNN + Transformer signal extractor, end-to-end net-Sharpe objective with factor neutralisation.

Protocol: discussions/Deep_Statistical_Arbitrage/001_claude_p6_deep_stat_arb_preregistration.md
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn


class CNNTransformer(nn.Module):
    def __init__(self, filters: int = 8, heads: int = 4, ff: int = 16):
        super().__init__()
        self.c1 = nn.Conv1d(1, filters, 2)
        self.c2 = nn.Conv1d(filters, filters, 2, padding=1)
        layer = nn.TransformerEncoderLayer(filters, heads, ff, dropout=0.0, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, 1)
        self.head = nn.Linear(filters, 1)

    def forward(self, x):                                  # x: (M, L)
        h = torch.relu(self.c1(x[:, None, :]))            # (M, F, L-1)
        h = h + torch.relu(self.c2(h))[..., : h.shape[-1]]
        h = self.tr(h.transpose(1, 2))                    # (M, L-1, F)
        return self.head(h[:, -1]).squeeze(-1)            # (M,)


def _weights(score, mask, X):
    """score, mask: (D, N); X: (D, N, K). Demean over mask, neutralise to X, scale sum|w| = 1."""
    m = mask.float()
    n = m.sum(1, keepdim=True).clamp(min=1)
    w = (score - (score * m).sum(1, keepdim=True) / n) * m
    Xm = X * m[..., None]
    XtX = Xm.transpose(1, 2) @ Xm + 1e-6 * torch.eye(X.shape[-1], device=X.device)
    beta = torch.linalg.solve(XtX, (Xm.transpose(1, 2) @ w[..., None]))
    w = (w - (Xm @ beta).squeeze(-1)) * m
    return w / w.abs().sum(1, keepdim=True).clamp(min=1e-12)


def _scores(model, x, mask):
    D, N, L = x.shape
    flat = x[mask]
    s = torch.zeros(D, N, device=x.device)
    if flat.shape[0]:
        s[mask] = model(flat)
    return s


def net_returns(model, x, mask, X, ret, cost, w_prev=None):
    """Daily net portfolio returns for consecutive decision days (turnover against the previous day's weights)."""
    w = _weights(_scores(model, x, mask), mask, X)
    gross = (w * ret).sum(1)
    prev = torch.cat([w_prev[None] if w_prev is not None else torch.zeros_like(w[:1]), w[:-1]])
    c = ((w - prev).abs() * cost).sum(1)
    return gross - c, w


def sharpe(r):
    return math.sqrt(252) * r.mean() / r.std().clamp(min=1e-12)


def to_tensors(d: dict, rows: np.ndarray, device):
    """Slice the day-level numpy arrays for decision days `rows` (targets already aligned to each row)."""
    t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a[rows]), dtype=dt, device=device)  # noqa: E731
    return {"x": t(np.nan_to_num(d["x"])), "mask": t(d["ok"], torch.bool), "X": t(np.nan_to_num(d["X"])),
            "ret": t(np.nan_to_num(d["ret"])), "cost": t(np.nan_to_num(d["cost"]))}


def evaluate_sharpe(model, T, chunk: int) -> float:
    model.eval()
    out, prev = [], None
    with torch.no_grad():
        D = T["x"].shape[0]
        for a in range(0, D, chunk):
            b = min(a + chunk, D)
            r, w = net_returns(model, T["x"][a:b], T["mask"][a:b], T["X"][a:b], T["ret"][a:b], T["cost"][a:b], prev)
            prev = w[-1]
            out.append(r)
    return float(sharpe(torch.cat(out)))


def fit(d: dict, train_rows, valid_rows, seed: int, epochs: int = 100, patience: int = 10, lr: float = 1e-3,
        chunk: int = 50, device: str | None = None, verbose: bool = False) -> tuple[nn.Module, dict]:
    """Train on contiguous chunks of `chunk` decision days (shuffled order each epoch); keep the epoch with the best
    validation net Sharpe."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNNTransformer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Tt, Tv = to_tensors(d, train_rows, device), to_tensors(d, valid_rows, device)
    D = Tt["x"].shape[0]
    starts = np.arange(0, D, chunk)
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -np.inf, None, 0, []
    for ep in range(epochs):
        model.train()
        for a in rng.permutation(starts):
            b = min(a + chunk, D)
            if b - a < 5:
                continue
            r, _ = net_returns(model, Tt["x"][a:b], Tt["mask"][a:b], Tt["X"][a:b], Tt["ret"][a:b], Tt["cost"][a:b])
            loss = -sharpe(r)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate_sharpe(model, Tv, chunk)
        hist.append(v)
        if verbose:
            print(f"    epoch {ep}: valid net Sharpe {v:.3f}", flush=True)
        if v > best:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_valid_sharpe": best, "epochs": len(hist), "train_sharpe": evaluate_sharpe(model, Tt, chunk)}


def predict_scores(model, d: dict, rows: np.ndarray, chunk: int = 50, device: str | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = np.full((len(rows), d["x"].shape[1]), np.nan, np.float32)
    with torch.no_grad():
        for a in range(0, len(rows), chunk):
            r = rows[a:a + chunk]
            T = to_tensors(d, r, device)
            s = _scores(model, T["x"], T["mask"]).cpu().numpy()
            out[a:a + len(r)] = np.where(d["ok"][r], s, np.nan)
    return out
