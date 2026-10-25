"""Generation-score and output-length predictors (stand-in for the paper's DistilBERT predictors).

Input : text embedding (+ log input length)   ->   2 heads x M models x B buckets (classification).
Paper : DistilBERT with an expert token, 10 buckets, top-1 acc 63.39% (score) / 72.97% (length).
Here  : MLP on frozen embeddings (cheap on CPU). To follow the paper more closely, fine-tune DistilBERT
        on the DGX instead and write pred_s / pred_d into the npz in the same format.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bucketize(x, xmax, B):
    return np.clip((x / xmax * B).astype(np.int64), 0, B - 1)


def centres(b, xmax, B):
    return (b + 0.5) * xmax / B


class PredNet(nn.Module):
    def __init__(self, d_in, M, B, h=256):
        super().__init__()
        self.M, self.B = M, B
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head_s, self.head_d = nn.Linear(h, M * B), nn.Linear(h, M * B)

    def forward(self, x):
        z = self.net(x)
        return self.head_s(z).view(-1, self.M, self.B), self.head_d(z).view(-1, self.M, self.B)


def train_predictors(npz_path: str, B: int = 10, epochs: int = 30, lr: float = 1e-3, device: str = "cpu", seed: int = 0):
    torch.manual_seed(seed)
    z = dict(np.load(npz_path, allow_pickle=False))
    score, dlen, p, split = z["score"], z["dlen"], z["p"], z["split"]
    smax, dmax = float(score.max()), float(np.percentile(dlen, 99))
    ys = bucketize(score, smax, B)
    yd = bucketize(np.minimum(dlen, dmax), dmax, B)
    X = np.concatenate([z["emb"], np.log1p(p)[:, None] / 7.0], axis=1).astype(np.float32)
    mu, sd = X[split == 0].mean(0), X[split == 0].std(0) + 1e-6
    X = (X - mu) / sd
    tr, ev = np.where(split == 0)[0], np.where(split == 1)[0]
    Xt = torch.tensor(X, device=device)
    ys_t, yd_t = torch.tensor(ys, device=device), torch.tensor(yd, device=device)
    net = PredNet(X.shape[1], score.shape[1], B).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep).permutation(tr)
        for i in range(0, len(perm), 256):
            b = torch.tensor(perm[i:i + 256], device=device)
            ls, ld = net(Xt[b])
            loss = F.cross_entropy(ls.reshape(-1, B), ys_t[b].reshape(-1)) + F.cross_entropy(ld.reshape(-1, B), yd_t[b].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        ls, ld = net(Xt)
    top1 = lambda lg, y: float((lg.argmax(-1) == y).float()[torch.tensor(ev)].mean())
    top3 = lambda lg, y: float((lg.topk(3, -1).indices == y[..., None]).any(-1).float()[torch.tensor(ev)].mean())
    stats = dict(score_top1=top1(ls, ys_t), score_top3=top3(ls, ys_t), len_top1=top1(ld, yd_t), len_top3=top3(ld, yd_t))
    z["pred_s"] = centres(ls.argmax(-1).cpu().numpy(), smax, B).astype(np.float32)
    z["pred_d"] = centres(ld.argmax(-1).cpu().numpy(), dmax, B).astype(np.float32)
    z["orc_s"] = centres(ys, smax, B).astype(np.float32)
    z["orc_d"] = centres(yd, dmax, B).astype(np.float32)
    np.savez_compressed(npz_path, **z)
    return stats
