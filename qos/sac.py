"""Discrete Soft Actor-Critic (Christodoulou 2019) for the routing MDP.

Encoders
  flat     : MLP on the whole observation vector (simple, works with any state mode)
  deepsets : permutation-invariant encoder for the 'full' state -- each request slot goes through a shared
             MLP, slots are masked-mean-pooled per expert. Cheap stand-in for the paper's graph attention;
             it is NOT the paper's HAN.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o, layers=2):
    mods, d = [], i
    for _ in range(layers):
        mods += [nn.Linear(d, h), nn.ReLU()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class DeepSetsEnc(nn.Module):
    def __init__(self, lay, hid=64):
        super().__init__()
        self.N, self.S, self.F, self.req, self.exp = lay["N"], lay["S"], lay["F"], lay["req"], lay["exp"]
        self.slot = nn.Sequential(nn.Linear(self.F, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())
        self.out_dim = self.req + self.exp + self.N * hid

    def forward(self, x):
        B = x.shape[0]
        o = 0
        req = x[:, o:o + self.req]; o += self.req
        exp = x[:, o:o + self.exp]; o += self.exp
        sl = x[:, o:o + self.N * self.S * self.F].view(B, self.N, self.S, self.F); o += self.N * self.S * self.F
        mk = x[:, o:o + self.N * self.S].view(B, self.N, self.S, 1)
        h = self.slot(sl) * mk
        pooled = h.sum(2) / mk.sum(2).clamp(min=1.0)
        return torch.cat([req, exp, pooled.reshape(B, -1)], dim=1)


class Net(nn.Module):
    def __init__(self, obs_dim, n_act, enc, lay, hidden):
        super().__init__()
        if enc == "deepsets":
            self.enc = DeepSetsEnc(lay)
            d = self.enc.out_dim
        else:
            self.enc, d = nn.Identity(), obs_dim
        self.body = mlp(d, hidden, n_act)

    def forward(self, x):
        return self.body(self.enc(x))


class ReplayBuffer:
    def __init__(self, obs_dim, n_act=0, size=200_000, store_mask=False):
        self.o = np.zeros((size, obs_dim), np.float32); self.o2 = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros(size, np.int64); self.r = np.zeros(size, np.float32); self.d = np.zeros(size, np.float32)
        self.store_mask = store_mask
        self.m = np.ones((size, n_act), bool) if store_mask else None
        self.m2 = np.ones((size, n_act), bool) if store_mask else None
        self.size, self.n, self.i = size, 0, 0

    def add(self, o, a, r, o2, d, m=None, m2=None):
        self.o[self.i], self.a[self.i], self.r[self.i], self.o2[self.i], self.d[self.i] = o, a, r, o2, d
        if self.store_mask and m is not None and m2 is not None:
            self.m[self.i] = m
            self.m2[self.i] = m2
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, bs, dev):
        ix = np.random.randint(0, self.n, bs)
        t = lambda x: torch.as_tensor(x[ix], device=dev)
        if self.store_mask and self.m is not None:
            return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d), t(self.m), t(self.m2)
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


class DiscreteSAC:
    def __init__(self, obs_dim, n_act, lay, encoder="flat", hidden=256, lr=3e-4, gamma=0.99, tau=0.005,
                 target_entropy_scale=0.6, device="cpu"):
        self.dev, self.gamma, self.tau, self.n_act = device, gamma, tau, n_act
        self.actor = Net(obs_dim, n_act, encoder, lay, hidden).to(device)
        self.q1, self.q2 = (Net(obs_dim, n_act, encoder, lay, hidden).to(device) for _ in range(2))
        self.q1t, self.q2t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.target_entropy = target_entropy_scale * math.log(n_act)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)

    @torch.no_grad()
    def act(self, obs, deterministic=False, action_mask=None):
        lg = self.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.dev)[None])
        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.dev)[None]
            lg = torch.where(mask_t, lg, torch.tensor(-1e9, device=self.dev))
        if deterministic:
            return int(lg.argmax(-1))
        return int(torch.distributions.Categorical(logits=lg).sample())

    def update(self, batch):
        if len(batch) == 7:
            o, a, r, o2, d, m, m2 = batch
        else:
            o, a, r, o2, d = batch
            m, m2 = None, None
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            lg2 = self.actor(o2)
            if m2 is not None:
                lg2 = torch.where(m2, lg2, torch.tensor(-1e9, device=self.dev))
            p2 = F.softmax(lg2, -1)
            lp2 = torch.log(p2.clamp(min=1e-8))
            qt = torch.min(self.q1t(o2), self.q2t(o2))
            v2 = torch.where(p2 > 0, p2 * (qt - alpha * lp2), torch.zeros_like(p2)).sum(-1)
            y = r + self.gamma * (1 - d) * v2
        q1 = self.q1(o).gather(1, a[:, None]).squeeze(1)
        q2 = self.q2(o).gather(1, a[:, None]).squeeze(1)
        lq = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q.zero_grad(); lq.backward(); self.opt_q.step()

        lg = self.actor(o)
        if m is not None:
            lg = torch.where(m, lg, torch.tensor(-1e9, device=self.dev))
        p = F.softmax(lg, -1)
        lp = torch.log(p.clamp(min=1e-8))
        with torch.no_grad():
            qm = torch.min(self.q1(o), self.q2(o))
        la = torch.where(p > 0, p * (alpha * lp - qm), torch.zeros_like(p)).sum(-1).mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()

        lal = torch.where(p > 0, p.detach() * (-self.log_alpha.exp() * (lp.detach() + self.target_entropy)), torch.zeros_like(p)).sum(-1).mean()
        self.opt_al.zero_grad(); lal.backward(); self.opt_al.step()

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1t), (self.q2, self.q2t)):
                for p_, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p_)
        return lq.item(), la.item(), self.log_alpha.exp().item()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.dev))
