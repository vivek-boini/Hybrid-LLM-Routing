"""RL environment + state builder.

State modes
  expert_only : per expert (memory utilisation, #running/R, #waiting/W)       -> paper's "Baseline RL" state
  full        : arriving-request features + expert features + per-request slot features (padded)
                -> flat-vector stand-in for the paper's heterogeneous-graph state (no graph attention here;
                   the DeepSets encoder in sac.py is a permutation-invariant alternative)
Reward modes
  baseline : QoS of requests that completed since the last decision                (paper's Baseline RL)
  qos      : the above  -  QoS "put at risk" on the chosen expert (impact estimator) (paper: QoS-aware reward)
"""
from __future__ import annotations

import numpy as np

from .sim import ExpertCfg, Simulator
from .workload import make_arrivals

SLOT_F = 6


class StateBuilder:
    def __init__(self, sim: Simulator, mode: str, baseline_include_req: bool = False):
        self.mode, self.N, self.S = mode, sim.N, sim.slots
        self.baseline_include_req = baseline_include_req
        self.req_dim = 1 + 2 * self.N
        self.exp_dim = 3 * self.N
        self.slot_dim = self.N * self.S * SLOT_F
        self.mask_dim = self.N * self.S
        if mode == "expert_only":
            self.obs_dim = (self.req_dim + self.exp_dim) if baseline_include_req else self.exp_dim
        else:
            self.obs_dim = self.req_dim + self.exp_dim + self.slot_dim + self.mask_dim

    def layout(self) -> dict:
        return dict(N=self.N, S=self.S, F=SLOT_F, req=self.req_dim, exp=self.exp_dim, mode=self.mode,
                    baseline_include_req=self.baseline_include_req)

    def build(self, sim: Simulator, req) -> np.ndarray:
        N, S = self.N, self.S
        exp = np.zeros((N, 3), np.float32)
        for n, e in enumerate(sim.experts):
            exp[n] = (e.util(), len(e.running) / sim.run_cap, len(e.waiting) / sim.wait_cap)
        if self.mode == "expert_only" and not self.baseline_include_req:
            return exp.ravel()
        d = sim.data
        ids = sim.ids
        reqf = np.concatenate([[min(req.p / 1024.0, 2.0)], d.act_s[req.idx, ids], d.act_d[req.idx, ids] / 512.0]).astype(np.float32)
        if self.mode == "expert_only" and self.baseline_include_req:
            return np.concatenate([reqf, exp.ravel()])
        slots = np.zeros((N, S, SLOT_F), np.float32)
        mask = np.zeros((N, S), np.float32)
        for n, e in enumerate(sim.experts):
            cap = e.cfg.cap_tokens
            for k, r in enumerate(e.running):
                l_t = (sim.t - r.arrival) / max(1, r.d_done)
                slots[n, k] = (min(r.p / 1024.0, 2.0), r.pred_s, r.pred_d / 512.0, (r.p + r.d_done) / cap,
                               r.d_done / 512.0, min(l_t / sim.L, 3.0) / 3.0)
                mask[n, k] = 1.0
            for k, r in enumerate(e.waiting):
                kk = sim.run_cap + k
                slots[n, kk] = (min(r.p / 1024.0, 2.0), r.pred_s, r.pred_d / 512.0, r.p / cap, 0.0,
                                min((sim.t - r.arrival) / 2.0, 1.0))
                mask[n, kk] = 1.0
        return np.concatenate([reqf, exp.ravel(), slots.ravel(), mask.ravel()])


class RoutingEnv:
    def __init__(self, data, expert_ids, cfgs, split="train", lam=5.0, n_req=300, L=0.030,
                 state="full", reward="qos", workload="poisson", burstgpt_csv=None, run_cap=5, wait_cap=5,
                 baseline_include_req: bool = False, admission_bound: str = "prompt_only"):
        self.data, self.split_idx = data, data.idx(split)
        self.sim = Simulator(data, expert_ids, cfgs, L=L, run_cap=run_cap, wait_cap=wait_cap,
                             admission_bound=admission_bound)
        self.sb = StateBuilder(self.sim, state, baseline_include_req=baseline_include_req)
        self.reward_mode, self.lam, self.n_req = reward, lam, n_req
        self.workload, self.burstgpt_csv = workload, burstgpt_csv
        self.obs_dim, self.n_actions = self.sb.obs_dim, self.sim.N + 1
        self.req = None
        self.obs = None

    def action_mask(self) -> np.ndarray:
        """Returns boolean mask where False means the action is invalid (e.g. expert queue is full)."""
        mask = np.ones(self.n_actions, dtype=bool)
        for n, e in enumerate(self.sim.experts):
            if len(e.waiting) >= self.sim.wait_cap:
                mask[n + 1] = False
        return mask

    def reset(self, seed: int | None = None):
        rng = np.random.default_rng(seed)
        arr = make_arrivals(self.workload, self.lam, self.n_req, rng, self.burstgpt_csv)
        idx = rng.choice(self.split_idx, self.n_req, replace=len(self.split_idx) < self.n_req)
        self.sim.reset(arr, idx)
        self.req = self.sim.advance()
        self.obs = self.sb.build(self.sim, self.req)
        return self.obs

    def step(self, action: int):
        sim, req = self.sim, self.req
        penalty = 0.0
        if self.reward_mode == "qos" and action > 0:
            n = action - 1
            d_true = float(sim.data.dlen[req.idx, sim.ids[n]])
            penalty = sim.impact_penalty(n, req.p, d_true, use_pred=False)
        sim.apply(action)
        nxt = sim.advance()
        reward = sum(sim.qos_of(c) for c in sim.completed) - penalty
        if nxt is None:
            self.req, self.obs = None, np.zeros(self.obs_dim, np.float32)
            return self.obs, float(reward), True, {}
        self.req = nxt
        self.obs = self.sb.build(sim, nxt)
        return self.obs, float(reward), False, {}
