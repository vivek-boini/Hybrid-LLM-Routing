"""Routing policies. Every policy implements act(env) -> action (0 = drop, n = expert n-1)."""
from __future__ import annotations

import numpy as np


class RoundRobin:
    name = "RR"

    def __init__(self):
        self.i = 0

    def act(self, env):
        a = self.i % env.sim.N + 1
        self.i += 1
        return a


class ShortestQueue:
    """SQF: send to the expert with the fewest running+waiting requests."""
    name = "SQF"

    def act(self, env):
        q = [len(e.running) + len(e.waiting) for e in env.sim.experts]
        return int(np.argmin(q)) + 1


class BertRouter:
    """BR: pick the expert with the highest PREDICTED generation score (load-blind)."""
    name = "BR"

    def act(self, env):
        d, r = env.sim.data, env.req
        return int(np.argmax(d.act_s[r.idx, env.sim.ids])) + 1


class GreedyImpact:
    """Our add-on baseline: a one-step greedy policy built from the paper's OWN action-impact estimator.

    value(n) = predicted_score(n) * 1[estimated own latency <= L]  -  estimated QoS put at risk on n
    Drops the request if every option has negative value. No learning beyond the score/length predictors.
    """
    name = "Greedy"

    def __init__(self, mode: str = "full"):
        self.mode = mode
        if mode == "score_only":
            self.name = "Greedy-Score"
        elif mode == "no_penalty":
            self.name = "Greedy-NoPen"
        else:
            self.name = "Greedy"

    def act(self, env):
        sim, d, r = env.sim, env.sim.data, env.req
        best_a, best_v = 0, 0.0
        for n in range(sim.N):
            e = sim.experts[n]
            if len(e.waiting) >= sim.wait_cap:
                continue
            j = sim.ids[n]
            s_hat, d_hat = float(d.act_s[r.idx, j]), max(1.0, float(d.act_d[r.idx, j]))
            if self.mode == "score_only":
                v = s_hat
            elif self.mode == "no_penalty":
                gain = s_hat if sim.est_own_latency(n, r.p, d_hat) <= sim.L else 0.0
                v = gain + 1e-3 * s_hat
            else:
                gain = s_hat if sim.est_own_latency(n, r.p, d_hat) <= sim.L else 0.0
                v = gain - sim.impact_penalty(n, r.p, d_hat, use_pred=True) + 1e-3 * s_hat
            if v > best_v:
                best_a, best_v = n + 1, v
        return best_a


class RLPolicy:
    def __init__(self, agent, name="RL", use_mask: bool = False):
        self.agent, self.name, self.use_mask = agent, name, use_mask

    def act(self, env):
        mask = env.action_mask() if self.use_mask and hasattr(env, "action_mask") else None
        return self.agent.act(env.obs, deterministic=True, action_mask=mask)
