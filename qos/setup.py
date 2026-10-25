"""Helpers that build the expert pool (which models, what hardware profile)."""
from __future__ import annotations

import json

import numpy as np

from .sim import ExpertCfg


def make_experts(data, n: int, profile: str | None = None, hetero: float = 0.2, seed: int = 123,
                 ids: list[int] | None = None):
    """Pick n experts spread across the quality ranking and give each a hardware profile.

    profile : JSON from `calibrate_vllm` with keys b,k1,k2,cap_tokens (base values). If None, the
              PLACEHOLDER defaults in sim.ExpertCfg are used.
    hetero  : each expert's b/k1/k2 is scaled by a factor in [1-hetero, 1+hetero] (seeded) so experts differ.
    """
    if ids is None:
        order = np.argsort(data.score.mean(0))               # worst ... best model
        ids = sorted(int(order[i]) for i in np.linspace(0, data.M - 1, n).round().astype(int))
    base = ExpertCfg()
    if profile:
        base = ExpertCfg(**{k: v for k, v in json.load(open(profile)).items() if k in ExpertCfg.__annotations__ and k != "name"})
    rng = np.random.default_rng(seed)
    cfgs = []
    for j in ids:
        m = rng.uniform(1 - hetero, 1 + hetero)
        cfgs.append(ExpertCfg(name=data.models[j], b=base.b * m, k1=base.k1 * m, k2=base.k2 * m,
                              cap_tokens=int(base.cap_tokens * rng.uniform(0.8, 1.2))))
    return ids, cfgs
