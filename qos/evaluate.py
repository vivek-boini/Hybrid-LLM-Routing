"""Run policies on identical workloads (same seeds) and aggregate metrics."""
from __future__ import annotations

import numpy as np
from scipy import stats


def run_episode(env, policy, seed: int) -> dict:
    env.reset(seed)
    while True:
        _, _, done, _ = env.step(policy.act(env))
        if done:
            break
    return env.sim.metrics()


def bootstrap_ci(vals: np.ndarray, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Empirical bootstrap confidence interval for the mean."""
    if len(vals) <= 1:
        v = float(vals[0]) if len(vals) == 1 else 0.0
        return v, v, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boot_means = np.mean(vals[indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    half = float((high - low) / 2.0)
    return low, high, half


def paired_pvalue(vals_a: np.ndarray, vals_b: np.ndarray) -> float:
    """Paired difference test (Wilcoxon signed-rank if n >= 5 and diffs non-zero, else paired t-test)."""
    diff = np.array(vals_a) - np.array(vals_b)
    if np.all(diff == 0):
        return 1.0
    if len(diff) >= 5:
        try:
            res = stats.wilcoxon(diff)
            return float(res.pvalue)
        except Exception:
            pass
    try:
        res = stats.ttest_rel(vals_a, vals_b)
        return float(res.pvalue)
    except Exception:
        return 1.0


def evaluate(env, policy, seeds, return_episodes: bool = False) -> dict | tuple[dict, list[dict]]:
    ms = [run_episode(env, policy, s) for s in seeds]
    out = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    qos_vals = np.array([m["qos"] for m in ms])
    out["qos_std"] = float(np.std(qos_vals))
    out["qos_ci95"] = float(1.96 * np.std(qos_vals, ddof=1) / np.sqrt(len(ms))) if len(ms) > 1 else 0.0
    ci_low, ci_high, ci_half = bootstrap_ci(qos_vals)
    out["qos_boot_ci_low"] = ci_low
    out["qos_boot_ci_high"] = ci_high
    out["qos_boot_ci95"] = ci_half
    if return_episodes:
        return out, ms
    return out
