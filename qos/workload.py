"""Arrival processes: Poisson, bursty (2-state MMPP), and BurstGPT-derived."""
from __future__ import annotations

import numpy as np


def poisson_arrivals(lam: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.cumsum(rng.exponential(1.0 / lam, n))


def bursty_arrivals(lam: float, n: int, rng: np.random.Generator,
                    burst_factor: float = 4.0, mean_dwell: float = 5.0) -> np.ndarray:
    """Two-state Markov-modulated Poisson process with mean rate ~= lam (synthetic burst stand-in)."""
    lo = lam * 2.0 / (1.0 + burst_factor)
    hi = lo * burst_factor
    t, out, high = 0.0, [], False
    state_end = rng.exponential(mean_dwell)
    while len(out) < n:
        t += rng.exponential(1.0 / (hi if high else lo))
        if t > state_end:
            high = not high
            state_end = t + rng.exponential(mean_dwell)
        out.append(t)
    return np.array(out)


def burstgpt_arrivals(csv_path: str, lam: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Take a random window of n requests from a BurstGPT CSV and rescale to mean rate lam.

    Expects a timestamp column (any column whose name contains 'time' or 'timestamp').
    Handles both numeric seconds/timestamps and ISO datetime strings.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    matching_cols = [c for c in df.columns if "time" in c.lower()]
    if not matching_cols:
        raise ValueError(f"No timestamp column found in {csv_path}. Columns: {list(df.columns)}")
    col = matching_cols[0]
    if pd.api.types.is_numeric_dtype(df[col]):
        ts = np.sort(df[col].to_numpy(dtype=float))
    else:
        # Parse datetime strings into epoch seconds
        ts = np.sort(pd.to_datetime(df[col]).astype("int64").to_numpy(dtype=float) / 1e9)
    start = rng.integers(0, max(1, len(ts) - n))
    w = ts[start:start + n]
    w = w - w[0]
    if len(w) < 2 or w[-1] <= 0:
        return poisson_arrivals(lam, n, rng)
    return w * ((n - 1) / lam) / w[-1] + 1e-6  # rescale so the mean rate equals lam


def make_arrivals(kind: str, lam: float, n: int, rng: np.random.Generator, burstgpt_csv: str | None = None):
    if kind == "poisson":
        return poisson_arrivals(lam, n, rng)
    if kind == "bursty":
        return bursty_arrivals(lam, n, rng)
    if kind == "burstgpt":
        if not burstgpt_csv:
            raise ValueError("--burstgpt-csv is required for workload=burstgpt")
        return burstgpt_arrivals(burstgpt_csv, lam, n, rng)
    raise ValueError(kind)
