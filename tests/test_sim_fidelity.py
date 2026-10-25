"""Fidelity and correctness unit tests for Simulator (Phase 1a/1b/1c).

Tests:
1. Prefill blocking delay
2. Batch context scaling (decode cost grows with batch context)
3. Strict queue capacity enforcement (R <= 5, W <= 5)
4. Memory-bounded admission (peak bound prevents KV overflow)
5. Drop semantics (action 0 gives zero QoS, drop rate tracking)
6. Simultaneous event tie-breaking (FIFO monotonicity)
7. Determinism (identical seeds produce identical metrics)
8. Action impact estimator formula validation
"""
import tempfile
import numpy as np
import pytest

from qos.data import Data, make_synthetic
from qos.sim import ExpertCfg, Simulator, Req


@pytest.fixture(scope="module")
def synthetic_data():
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        make_synthetic(f.name, n=150, m=4, seed=42)
        return Data(f.name)


def test_prefill_blocking(synthetic_data):
    """When a new request is admitted to a running batch, its prefill phase blocks the iteration."""
    d = synthetic_data
    cfg = ExpertCfg(b=0.02, k1=1e-3, k2=0.0, cap_tokens=10000)
    sim = Simulator(d, [0], [cfg], L=1.0)
    # req 0 arrives at 0.0, req 1 arrives at 0.005 (while req 0 is running)
    sim.reset(np.array([0.0, 0.005]), np.array([0, 1]))
    
    # Process req 0
    r0 = sim.advance()
    sim.apply(1)
    
    # Process req 1
    r1 = sim.advance()
    sim.apply(1)
    
    # Finish all
    while sim.advance() is not None:
        pass
    
    # Analytical finish time calculation:
    # Iteration 1 (only req 0): duration = b + k1 * p0
    # Iteration 2 (req 1 admitted): duration = b + k1 * p1
    # Subsequent iterations: duration = b
    p0 = int(d.p[0])
    d0 = int(d.dlen[0, 0])
    p1 = int(d.p[1])
    d1 = int(d.dlen[1, 0])
    expected_r0 = (cfg.b + cfg.k1 * p0) + (cfg.b + cfg.k1 * p1) + (d0 - 2) * cfg.b
    expected_r1 = (cfg.b + cfg.k1 * p0) + (cfg.b + cfg.k1 * p1) + (d1 - 1) * cfg.b
    assert abs(r0.finish - expected_r0) < 1e-9
    assert abs(r1.finish - expected_r1) < 1e-9


def test_batch_context_scaling(synthetic_data):
    """Decode cost grows linearly with batch context according to k2 * S."""
    d = synthetic_data
    # Isolate k2
    b, k2 = 0.010, 1e-4
    cfg = ExpertCfg(b=b, k1=0.0, k2=k2, cap_tokens=100000)
    sim = Simulator(d, [0], [cfg], L=1.0)
    sim.reset(np.array([0.0]), np.array([0]))
    
    r = sim.advance()
    sim.apply(1)
    assert sim.advance() is None
    
    p = int(d.p[0])
    dlen = int(d.dlen[0, 0])
    # Expected total time: sum_{step=1..dlen} (b + k2 * (p + step - 1))
    expected = sum(b + k2 * (p + step - 1) for step in range(1, dlen + 1))
    assert abs(r.finish - expected) < 1e-9


def test_queue_caps_strict(synthetic_data):
    """Queue bounds (R <= run_cap, W <= wait_cap) are never violated under heavy arrival bursts."""
    d = synthetic_data
    cfg = ExpertCfg(b=0.05, k1=1e-3, k2=1e-5, cap_tokens=10000)
    run_cap, wait_cap = 5, 5
    sim = Simulator(d, [0], [cfg], L=0.03, run_cap=run_cap, wait_cap=wait_cap)
    
    # Flood with 50 simultaneous arrivals at t=0
    n_flood = 50
    sim.reset(np.zeros(n_flood), np.zeros(n_flood, dtype=int))
    
    r = sim.advance()
    while r is not None:
        sim.apply(1)  # keep routing to expert 0
        e = sim.experts[0]
        assert len(e.running) <= run_cap, f"Running queue overflow: {len(e.running)} > {run_cap}"
        assert len(e.waiting) <= wait_cap, f"Waiting queue overflow: {len(e.waiting)} > {wait_cap}"
        r = sim.advance()
    
    assert len(sim.reqs) == n_flood
    # First (run_cap + wait_cap) accepted, rest dropped due to queue overflow
    accepted = [q for q in sim.reqs if not q.dropped]
    assert len(accepted) <= run_cap + wait_cap


def test_memory_admission_peak_bound_prevents_overflow(synthetic_data):
    """Under admission_bound='peak', the KV cache never exceeds cap_tokens."""
    d = synthetic_data
    # Very constrained memory capacity
    cap_tokens = 500
    cfg = ExpertCfg(b=0.02, k1=1e-4, k2=1e-6, cap_tokens=cap_tokens)
    sim = Simulator(d, [0], [cfg], L=0.5, run_cap=5, wait_cap=5, admission_bound="peak")
    
    rng = np.random.default_rng(123)
    arrivals = np.cumsum(rng.exponential(0.01, 30))
    indices = rng.integers(0, len(d.p), 30)
    sim.reset(arrivals, indices)
    
    r = sim.advance()
    while r is not None:
        sim.apply(1)
        # Verify running KV cache never exceeds capacity
        e = sim.experts[0]
        assert e.ctx_tokens() <= cap_tokens, f"KV cache overflow: {e.ctx_tokens()} > {cap_tokens}"
        r = sim.advance()
    
    assert sim.kv_overflow_count == 0


def test_drop_semantics_zero_qos(synthetic_data):
    """Action 0 and overflow drops yield exactly 0 QoS and increment drop_rate."""
    d = synthetic_data
    sim = Simulator(d, [0], [ExpertCfg()], L=0.03)
    sim.reset(np.array([0.0, 0.1]), np.array([0, 1]))
    
    # Drop first request
    r0 = sim.advance()
    sim.apply(0)
    assert r0.dropped
    assert sim.qos_of(r0) == 0.0
    
    # Route second request
    r1 = sim.advance()
    sim.apply(1)
    
    while sim.advance() is not None:
        pass
    
    m = sim.metrics()
    assert m["drop_rate"] == 0.5


def test_tie_breaking_simultaneous_events(synthetic_data):
    """Simultaneous iteration completion events are processed deterministically via FIFO counter."""
    d = synthetic_data
    # Two identical experts with identical configuration and same model id [0, 0]
    cfg = ExpertCfg(b=0.02, k1=0.0, k2=0.0, cap_tokens=10000)
    sim = Simulator(d, [0, 0], [cfg, cfg], L=1.0)
    # Two simultaneous arrivals at t=0
    sim.reset(np.array([0.0, 0.0]), np.array([0, 0]))
    
    r0 = sim.advance()
    sim.apply(1)  # expert 0
    r1 = sim.advance()
    sim.apply(2)  # expert 1
    
    # Drain simulation to completion
    while sim.advance() is not None:
        pass

    # Both requests run on identical experts with k1=0, k2=0, so total duration is d_total * b
    dlen = int(d.dlen[0, 0])
    expected_finish = dlen * cfg.b
    assert abs(r0.finish - expected_finish) < 1e-9
    assert abs(r1.finish - expected_finish) < 1e-9
    assert abs(r0.finish - r1.finish) < 1e-9


def test_determinism_identical_metrics(synthetic_data):
    """Two independent simulation runs with the same arrivals and routing yield identical metrics."""
    d = synthetic_data
    cfg = ExpertCfg()
    
    def run_sim(seed):
        rng = np.random.default_rng(seed)
        arr = np.cumsum(rng.exponential(0.1, 50))
        idx = rng.integers(0, len(d.p), 50)
        sim = Simulator(d, [0, 1], [cfg, cfg], L=0.03)
        sim.reset(arr, idx)
        r = sim.advance()
        action = 1
        while r is not None:
            sim.apply(action)
            action = 2 if action == 1 else 1
            r = sim.advance()
        return sim.metrics()
    
    m1 = run_sim(42)
    m2 = run_sim(42)
    for k in m1:
        assert m1[k] == pytest.approx(m2[k], abs=1e-12), f"Non-deterministic metric {k}: {m1[k]} != {m2[k]}"


def test_action_impact_estimator_penalty(synthetic_data):
    """Action impact estimator penalty matches theoretical definition."""
    d = synthetic_data
    # L = 0.030 s/tok. If running request's latency jumps past 0.030, penalty must equal its score.
    cfg = ExpertCfg(b=0.020, k1=1e-4, k2=1e-6, cap_tokens=10000)
    sim = Simulator(d, [0], [cfg], L=0.030)
    sim.reset(np.array([0.0]), np.array([0]))
    
    r0 = sim.advance()
    sim.apply(1)
    
    # Case 1: Small request does not violate latency limit -> penalty exactly 0.0
    pen_no_viol = sim.impact_penalty(0, 10, 10, use_pred=False)
    assert pen_no_viol == 0.0

    # Case 2: Large prefill causes violation of L=0.030 -> penalty exactly r0.score
    pen_viol = sim.impact_penalty(0, 60000, 10, use_pred=False)
    assert abs(pen_viol - r0.score) < 1e-9
