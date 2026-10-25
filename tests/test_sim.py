"""Sanity tests for the simulator. Run from the repo root:  python tests/test_sim.py  (or pytest -q tests)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qos.data import Data, make_synthetic  # noqa: E402
from qos.sim import ExpertCfg, Simulator  # noqa: E402


def _data():
    f = os.path.join(tempfile.mkdtemp(), "d.npz")
    make_synthetic(f, n=200, m=3)
    return Data(f)


def test_single_request_latency_matches_formula():
    d = _data()
    cfg = ExpertCfg(b=0.02, k1=1e-4, k2=1e-6, cap_tokens=10**6)
    sim = Simulator(d, [0], [cfg], L=1.0)
    sim.reset(np.array([0.0]), np.array([0]))
    r = sim.advance()
    sim.apply(1)
    assert sim.advance() is None
    p, n = int(d.p[0]), int(d.dlen[0, 0])
    # iteration k: T = b + k2*(p + k - 1); the first iteration also pays the prefill k1*p
    expect = sum(cfg.b + cfg.k2 * (p + k - 1) for k in range(1, n + 1)) + cfg.k1 * p
    assert abs(r.finish - expect) < 1e-9, (r.finish, expect)


def test_conservation_and_queue_caps():
    d = _data()
    cfg = ExpertCfg()
    sim = Simulator(d, [0, 1], [cfg, cfg], L=0.03)
    rng = np.random.default_rng(0)
    sim.reset(np.cumsum(rng.exponential(0.05, 200)), rng.integers(0, 200, 200))
    r = sim.advance()
    while r is not None:
        sim.apply(int(rng.integers(0, 3)))
        for e in sim.experts:
            assert len(e.running) <= sim.run_cap and len(e.waiting) <= sim.wait_cap
        r = sim.advance()
    assert len(sim.reqs) == 200
    assert all(q.dropped or q.finish >= q.arrival for q in sim.reqs)
    m = sim.metrics()
    assert 0.0 <= m["qos"] <= 1.0 and all(np.isfinite(v) for v in m.values())


def test_drop_gives_zero_qos():
    d = _data()
    sim = Simulator(d, [0], [ExpertCfg()], L=0.03)
    sim.reset(np.array([0.0]), np.array([0]))
    r = sim.advance()
    sim.apply(0)
    sim.advance()
    assert sim.qos_of(r) == 0.0 and r.dropped


if __name__ == "__main__":
    test_single_request_latency_matches_formula()
    test_conservation_and_queue_caps()
    test_drop_gives_zero_qos()
    print("all tests passed")
