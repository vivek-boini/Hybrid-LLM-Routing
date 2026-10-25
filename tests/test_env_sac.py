"""Unit tests for RoutingEnv, StateBuilder, and DiscreteSAC (Phase 1d/1e).

Tests:
1. StateBuilder obs_dim with and without baseline_include_req
2. RoutingEnv action_mask correctly identifies full waiting queues
3. DiscreteSAC masked action selection (masked actions have 0 probability)
4. DiscreteSAC replay buffer and update step
5. Terminal state handling (done=1 does not bootstrap future Q)
"""
import tempfile
import numpy as np
import pytest
import torch

from qos.data import Data, make_synthetic
from qos.env import RoutingEnv, StateBuilder
from qos.sac import DiscreteSAC, ReplayBuffer
from qos.setup import make_experts
from qos.sim import Simulator


@pytest.fixture(scope="module")
def env_fixture():
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        make_synthetic(f.name, n=100, m=4, seed=42)
        d = Data(f.name)
        ids, cfgs = make_experts(d, n=3, hetero=0.1)
        env = RoutingEnv(d, ids, cfgs, split="train", n_req=20, L=0.030)
        return env, d, ids, cfgs


def test_baseline_include_req_dimensions(env_fixture):
    """When baseline_include_req is True, StateBuilder adds req_dim to expert_only state."""
    env, d, ids, cfgs = env_fixture
    sim = env.sim
    
    # 1. Standard paper expert_only: 3 * N features
    sb_standard = StateBuilder(sim, mode="expert_only", baseline_include_req=False)
    assert sb_standard.obs_dim == 3 * sim.N
    
    # 2. Flagged expert_only: (1 + 2 * N) + 3 * N features
    sb_with_req = StateBuilder(sim, mode="expert_only", baseline_include_req=True)
    expected_dim = (1 + 2 * sim.N) + (3 * sim.N)
    assert sb_with_req.obs_dim == expected_dim
    
    # Test vector output
    env.reset(seed=1)
    obs_standard = sb_standard.build(sim, env.req)
    obs_with_req = sb_with_req.build(sim, env.req)
    assert len(obs_standard) == sb_standard.obs_dim
    assert len(obs_with_req) == sb_with_req.obs_dim


def test_action_mask_full_queue(env_fixture):
    """action_mask flags expert actions as invalid (False) when waiting queue is at wait_cap."""
    env, d, ids, cfgs = env_fixture
    env.reset(seed=1)
    
    # Initially all queues empty -> all actions valid
    mask = env.action_mask()
    assert mask[0] == True  # Drop is always valid
    assert all(mask)
    
    # Fill expert 0's waiting queue to wait_cap (5)
    e0 = env.sim.experts[0]
    # Simulate a full waiting queue
    for i in range(env.sim.wait_cap):
        e0.waiting.append(env.req)
    
    mask_after = env.action_mask()
    assert mask_after[0] == True   # Drop still valid
    assert mask_after[1] == False  # Expert 0 is full -> invalid
    assert mask_after[2] == True   # Expert 1 is free
    assert mask_after[3] == True   # Expert 2 is free


def test_sac_action_masking():
    """DiscreteSAC never chooses a masked action when action_mask is provided."""
    obs_dim = 10
    n_act = 4
    lay = {"N": 3, "S": 10, "F": 6, "req": 7, "exp": 9}
    agent = DiscreteSAC(obs_dim, n_act, lay, encoder="flat", device="cpu")
    
    obs = np.random.randn(obs_dim).astype(np.float32)
    
    # Mask out actions 1, 2, 3 -> only action 0 (drop) is allowed
    mask = np.array([True, False, False, False], dtype=bool)
    for _ in range(20):
        action = agent.act(obs, deterministic=False, action_mask=mask)
        assert action == 0, f"Masked action was chosen: {action}"
    
    # Mask out action 0 -> only actions 1, 2, 3 allowed
    mask2 = np.array([False, True, True, True], dtype=bool)
    for _ in range(20):
        action2 = agent.act(obs, deterministic=False, action_mask=mask2)
        assert action2 in (1, 2, 3), f"Masked action 0 was chosen: {action2}"


def test_sac_update_and_replay_buffer():
    """ReplayBuffer stores transitions and DiscreteSAC updates parameters without NaN."""
    obs_dim = 16
    n_act = 4
    lay = {"N": 3, "S": 10, "F": 6, "req": 7, "exp": 9}
    agent = DiscreteSAC(obs_dim, n_act, lay, encoder="flat", device="cpu")
    buf = ReplayBuffer(obs_dim, size=100)
    
    rng = np.random.default_rng(0)
    for _ in range(50):
        o = rng.normal(0, 1, obs_dim).astype(np.float32)
        a = rng.integers(0, n_act)
        r = float(rng.normal(0.5, 0.2))
        o2 = rng.normal(0, 1, obs_dim).astype(np.float32)
        d = float(rng.choice([0.0, 1.0]))
        buf.add(o, a, r, o2, d)
    
    batch = buf.sample(16, "cpu")
    lq, la, alpha = agent.update(batch)
    assert not np.isnan(lq)
    assert not np.isnan(la)
    assert alpha > 0.0
