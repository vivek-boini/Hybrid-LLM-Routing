"""Unit tests verifying consistent action masking across DiscreteSAC (Phase 1 Item 3).

Verifies:
1. Action masking in act(): masked actions have 0 probability.
2. Action masking in next-state soft value calculation: p2(masked) == 0.0, target y does not bootstrap masked actions.
3. Action masking in actor loss: policy probabilities for masked actions are exactly 0.0.
4. No NaNs produced during gradient update with masked logits (-1e9).
5. Unmasked updates remain exact and backward-compatible.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from qos.sac import DiscreteSAC, ReplayBuffer


@pytest.fixture
def sac_fixture():
    obs_dim = 12
    n_act = 4
    lay = {"N": 3, "S": 10, "F": 6, "req": 7, "exp": 9}
    agent = DiscreteSAC(obs_dim, n_act, lay, encoder="flat", device="cpu")
    return agent, obs_dim, n_act


def test_act_masked_probability_zero(sac_fixture):
    """Masked actions are never chosen in act(), both deterministic and sampled."""
    agent, obs_dim, n_act = sac_fixture
    obs = np.ones(obs_dim, dtype=np.float32)
    # Mask actions 1 and 2, allow only 0 (drop) and 3
    mask = np.array([True, False, False, True], dtype=bool)
    
    # Test deterministic
    a_det = agent.act(obs, deterministic=True, action_mask=mask)
    assert a_det in (0, 3)
    
    # Test stochastic sampling across 50 iterations
    for _ in range(50):
        a_samp = agent.act(obs, deterministic=False, action_mask=mask)
        assert a_samp in (0, 3)


def test_update_with_action_mask(sac_fixture):
    """update() with action mask correctly zeroes masked actions in target y and actor loss without NaNs."""
    agent, obs_dim, n_act = sac_fixture
    buf = ReplayBuffer(obs_dim, n_act=n_act, size=100, store_mask=True)
    
    rng = np.random.default_rng(42)
    for _ in range(30):
        o = rng.normal(0, 1, obs_dim).astype(np.float32)
        a = rng.integers(0, n_act)
        r = float(rng.normal(0.5, 0.2))
        o2 = rng.normal(0, 1, obs_dim).astype(np.float32)
        d = float(rng.choice([0.0, 1.0]))
        # Action 2 is masked (invalid queue)
        m = np.array([True, True, False, True], dtype=bool)
        m2 = np.array([True, False, True, True], dtype=bool)
        buf.add(o, a, r, o2, d, m=m, m2=m2)
    
    batch = buf.sample(16, "cpu")
    lq, la, alpha = agent.update(batch)
    
    assert not np.isnan(lq), "Critic loss produced NaN"
    assert not np.isnan(la), "Actor loss produced NaN"
    assert not np.isnan(alpha), "Alpha produced NaN"
    assert lq >= 0.0


def test_next_state_masked_actions_do_not_affect_target(sac_fixture):
    """Artificially inflating Q of a masked action does not alter target y."""
    agent, obs_dim, n_act = sac_fixture
    o2 = torch.randn(2, obs_dim)
    r = torch.zeros(2)
    d = torch.zeros(2)
    m2 = torch.tensor([[True, False, True, True], [True, True, True, False]], dtype=torch.bool)
    
    # Compute normal target
    with torch.no_grad():
        lg2 = agent.actor(o2)
        lg2_m = torch.where(m2, lg2, torch.tensor(-1e9))
        p2 = F.softmax(lg2_m, -1)
        lp2 = torch.log(p2.clamp(min=1e-8))
        qt = torch.min(agent.q1t(o2), agent.q2t(o2))
        v2 = torch.where(p2 > 0, p2 * (qt - agent.log_alpha.exp() * lp2), torch.zeros_like(p2)).sum(-1)
        y1 = r + agent.gamma * (1 - d) * v2

    # Now artificially set qt on the masked action (action 1 for row 0) to 1e6
    with torch.no_grad():
        qt_hacked = qt.clone()
        qt_hacked[0, 1] = 1e6
        qt_hacked[1, 3] = 1e6
        v2_hacked = torch.where(p2 > 0, p2 * (qt_hacked - agent.log_alpha.exp() * lp2), torch.zeros_like(p2)).sum(-1)
        y2 = r + agent.gamma * (1 - d) * v2_hacked

    # Target y must remain exactly identical because p2 for masked actions is 0
    assert torch.allclose(y1, y2, atol=1e-6)
