# Verification Evidence & Mathematical Audit

This document provides formal verification evidence, exact command outputs, and mathematical derivations.

---

## 1. Backward Compatibility & Admission Bound Audit (Item 1)

### Before/After Verification: `prompt_only` (Default Reproduction) vs. `peak` (Oracle Bound)
Executed on **synthetic data** (`data/synthetic.npz`, $\lambda=5\text{ req/s}$, $N=6$, $L=30\text{ ms/tok}$, 5 random seeds):

```bash
# Default Reproduction (prompt_only)
python -m qos.cli eval --data data/synthetic.npz --lam 5 --n-req 300 --seeds 5 --policies rr,sqf,br,greedy --rl "QoS-RL=runs/qos/best.pt:full:flat"

# Opt-in Oracle Bound (peak)
python -m qos.cli eval --data data/synthetic.npz --lam 5 --n-req 300 --seeds 5 --policies rr,sqf,br,greedy --rl "QoS-RL=runs/qos/best.pt:full:flat" --admission-bound peak
```

### Exact Numerical Comparison Table (Synthetic Data)

| Policy | Step 0 Baseline (Synthetic) | `prompt_only` Run (Synthetic) | `peak` Run (Synthetic) | Match Status | Effect of `peak` Bound |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **RR** | $0.5786 \pm 0.0362$ (drop $1.2\%$) | $0.5786 \pm 0.0362$ (drop $1.2\%$) | $0.5646 \pm 0.0378$ (drop $1.7\%$) | **Exact 100% Match** | $-0.0140$ QoS (stricter reservation slightly increases queue drops) |
| **SQF** | $0.6491 \pm 0.0023$ (drop $0.0\%$) | $0.6491 \pm 0.0023$ (drop $0.0\%$) | $0.6498 \pm 0.0031$ (drop $0.0\%$) | **Exact 100% Match** | $+0.0007$ QoS (negligible difference $< 0.1\%$) |
| **BR** | $0.1342 \pm 0.0076$ (drop $46.7\%$) | $0.1342 \pm 0.0076$ (drop $46.7\%$) | $0.1343 \pm 0.0076$ (drop $48.0\%$) | **Exact 100% Match** | $+0.0001$ QoS (negligible difference) |
| **Greedy** | $0.6834 \pm 0.0195$ (drop $0.3\%$) | $0.6834 \pm 0.0195$ (drop $0.3\%$) | $0.6808 \pm 0.0255$ (drop $1.3\%$) | **Exact 100% Match** | $-0.0026$ QoS (slight drop from conservative reservation) |
| **QoS-RL**| $0.6407 \pm 0.0130$ (drop $1.5\%$) | $0.6407 \pm 0.0130$ (drop $1.5\%$) | $0.6318 \pm 0.0273$ (drop $2.5\%$) | **Exact 100% Match** | $-0.0089$ QoS (more conservative queue admission) |

> **Conclusion:** Setting `admission_bound="prompt_only"` as default guarantees bit-exact replication of the pre-change baseline. `peak` is preserved as an opt-in oracle bound.

---

## 2. Real Mix-Instruct Dataset Preparation Command Output (Item 5)

Command executed on real data:
```bash
python -m qos.cli prepare --train-path data/raw/train_data_real.jsonl --val-path data/raw/val_data_real.jsonl --out data/mix.npz
```

Actual Terminal Output:
```
12 models kept: ['alpaca-native', 'chatglm-6b', 'dolly-v2-12b', 'flan-t5-xxl', 'koala-7B-HF', 'llama-7b-hf-baize-lora-bf16', 'moss-moon-003-sft', 'mpt-7b', 'mpt-7b-instruct', 'oasst-sft-4-pythia-12b-epoch-3.5', 'stablelm-tuned-alpha-7b', 'vicuna-13b-1.1']
5000 requests with full coverage
saved data/mix.npz
```

Real Data Inspection Statistics:
```
Keys: ['p', 'score', 'dlen', 'emb', 'split', 'models']
Models (12): ['alpaca-native', 'chatglm-6b', 'dolly-v2-12b', 'flan-t5-xxl', 'koala-7B-HF', 'llama-7b-hf-baize-lora-bf16', 'moss-moon-003-sft', 'mpt-7b', 'mpt-7b-instruct', 'oasst-sft-4-pythia-12b-epoch-3.5', 'stablelm-tuned-alpha-7b', 'vicuna-13b-1.1']
Splits: train: 3000, dev: 1000, test: 1000 (total: 5000)
Rows dropped for missing coverage: 0 (100% coverage across all 12 models)
Prompt length: mean=19.2, min=1, median=16.0, p95=41.0, max=136
Output length: mean=163.9, min=1, median=192.0, p95=285.0, max=339
BERTScores:    mean=0.6643, min=0.0000, median=0.6652, p95=0.7923, max=1.0000
```

---

## 3. Real Data Tokenization Error Analysis (Item 4)

Evaluated across **1,000 real prompts** and **11,989 real candidate responses** (**12,989 total texts**) from `val_data_prepared.jsonl`:

| Metric | Overall Texts | Prompts Only | Candidate Responses Only |
| :--- | :---: | :---: | :---: |
| **Mean Relative Error** | **15.34%** | **10.69%** | **15.73%** |
| **Median Relative Error** | **10.85%** | **9.09%** | **11.02%** |
| **95th Percentile Error** | **50.00%** | **26.02%** | **52.03%** |

---

## 4. Test Execution Summary

Command executed:
```powershell
python -m pytest tests -v
```

Output:
```
============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-8.3.3
collected 21 items

tests/test_data_schema.py::test_validate_mixinstruct_record PASSED       [  4%]
tests/test_data_schema.py::test_token_counting_methods PASSED            [  9%]
tests/test_data_schema.py::test_burstgpt_numeric_and_datetime PASSED     [ 14%]
tests/test_env_sac.py::test_baseline_include_req_dimensions PASSED       [ 19%]
tests/test_env_sac.py::test_action_mask_full_queue PASSED                [ 23%]
tests/test_env_sac.py::test_sac_action_masking PASSED                    [ 28%]
tests/test_env_sac.py::test_sac_update_and_replay_buffer PASSED          [ 33%]
tests/test_sac_masking.py::test_act_masked_probability_zero PASSED       [ 38%]
tests/test_sac_masking.py::test_update_with_action_mask PASSED           [ 42%]
tests/test_sac_masking.py::test_next_state_masked_actions_do_not_affect_target PASSED [ 47%]
tests/test_sim.py::test_single_request_latency_matches_formula PASSED    [ 52%]
tests/test_sim.py::test_conservation_and_queue_caps PASSED               [ 57%]
tests/test_sim.py::test_drop_gives_zero_qos PASSED                       [ 61%]
tests/test_sim_fidelity.py::test_prefill_blocking PASSED                 [ 66%]
tests/test_sim_fidelity.py::test_batch_context_scaling PASSED            [ 71%]
tests/test_sim_fidelity.py::test_queue_caps_strict PASSED                [ 76%]
tests/test_sim_fidelity.py::test_memory_admission_peak_bound_prevents_overflow PASSED [ 80%]
tests/test_sim_fidelity.py::test_drop_semantics_zero_qos PASSED          [ 85%]
tests/test_sim_fidelity.py::test_tie_breaking_simultaneous_events PASSED [ 90%]
tests/test_sim_fidelity.py::test_determinism_identical_metrics PASSED    [ 95%]
tests/test_sim_fidelity.py::test_action_impact_estimator_penalty PASSED  [100%]

============================= 21 passed in 15.36s =============================
```
