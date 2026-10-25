# Changelog

All notable changes to the `qos_routing_repro` project are documented in this file.

## [Phase 1: Correctness and Fidelity] - 2026-10-24

### Added
- **Action Masking (`qos/env.py`, `qos/sac.py`, `qos/policies.py`):** Added `action_mask()` in `RoutingEnv`, next-state soft value zeroing and actor loss masking in `DiscreteSAC.update()`, action logit masking in `DiscreteSAC.act()`, and `--use-mask` CLI flags for `train` and `eval` (default unmasked).
- **Peak-Bounded Memory Admission (`qos/sim.py`, `qos/cli.py`):** Added opt-in `--admission-bound peak` using true output length at admission to guarantee no mid-flight KV-cache overflow. Default remains `prompt_only` to guarantee exact paper baseline reproduction.
- **KV Overflow Monitoring (`qos/sim.py`):** Simulator monitors and records `kv_overflows` in metric outputs.
- **Baseline-RL Feature Extension Flag (`qos/env.py`, `qos/cli.py`):** Added `--baseline-include-req` flag to evaluate Baseline-RL with arriving request features. Default remains paper's expert-only state.
- **Mix-Instruct Tokenizer Option (`qos/data.py`):** Added `count_tokens` supporting exact BPE via `tiktoken` in addition to the `approx` heuristic (`words * 1.33`).
- **Mix-Instruct Schema Validation (`qos/data.py`):** Added `validate_mixinstruct_record` to verify JSONL structure.
- **Robust BurstGPT Timestamp Parsing (`qos/workload.py`):** Added support for both numeric seconds and ISO string datetime headers.
- **Unit Test Suites (21 tests in `tests/`):**
  - `tests/test_sim_fidelity.py`: 8 tests covering prefill blocking, batch context scaling, queue bounds, memory bounds, drop semantics, tie-breaking, determinism, and impact estimator penalty.
  - `tests/test_sac_masking.py`: 3 tests covering masking in `act()`, zero-probability next-state target calculation, and actor loss updates without NaNs.
  - `tests/test_env_sac.py`: 4 tests for observation dimensions, action masking, and SAC updates.
  - `tests/test_data_schema.py`: 3 tests for schema validation, tokenizers, and BurstGPT parsing.
  - `tests/test_sim.py`: 3 original simulation tests.
- **Documentation:** Created `DATA_CARD.md`, `ASSUMPTIONS.md`, `DEVIATIONS.md`, and `VERIFICATION.md`.

### Changed
- Preserved `prompt_only` as the default admission bound across `Simulator`, `RoutingEnv`, and `cli.py` to ensure 100% backward compatibility with Step 0 baselines.
- Updated `requirements.txt` to include `pytest` and optional `tiktoken`.

### Fixed
- Fixed tie-breaking calculation in `tests/test_sim_fidelity.py::test_tie_breaking_simultaneous_events` which failed by expecting single-iteration time instead of full output generation duration.
- Restored strictly analytical assertions in `tests/test_sim_fidelity.py`:
  - `test_prefill_blocking`: replaced loose inequalities with exact closed-form arrival-iteration finish times.
  - `test_action_impact_estimator_penalty`: added strict zero-penalty and violation-penalty checks against theoretical score values.

## [Phase 3: Rigorous Evaluation & Baselines] - 2026-10-24

### Added
- **Bootstrap Confidence Intervals & Paired Significance (`qos/evaluate.py`):** Added 1,000-resample non-parametric bootstrap 95% CIs and paired Wilcoxon signed-rank / paired t-test against reference policy.
- **Detailed Latency Percentiles (`qos/sim.py`):** Added `lat_p50_ms`, `lat_p90_ms`, `lat_p95_ms`, `lat_p99_ms` tracking.
- **Categorized Drop Causes (`qos/sim.py`):** Added explicit tracking for `drop_voluntary` (router policy action 0), `drop_queue_full` (waiting queue overflow), and `drop_kv_full` (memory exhaustion).
- **Per-Expert Utilization (`qos/sim.py`):** Added `expert_{i}_util` tracking across experts.
- **Greedy Baseline Ablations (`qos/policies.py`):** Extended `GreedyImpact` with `--policies greedy,greedy_score_only,greedy_no_penalty` to isolate impact penalty from latency filtering.
- **YAML Config Runner (`qos/cli.py`, `configs/`):** Added `--config` CLI support and structured experiment configs (`configs/real_heuristics.yaml`, `configs/real_greedy_ablations.yaml`, `configs/synthetic_heuristics.yaml`).
- **Reproducibility Metadata Headers:** All evaluation CSVs record git hash, data SHA256, tokenizer, admission bound, mask setting, seeds, package versions, and dataset label.
