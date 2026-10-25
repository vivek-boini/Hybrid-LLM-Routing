# QoS-aware LLM routing: reproduction kit

Reproduces the **core** of *Quality-of-Service Aware LLM Routing for Edge Computing With Multiple Experts*
(Yang et al., IEEE TMC 2025; arXiv 2508.00234) as a "core-plus" reproduction for a review paper, plus one add-on:
a **greedy policy built from the paper's own action-impact estimator** (a baseline the paper does not report).

> The numbers printed by the smoke tests in this repo come from **synthetic data**. They only prove the code runs.
> Nothing here is a result. Real results need Mix-Instruct (+ your DGX calibration).

## What is implemented

| Paper component | Here | Faithful? |
|---|---|---|
| Edge experts with running/waiting queues (cap 5/5), prefill blocks the batch, decode cost grows with batch context | `qos/sim.py` (iteration-level discrete-event simulator; unit-tested against the closed-form latency) | Yes, in structure. Constants are placeholders until you calibrate |
| QoS = BERTScore × 1[latency per token ≤ L] | `Simulator.qos_of` | Yes |
| Generation-score and output-length predictors (10 buckets) | `qos/predictors.py`: MLP on frozen embeddings | **Deviation** (paper: fine-tuned DistilBERT) |
| Dynamic state abstraction (heterogeneous graph + HAN) | `qos/env.py` flat padded state, or a DeepSets encoder in `qos/sac.py` | **Deviation** (no graph attention) |
| Action impact estimator + QoS-aware reward | `Simulator.impact_penalty`, `RoutingEnv(reward="qos")` | Yes |
| SAC router (action = drop or expert n) | `qos/sac.py` discrete SAC | Close (details such as target entropy are not specified in the paper) |
| Baselines BR, RR, SQF, Baseline RL | `qos/policies.py`, `--state expert_only --reward baseline` | Close (BR uses the same predictor's score) |
| **Add-on:** greedy on the impact estimator | `policies.GreedyImpact` | New |

## Tested vs not tested (be honest about this in your paper)

**Tested in a CPU-only environment (21 automated unit tests in `pytest tests/`):**
- Simulator fidelity: iteration latency model, prefill blocking, batch context scaling, strict queue bounds ($R \le 5, W \le 5$), memory bounds, tie-breaking, and determinism.
- Reinforcement learning: discrete SAC loss updates, next-state soft value calculation, action masking in actor/critic/evaluation, and replay buffer transitions.
- Real data pipeline: verified against real `val_data_prepared.jsonl` downloaded from HuggingFace Hub (`llm-blender/mix-instruct`), producing `data/mix.npz` (5,000 requests, 12 models kept, 100% coverage).
- Tokenization error: empirical relative error computed over 12,989 real texts from Mix-Instruct (mean $15.34\%$, median $10.85\%$, p95 $50.00\%$).
- BurstGPT workload: verified parsing of both numeric float seconds and ISO datetime string timestamps.
- Heuristics and SAC training/eval across arrival rates ($\lambda$), SLO limits ($L$), and bursty workloads on **synthetic data**.

**Not tested on physical hardware:**
- `calibrate_vllm.py` on a live GPU running the vLLM engine (requires DGX or physical GPU instance).

## New CLI Flags
- `--admission-bound {prompt_only, peak}`: Default `prompt_only` (exact paper baseline). `peak` enables an oracle upper bound using true output length $d_{\text{total}}$ at admission to guarantee zero KV overflow.
- `--baseline-include-req`: Augments Baseline-RL's state with arriving request features ($1 + 2N$ features). Default is False (retaining the paper's expert-only state).
- `--use-mask`: Enables invalid action masking for SAC during training and evaluation (masks saturated experts where $W \ge 5$). Default is False (unmasked).
- `--tokenizer {approx, tiktoken}`: Selects tokenization method in `prepare`. `approx` uses $1.33 \times \text{words}$; `tiktoken` uses exact GPT-2 BPE tokens.

## Quick start (5-minute dry run, laptop)

> **Disclaimer:** All numbers produced in this quickstart come from **synthetic data** (`data/synthetic.npz`) and prove code correctness only. Real results require `data/mix.npz`.

```bash
pip install -r requirements.txt
make test                                                     # runs 21 unit tests
python -m qos.cli synthetic --out data/synthetic.npz          # generates synthetic data
python -m qos.cli predict --data data/synthetic.npz           # trains predictors on synthetic data
python -m qos.cli eval --data data/synthetic.npz --lam 5 --n-req 200 --policies rr,sqf,br,greedy
```

## Real data (Mix-Instruct, MIT-licensed on Hugging Face: `llm-blender/mix-instruct`)

Train and validation files include per-candidate BERTScore for candidate LLMs. We use the validation file for dev/test (split in half) and train files for learning:

```bash
# 1) Download real validation file from Hugging Face
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('llm-blender/mix-instruct', 'val_data_prepared.jsonl', repo_type='dataset', local_dir='data/raw')"

# 2) Inspect real record schema
python -m qos.cli inspect --path data/raw/val_data_prepared.jsonl

# 3) Build the compact dataset (prepared and tested on 5,000 real records)
python -m qos.cli prepare --val-path data/raw/val_data_prepared.jsonl --train-n 5000 --val-n 5000 --out data/mix.npz

# 4) Train predictors on real data (10 buckets)
python -m qos.cli predict --data data/mix.npz --epochs 30
```

## 5-day plan (laptop now, DGX from day 3-4)

| Day | Where | Do | Done when |
|---|---|---|---|
| 1 | laptop | Install, run tests, dry run, `inspect` + `prepare` real data, `predict`, run heuristics on real data | `data/mix.npz` exists; predictor accuracy printed; RR/SQF/BR/Greedy numbers |
| 2 | laptop | Short training runs (50k steps) of the three RL configs; pick an **informative load** λ (not saturated, not idle) with a heuristic sweep | Learning curves in `runs/*/log.csv`; chosen λ and L written down |
| 3 | laptop to college | Freeze configs; copy repo to the DGX; check GPU and vLLM install | `nvidia-smi` and `import vllm` work |
| 4 | DGX | `calibrate_vllm` for 1-3 model sizes; `scripts/train_all.sh` (300k steps, run configs in parallel) | `profiles/*.json`; `runs/*/best.pt` |
| 5 | DGX | `scripts/eval_all.sh` (10+ seeds, sweeps, ablations), plots, fill the results table | `results/*.csv/png` |

If time runs short, cut in this order: N sweep, L sweep, BurstGPT, DeepSets. **Do not cut**: the main comparison
with confidence intervals, the three-config RL ablation, and the greedy add-on.

## Experiments (what the paper's figures correspond to)

```bash
# main comparison (paper: QoS vs baselines at N=6, lambda=5)
python -m qos.cli eval --data data/mix.npz --lam 5 --n-req 500 --seeds 10 --policies rr,sqf,br,greedy \
  --rl "Baseline-RL=runs/baseline/best.pt:expert_only:flat" \
  --rl "+State=runs/state/best.pt:full:flat" \
  --rl "QoS-RL=runs/qos/best.pt:full:flat" --out results/main.csv

# sweeps: lam (test-time), L_ms, workload=poisson,bursty,burstgpt ; n_experts needs per-N retraining
python -m qos.cli eval ... --sweep lam=2,3,5,7,10 --out results/lam.csv
python -m qos.cli eval ... --sweep L_ms=20,25,30,35,40 --out results/L.csv

# predictor ablation (paper: with vs without score/length prediction)
python -m qos.cli eval ... --pred-s none --pred-d none --policies greedy,br
python -m qos.cli eval ... --pred-s oracle --pred-d oracle --policies greedy,br   # upper bound

python -m qos.plot_results results/main.csv
```

Three RL configs mirror the paper's ablation: `Baseline-RL` (expert-level state, plain reward),
`+State` (rich state, plain reward), `QoS-RL` (rich state + impact-aware reward).

## Things you must decide and state in the paper

1. **Latency constants.** `ExpertCfg` defaults are placeholders. Use `calibrate_vllm.py` on the DGX and pass `--profile`.
2. **Which experts.** `make_experts` picks `n` models spread across the quality ranking. The paper does not say which six it used.
3. **Token counts.** Output/input lengths are estimated as words × 1.33, not with each model's tokenizer.
4. **Memory model.** Admission checks KV tokens; no mid-flight eviction.
5. **Dev/test.** Checkpoints are chosen on `dev`; report on `test` (`--split test`, the default).
6. **Training budget.** The paper trains 1M steps; you will likely use fewer. Report the number.
7. **Sim-to-real.** This is a simulator. If time allows, replay about 300 requests on live vLLM and report the latency error.

## Files

```
qos/sim.py          simulator, QoS, impact estimator, metrics
qos/env.py          state builder + RL environment (reward modes)
qos/policies.py     RR, SQF, BR, Greedy, RLPolicy
qos/sac.py          discrete SAC (flat / DeepSets encoders)
qos/predictors.py   score & length predictors
qos/data.py         Mix-Instruct loader, synthetic data, Data class
qos/workload.py     Poisson, bursty (MMPP), BurstGPT arrivals
qos/calibrate_vllm.py  fit b, k1, k2 on a real GPU
qos/cli.py          synthetic | inspect | prepare | predict | train | eval
qos/plot_results.py
tests/test_sim.py
scripts/            train_all.sh, eval_all.sh
```
