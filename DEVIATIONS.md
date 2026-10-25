# Deviations from Original Paper (Yang et al., IEEE TMC 2025)

This document provides a transparent and verified accounting of differences between this reproduction codebase and the paper:
> **"Quality-of-Service Aware LLM Routing for Edge Computing With Multiple Experts"** (Yang et al., IEEE Transactions on Mobile Computing 2025 / arXiv:2508.00234).

---

## 1. Verified Paper Specifications & Code Deviations

### A. Memory Constraints & Bounded Admission
- **Paper Specification (Section IV-A, Eq. 1 & Section V-B2):**
  - Section IV-A Eq. (1) states: $C_{j,n,t} \le C_n$, asserting that the memory allocated to request $q_j$ cannot exceed the expert's GPU memory capacity $C_n$.
  - Section V-B2 defines expert GPU memory utilization as $e_{n,t} = \sum_{j \in Q^{\text{running}}_{n,t}} C_{j,n,t} / C_n$.
  - **Silence in Paper:** The paper does not specify a concrete admission gating rule, an output length reservation algorithm, or a mid-flight eviction policy when cumulative running KV usage reaches $C_n$.
  - **Reproduction Choice (Default):** `admission_bound="prompt_only"`, which admits head requests from the waiting queue as long as $S_{\text{ctx}} + p_{\text{head}} + \text{reserve} \le \text{cap\_tokens}$.
  - **Opt-in Alternative:** `admission_bound="peak"`, which uses each request's true output length $d_{\text{total}}$ at admission ($S_{\text{ctx, peak}} + p_{\text{head}} + d_{\text{head}} + \text{reserve} \le \text{cap\_tokens}$) to guarantee no mid-flight overflow. This is an oracle assumption and is disabled by default.

---

### B. Queue Capacity & Action Masking
- **Paper Specification (Section V-A, Section V-D, Section VI-A):**
  - Section V-D specifies: *"By default, the capacity of the running queue and waiting queue for each edge expert is set to 5."*
  - **Silence in Paper:** The paper does not specify how the DRL router handles an action that selects an expert whose waiting queue is already at capacity $|Q^{\text{waiting}}| \ge 5$ (e.g. invalid action masking vs. immediate drop vs. negative penalty).
  - **Reproduction Choice (Default):** Drops the request ($r.\text{dropped}=\text{True}$, returning 0 QoS), matching the baseline behavior where queue overflows are rejected.
  - **Opt-in Alternative:** `--use-mask`, which masks out saturated experts directly in the policy's logit distribution ($logit = -10^9$), guaranteeing the agent never selects a full queue. Disabled by default.

---

### C. State Abstraction Architecture
- **Paper Specification (Section V-B2):**
  - Uses a Heterogeneous Attention Network (HAN) with 2 layers and 4 attention heads implemented via PyTorch Geometric to embed a graph of arriving request, expert nodes, running slots, and waiting slots.
  - Uses DistilBERT with an expert token and 10 classification buckets for generation score and length prediction.
- **Reproduction Implementation:**
  - Employs a flat padded observation vector with a multi-layer perceptron (or a permutation-invariant DeepSets masked-pooling encoder) in `qos/sac.py`.
  - Employs a two-head MLP on frozen embeddings in `qos/predictors.py`.
  - **Rationale:** Implemented as a lightweight, CPU-runnable baseline for laptop execution without PyTorch Geometric dependencies.

---

### D. Greedy Impact Policy (Add-on Baseline)
- **Paper Specification:**
  - Not reported in the paper. The paper compares against Round-Robin (RR), Shortest Queue First (SQF), BERT Router (BR), and Baseline RL.
- **Reproduction Implementation (`qos/policies.py::GreedyImpact`):**
  - Evaluates the paper's action-impact estimator directly in a one-step greedy decision rule ($\text{Gain} - \text{Penalty}$). Added to isolate the benefit of the estimator itself from multi-step reinforcement learning.
