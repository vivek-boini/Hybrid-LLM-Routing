# Explicit Assumptions in QoS-Aware LLM Routing Reproduction

This document catalogs every explicit design choice, hardware assumption, and mathematical parameter in this repository.

---

## 1. Hardware Latency Model Parameters (`sim.ExpertCfg`)
- **Default Baseline Profile (Placeholder roughly matching an NVIDIA RTX 4090 / 7B model):**
  - Base decode iteration overhead: $b = 0.020\text{ s}$ ($20\text{ ms}$).
  - Prefill cost per input token: $k_1 = 1.2 \times 10^{-4}\text{ s/token}$.
  - Decode slowdown per total batch context token: $k_2 = 1.5 \times 10^{-6}\text{ s/token}$.
  - KV-cache capacity: $\text{cap\_tokens} = 3,600\text{ tokens}$ (or 20,000 for server-class deployments).
- **Heterogeneity Scaling:**
  - Experts have their latency parameters scaled by $m \sim \text{Uniform}(1 - \text{hetero}, 1 + \text{hetero})$, with $\text{hetero} = 0.2$ by default, modeling different model sizes, quantization levels, or GPU clock states.

---

## 2. Queueing & Memory Admission Model
- **Running Capacity ($R$) and Waiting Capacity ($W$):**
  - $R = 5$, $W = 5$ per expert, matching the paper's specification ($R = W = 5$).
- **Memory Admission Bounds:**
  - **`prompt_only` (Default Reproduction Setting):**
    Admit waiting requests if current context plus prompt tokens fit in KV memory:
    $$S_{\text{ctx}} + \text{head}.p + \text{reserve} \le \text{cap\_tokens}$$
    Matches the reproduction baseline, but risks mid-flight memory overcommit as tokens decode without mid-flight eviction.
  - **`peak` (Opt-in Oracle Bound):**
    Verifies that cumulative peak required tokens fit in KV memory:
    $$\sum_{r \in \text{running}} (r.p + r.d_{\text{total}}) + \text{head}.p + \text{head}.d_{\text{total}} + \text{reserve} \le \text{cap\_tokens}$$
    > [!IMPORTANT]
    > **Oracle Assumption Warning:** `peak` admission uses each request's **true output length** ($d_{\text{total}}$) at admission time. In a deployed router, true output length is not known in advance. This setting acts as a conservative oracle upper bound on admission safety.
- **Reservation Buffer:**
  - A fixed buffer of $\text{reserve} = 32\text{ tokens}$ is reserved to prevent boundary edge deadlocks.

---

## 3. Workload Assumptions
- **Poisson Arrival Process:** Inter-arrival times drawn from $\text{Exponential}(1/\lambda)$ with mean rate $\lambda = 5.0\text{ req/s}$.
- **Bursty (MMPP):** Two-state continuous-time Markov chain alternating between high-intensity ($\lambda_{\text{hi}}$) and low-intensity ($\lambda_{\text{lo}}$) arrival rates with mean dwell time $5.0\text{ s}$.
- **BurstGPT:** Rescales actual Microsoft trace timestamp windows to match target arrival rate $\lambda$.

---

## 4. Synthetic Data Disclaimers
- All smoke tests and dry runs using `data/synthetic.npz` operate on **synthetic data**.
- Numbers produced on synthetic data demonstrate code correctness and software mechanics only; they do not represent real-world benchmark performance.
