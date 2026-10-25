"""Fit the simulator's latency constants on a real GPU with vLLM.   >>> RUN THIS ON THE DGX <<<

    python -m qos.calibrate_vllm --model <hf-model-id-or-local-path> --out profiles/7b.json

Measures
  1) prefill : max_tokens=1, prompt length P in a grid            ->  time = c + k1 * P
  2) decode  : batch B (1..5) x context P0, generate n tokens     ->  iteration = b + k2 * S,
               S = B * (P0 + n/2),  iteration = (T_n - T_1) / (n - 1)
  3) KV-cache capacity in tokens (num_gpu_blocks * block_size)

Writes JSON {b, k1, k2, cap_tokens} that `--profile` accepts. Untested against a live GPU in the environment where
this was written: vLLM's Python API changes between versions, so adjust `LLM(...)`/`generate(...)` calls if needed.
Use one profile per model size; give different experts different profiles to model heterogeneity.
"""
import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-tokens", type=int, default=64)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams  # imported late so the rest of the repo works without vLLM

    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_len, dtype="float16")
    vocab = min(llm.get_tokenizer().vocab_size, 30000)
    rng = np.random.default_rng(0)

    def prompt(n):
        return {"prompt_token_ids": rng.integers(100, vocab, n).tolist()}

    def timed(prompts, max_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        t = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - t

    for _ in range(3):  # warm-up (CUDA graphs, kernels)
        timed([prompt(128)], 8)

    # 1) prefill
    P = [64, 128, 256, 512, 1024]
    t1 = [float(np.median([timed([prompt(p)], 1) for _ in range(a.reps)])) for p in P]
    k1, _c = np.polyfit(P, t1, 1)

    # 2) decode
    S, it = [], []
    for B in (1, 2, 3, 4, 5):
        for P0 in (128, 256, 512):
            ps = [prompt(P0) for _ in range(B)]
            tn = np.median([timed(ps, a.n_tokens) for _ in range(a.reps)])
            t_1 = np.median([timed(ps, 1) for _ in range(a.reps)])
            S.append(B * (P0 + a.n_tokens / 2))
            it.append((tn - t_1) / (a.n_tokens - 1))
    k2, b = np.polyfit(S, it, 1)

    # 3) KV capacity
    try:
        cc = llm.llm_engine.cache_config
        cap = int(cc.num_gpu_blocks * cc.block_size)
    except Exception:
        cap = 20000
        print("could not read KV capacity from vLLM; defaulting to", cap)

    prof = dict(b=float(b), k1=float(k1), k2=float(max(k2, 0.0)), cap_tokens=cap)
    json.dump(prof, open(a.out, "w"), indent=2)
    print(json.dumps(prof, indent=2))
    print("fit quality (decode):  R^2 =", float(np.corrcoef(S, it)[0, 1] ** 2))


if __name__ == "__main__":
    main()
