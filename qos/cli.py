"""python -m qos.cli <command> ...   (see README.md for the full workflow)"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .data import Data, build_npz, inspect_record, make_synthetic, read_mixinstruct
from .env import RoutingEnv
from .evaluate import evaluate, paired_pvalue
from .policies import BertRouter, GreedyImpact, RLPolicy, RoundRobin, ShortestQueue
from .setup import make_experts

HEURISTICS = {
    "rr": RoundRobin,
    "sqf": ShortestQueue,
    "br": BertRouter,
    "greedy": GreedyImpact,
    "greedy_score_only": lambda: GreedyImpact(mode="score_only"),
    "greedy_no_penalty": lambda: GreedyImpact(mode="no_penalty"),
}


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def get_file_sha256(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unknown"


def common(p):
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--data", default=None, help="Path to npz data file")
    p.add_argument("--n-experts", type=int, default=6)
    p.add_argument("--profile", default=None, help="JSON from calibrate_vllm (b,k1,k2,cap_tokens)")
    p.add_argument("--hetero", type=float, default=0.2)
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--latency-ms", type=float, default=30.0, help="latency-per-token limit L")
    p.add_argument("--n-req", type=int, default=300)
    p.add_argument("--workload", default="poisson", choices=["poisson", "bursty", "burstgpt"])
    p.add_argument("--burstgpt-csv", default=None)
    p.add_argument("--pred-s", default="pred", choices=["pred", "oracle", "none"])
    p.add_argument("--pred-d", default="pred", choices=["pred", "oracle", "none"])
    p.add_argument("--admission-bound", default="prompt_only", choices=["prompt_only", "peak"],
                   help="KV-cache admission bound: 'prompt_only' (replicates paper default) or 'peak' (opt-in oracle bound)")
    p.add_argument("--baseline-include-req", action="store_true",
                   help="Include arriving request features in expert_only (Baseline RL) state")


def apply_yaml_config(a):
    if getattr(a, "config", None) and os.path.exists(a.config):
        import yaml
        with open(a.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict):
            for k, v in cfg.items():
                setattr(a, k.replace("-", "_"), v)


def make_env(a, split, state="full", reward="qos", **over):
    if not a.data:
        raise ValueError("--data is required (either via CLI argument or in YAML config)")
    data = Data(a.data).use_predictions(a.pred_s, a.pred_d)
    ids, cfgs = make_experts(data, over.get("n_experts", a.n_experts), a.profile, a.hetero)
    return RoutingEnv(data, ids, cfgs, split=split, lam=over.get("lam", a.lam), n_req=a.n_req,
                      L=over.get("L_ms", a.latency_ms) / 1000.0, state=state, reward=reward,
                      workload=over.get("workload", a.workload), burstgpt_csv=a.burstgpt_csv,
                      baseline_include_req=over.get("baseline_include_req", getattr(a, "baseline_include_req", False)),
                      admission_bound=over.get("admission_bound", getattr(a, "admission_bound", "prompt_only")))


def cmd_synthetic(a):
    make_synthetic(a.out, n=a.n, m=a.m)


def cmd_inspect(a):
    inspect_record(a.path)


def cmd_prepare(a):
    tr = read_mixinstruct(a.train_path, "train_data_prepared.jsonl", a.train_n, tokenizer=a.tokenizer)
    va = read_mixinstruct(a.val_path, "val_data_prepared.jsonl", a.val_n, tokenizer=a.tokenizer)
    build_npz(tr, va, a.out, embed=a.embed, tokenizer=a.tokenizer)


def cmd_predict(a):
    from .predictors import train_predictors
    stats = train_predictors(a.data, B=a.buckets, epochs=a.epochs, device=a.device)
    print(json.dumps(stats, indent=2), "\n(paper reports top-1 63.39% score / 72.97% length with DistilBERT)")


def cmd_train(a):
    apply_yaml_config(a)
    import torch
    from .sac import DiscreteSAC, ReplayBuffer
    dev = a.device if (a.device != "auto") else ("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(a, "train", a.state, a.reward)
    eval_env = make_env(a, "dev", a.state, a.reward)
    agent = DiscreteSAC(env.obs_dim, env.n_actions, env.sb.layout(), a.encoder, device=dev)
    use_mask = getattr(a, "use_mask", False)
    buf = ReplayBuffer(env.obs_dim, n_act=env.n_actions, store_mask=use_mask)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump({k: v for k, v in vars(a).items() if k != "f"}, open(out / "args.json", "w"), indent=2)
    log = csv.writer(open(out / "log.csv", "w", newline=""))
    log.writerow(["step", "eval_qos", "eval_lat_ms", "eval_drop", "train_ep_return", "alpha"])
    best, ep, ep_ret, alpha, t0 = -1.0, 0, 0.0, 1.0, time.time()
    obs, ep_rets = env.reset(seed=ep), []
    for step in range(1, a.steps + 1):
        mask = env.action_mask() if use_mask else None
        if step <= a.warmup:
            act = np.random.choice(np.where(mask)[0]) if use_mask else np.random.randint(env.n_actions)
        else:
            act = agent.act(obs, action_mask=mask)
        nobs, r, done, _ = env.step(act)
        nmask = env.action_mask() if use_mask else None
        buf.add(obs, act, r, nobs, float(done), m=mask, m2=nmask)
        obs, ep_ret = nobs, ep_ret + r
        if done:
            ep_rets.append(ep_ret); ep, ep_ret = ep + 1, 0.0
            obs = env.reset(seed=ep)
        if step > a.warmup and step % a.update_every == 0:
            _, _, alpha = agent.update(buf.sample(a.batch, dev))
        if step % a.eval_every == 0:
            m = evaluate(eval_env, RLPolicy(agent, use_mask=use_mask), seeds=range(1000, 1000 + a.eval_seeds))
            log.writerow([step, m["qos"], m["lat_per_token_ms"], m["drop_rate"], float(np.mean(ep_rets[-20:] or [0])), alpha])
            print(f"step {step:>8} | eval QoS {m['qos']:.4f} | lat {m['lat_per_token_ms']:.1f} ms/tok | "
                  f"drop {m['drop_rate']:.3f} | alpha {alpha:.3f} | {time.time() - t0:.0f}s", flush=True)
            if m["qos"] > best:
                best = m["qos"]; agent.save(out / "best.pt")
    agent.save(out / "last.pt")
    print("best eval QoS", round(best, 4))


def _load_rl(spec, env_kwargs):
    """spec = label=ckpt:state:encoder  (e.g. QoS-RL=runs/qos/best.pt:full:flat)"""
    label, rest = spec.split("=", 1)
    ckpt, state, enc = rest.split(":")
    return label, ckpt, state, enc


def cmd_eval(a):
    apply_yaml_config(a)
    import torch
    import scipy
    from .sac import DiscreteSAC

    sweeps = [("base", {})]
    if a.sweep:
        key, vals = a.sweep.split("=")
        cast = {"lam": float, "n_experts": int, "L_ms": float, "workload": str}[key]
        sweeps = [(f"{key}={v}", {key: cast(v)}) for v in vals.split(",")]

    git_hash = get_git_hash()
    data_sha = get_file_sha256(a.data) if (a.data and os.path.exists(a.data)) else "unknown"
    data_obj = Data(a.data) if (a.data and os.path.exists(a.data)) else None
    tokenizer_tag = getattr(data_obj, "tokenizer", "approx") if data_obj else "approx"
    admission_bound = getattr(a, "admission_bound", "prompt_only")
    use_mask = getattr(a, "use_mask", False)
    pkg_versions = f"torch={torch.__version__},numpy={np.__version__},scipy={scipy.__version__}"
    dataset_label = "real data, uncalibrated latency constants" if (a.data and "mix" in str(a.data)) else "synthetic data"

    rows = []
    for tag, over in sweeps:
        policies = []
        for n in a.policies.split(","):
            n = n.strip()
            if not n:
                continue
            pol_fn = HEURISTICS[n]
            pol_inst = pol_fn() if callable(pol_fn) else pol_fn
            policies.append((pol_inst.name, make_env(a, a.split, "expert_only", "baseline", **over), pol_inst))

        for spec in a.rl or []:
            label, ckpt, state, enc = _load_rl(spec, over)
            env = make_env(a, a.split, state, "qos", **over)
            agent = DiscreteSAC(env.obs_dim, env.n_actions, env.sb.layout(), enc)
            try:
                agent.load(ckpt)
            except RuntimeError:
                print(f"[skip] {label}: checkpoint does not match this setting ({tag})")
                continue
            policies.append((label, env, RLPolicy(agent, label, use_mask=use_mask)))

        # Paired evaluation over identical seeds
        pol_episodes = {}
        for name, env, pol in policies:
            m, episodes = evaluate(env, pol, seeds=range(a.seeds), return_episodes=True)
            pol_episodes[name] = [ep["qos"] for ep in episodes]

            # Reference policy for paired significance test is the first policy (typically RoundRobin or Greedy)
            ref_name = policies[0][0]
            ref_qos = pol_episodes.get(ref_name, [])
            p_val = paired_pvalue(pol_episodes[name], ref_qos) if (len(ref_qos) > 1 and name != ref_name) else 1.0

            row = dict(
                setting=tag,
                policy=name,
                git_hash=git_hash,
                data_sha256=data_sha,
                tokenizer=tokenizer_tag,
                admission_bound=admission_bound,
                mask_setting=use_mask,
                seeds=a.seeds,
                package_versions=pkg_versions,
                dataset_label=dataset_label,
                p_val_vs_ref=p_val,
                **m
            )
            rows.append(row)
            print(f"{tag:>14} | {name:<12} QoS {m['qos']:.4f} ±{m['qos_boot_ci95']:.4f} | "
                  f"lat {m['lat_per_token_ms']:.1f} ms/tok (p50 {m['lat_p50_ms']:.1f}, p95 {m['lat_p95_ms']:.1f}) | "
                  f"drop {m['drop_rate']:.3f} (vol {m['drop_voluntary']:.3f}, q_full {m['drop_queue_full']:.3f}) | "
                  f"viol {m['slo_violation']:.3f}")

    if a.out and rows:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            f.write(f"# git_hash: {git_hash}\n")
            f.write(f"# data_file: {a.data}\n")
            f.write(f"# data_sha256: {data_sha}\n")
            f.write(f"# tokenizer: {tokenizer_tag}\n")
            f.write(f"# admission_bound: {admission_bound}\n")
            f.write(f"# mask_setting: {use_mask}\n")
            f.write(f"# seeds: {a.seeds}\n")
            f.write(f"# package_versions: {pkg_versions}\n")
            f.write(f"# dataset_label: {dataset_label}\n")
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser(prog="qos")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("synthetic"); p.add_argument("--out", default="data/synthetic.npz")
    p.add_argument("--n", type=int, default=6000); p.add_argument("--m", type=int, default=8); p.set_defaults(f=cmd_synthetic)

    p = sp.add_parser("inspect"); p.add_argument("--path", required=True); p.set_defaults(f=cmd_inspect)

    p = sp.add_parser("prepare", help="Mix-Instruct -> npz")
    p.add_argument("--train-path"); p.add_argument("--val-path")
    p.add_argument("--train-n", type=int, default=20000); p.add_argument("--val-n", type=int, default=5000)
    p.add_argument("--embed", default="tfidf", choices=["tfidf", "minilm"])
    p.add_argument("--tokenizer", default="approx", choices=["approx", "tiktoken"], help="Tokenizer for word/token counts")
    p.add_argument("--out", default="data/mix.npz")
    p.set_defaults(f=cmd_prepare)

    p = sp.add_parser("predict", help="train score/length predictors, write predictions into the npz")
    p.add_argument("--data", required=True); p.add_argument("--buckets", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--device", default="cpu"); p.set_defaults(f=cmd_predict)

    p = sp.add_parser("train", help="train a discrete-SAC router"); common(p)
    p.add_argument("--state", default="full", choices=["expert_only", "full"])
    p.add_argument("--reward", default="qos", choices=["baseline", "qos"])
    p.add_argument("--encoder", default="flat", choices=["flat", "deepsets"])
    p.add_argument("--steps", type=int, default=300000); p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--update-every", type=int, default=1); p.add_argument("--batch", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=10000); p.add_argument("--eval-seeds", type=int, default=5)
    p.add_argument("--device", default="auto"); p.add_argument("--out", required=True)
    p.add_argument("--use-mask", action="store_true", help="Enable invalid action masking during training")
    p.set_defaults(f=cmd_train)

    p = sp.add_parser("eval", help="compare policies (optionally over a sweep)"); common(p)
    p.add_argument("--policies", default="rr,sqf,br,greedy"); p.add_argument("--rl", action="append")
    p.add_argument("--seeds", type=int, default=10); p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--sweep", default=None, help="e.g. lam=2,3,5,7,10 | n_experts=3,6,9 | L_ms=20,30,40 | workload=poisson,bursty")
    p.add_argument("--use-mask", action="store_true", help="Enable action masking for RL policy")
    p.add_argument("--out", default=None); p.set_defaults(f=cmd_eval)

    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
