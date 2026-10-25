"""Discrete-event simulator of N edge LLM experts (iteration-level, vLLM-like).

Per expert n:
  * waiting queue (cap W) and running batch (cap R) -- paper uses R = W = 5
  * every iteration produces one token for every running request
  * iteration time  T = b_n + k2_n * S  (+ k1_n * p_new if a request is prefilled in it)
        S = sum over running requests of (input tokens + tokens generated so far)
    -> decode cost grows with total context in the batch (paper: decode-latency model), and a newly
       admitted request's prefill blocks the batch (paper: prefill-latency model).
  * KV-cache capacity in tokens limits admission.

QoS of a finished request (paper: QoS definition):   phi = score * 1[latency_per_token <= L]
latency_per_token = (finish - arrival) / output_tokens   (queueing + prefill + decode).

!! b, k1, k2 and cap_tokens below are PLACEHOLDERS (roughly a 7B model on a 4090-class GPU).
   Replace them with values measured on your GPU: `python -m qos.calibrate_vllm ...`.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ExpertCfg:
    name: str = "expert"
    b: float = 0.020          # base decode iteration time (s)
    k1: float = 1.2e-4        # prefill seconds per input token
    k2: float = 1.5e-6        # extra decode seconds per context token in the batch
    cap_tokens: int = 3600    # KV-cache capacity (tokens)


@dataclass
class Req:
    rid: int
    idx: int
    arrival: float
    p: int
    expert: int = -1
    d_total: int = 0
    score: float = 0.0
    pred_s: float = 0.0
    pred_d: float = 0.0
    d_done: int = 0
    start: float = -1.0
    finish: float = -1.0
    dropped: bool = False
    drop_cause: str | None = None

    @property
    def lat_per_tok(self) -> float:
        return (self.finish - self.arrival) / max(1, self.d_total)


class Expert:
    def __init__(self, cfg: ExpertCfg):
        self.cfg = cfg
        self.running: list[Req] = []
        self.waiting: deque[Req] = deque()
        self.busy = False
        self.new_in_iter: list[Req] = []

    def ctx_tokens(self) -> int:
        return sum(r.p + r.d_done for r in self.running)

    def peak_tokens(self) -> int:
        """Maximum context tokens this expert's running batch will reach if all requests finish."""
        return sum(r.p + r.d_total for r in self.running)

    def util(self) -> float:
        return min(1.0, self.ctx_tokens() / self.cfg.cap_tokens)


class Simulator:
    def __init__(self, data, expert_ids, cfgs, L: float = 0.030, run_cap: int = 5,
                 wait_cap: int = 5, reserve: int = 32, admission_bound: str = "prompt_only"):
        self.data, self.ids, self.cfgs = data, list(expert_ids), cfgs
        self.N, self.L = len(self.ids), L
        self.run_cap, self.wait_cap, self.reserve = run_cap, wait_cap, reserve
        self.slots = run_cap + wait_cap
        self.admission_bound = admission_bound  # "prompt_only" (paper original default) or "peak" (opt-in oracle bound)
        self.kv_overflow_count = 0

    # ------------------------------------------------------------------ run control
    def reset(self, arrival_times: np.ndarray, req_idx: np.ndarray):
        self.experts = [Expert(c) for c in self.cfgs]
        self.events: list = []
        self._seq = itertools.count()
        self.t = 0.0
        self.arrivals, self.req_idx, self.next_i = arrival_times, req_idx, 0
        self.reqs: list[Req] = []
        self.completed: list[Req] = []
        self.pending: Req | None = None
        self.util_samples: list[float] = []
        self.expert_util_samples: list[list[float]] = []
        self.kv_overflow_count = 0

    def advance(self) -> Req | None:
        """Process events until the next arrival; return it (None once everything has drained)."""
        self.completed = []
        inf = float("inf")
        while True:
            t_arr = self.arrivals[self.next_i] if self.next_i < len(self.arrivals) else inf
            t_ev = self.events[0][0] if self.events else inf
            if t_arr <= t_ev and t_arr < inf:
                self.t = float(t_arr)
                idx = int(self.req_idx[self.next_i])
                self.next_i += 1
                self.pending = Req(rid=len(self.reqs), idx=idx, arrival=self.t, p=int(self.data.p[idx]))
                self.util_samples.append(float(np.mean([e.util() for e in self.experts])))
                self.expert_util_samples.append([float(e.util()) for e in self.experts])
                return self.pending
            if t_ev == inf:
                self.pending = None
                return None
            t, _, n = heapq.heappop(self.events)
            self.t = t
            self._iter_end(n)

    def apply(self, action: int) -> bool:
        """action 0 = drop, action n (1..N) = route to expert n-1. Returns True if accepted."""
        r = self.pending
        self.reqs.append(r)
        if action == 0:
            r.dropped = True
            r.drop_cause = "voluntary"
            return False
        n = action - 1
        e = self.experts[n]
        if len(e.waiting) >= self.wait_cap:
            r.dropped = True  # queue full -> rejected
            r.drop_cause = "queue_full"
            return False
        j = self.ids[n]
        r.expert, r.d_total = n, int(self.data.dlen[r.idx, j])
        r.score, r.pred_s, r.pred_d = float(self.data.score[r.idx, j]), float(self.data.act_s[r.idx, j]), float(self.data.act_d[r.idx, j])
        e.waiting.append(r)
        self._start_iter(n, self.t)
        return True

    # ------------------------------------------------------------------ expert mechanics
    def _start_iter(self, n: int, now: float):
        e = self.experts[n]
        if e.busy:
            return
        new = []
        while e.waiting and len(e.running) < self.run_cap:
            head = e.waiting[0]
            if self.admission_bound == "peak":
                # Strict bound: peak required tokens during full generation must fit in cap_tokens
                fits = e.peak_tokens() + head.p + head.d_total + self.reserve <= e.cfg.cap_tokens
            else:
                # Prompt-only admission (paper's original check, subject to mid-flight overcommit)
                fits = e.ctx_tokens() + head.p + self.reserve <= e.cfg.cap_tokens
            if not fits and e.running:
                break  # wait for memory (if nothing is running, admit anyway to avoid deadlock)
            e.waiting.popleft()
            head.start = now
            e.running.append(head)
            new.append(head)
        if not e.running:
            return
        # Track mid-flight KV overflows if prompt_only admitted without enough peak space
        if e.ctx_tokens() > e.cfg.cap_tokens:
            self.kv_overflow_count += 1
        T = e.cfg.b + e.cfg.k2 * e.ctx_tokens() + e.cfg.k1 * sum(r.p for r in new)
        e.busy = True
        e.new_in_iter = new
        heapq.heappush(self.events, (now + T, next(self._seq), n))

    def _iter_end(self, n: int):
        e = self.experts[n]
        e.busy = False
        still = []
        for r in e.running:
            r.d_done += 1
            if r.d_done >= r.d_total:
                r.finish = self.t
                self.completed.append(r)
            else:
                still.append(r)
        e.running = still
        self._start_iter(n, self.t)

    # ------------------------------------------------------------------ QoS & estimator
    def qos_of(self, r: Req) -> float:
        return r.score if (not r.dropped and r.finish >= 0 and r.lat_per_tok <= self.L) else 0.0

    def impact_penalty(self, n: int, p_j: int, d_j: float, use_pred: bool) -> float:
        """Action impact estimator (paper: action impact estimator): total score of RUNNING requests on expert n that
        would exceed the latency limit if a new request (p_j in, d_j out) were added to it.
        use_pred=False -> true values (training environment); True -> predictions (deployable router)."""
        e = self.experts[n]
        c = e.cfg
        pen = 0.0
        for r in e.running:
            d_i = r.pred_d if use_pred else r.d_total
            rem = max(1.0, d_i - r.d_done)
            K = int(max(1, min(rem, d_j)))
            extra = c.k1 * p_j + c.k2 * (K * p_j + K * (K + 1) / 2.0)
            l_plus = extra / max(1.0, d_i)
            l_now = (self.t - r.arrival) / max(1, r.d_done)
            if l_now + l_plus >= self.L:
                pen += r.pred_s if use_pred else r.score
        return pen

    def est_own_latency(self, n: int, p_j: int, d_hat: float) -> float:
        """Heuristic estimate of the NEW request's latency per token on expert n (uses predictions only)."""
        e = self.experts[n]
        c = e.cfg
        d_hat = max(1.0, d_hat)
        S = e.ctx_tokens()
        iter_t = c.b + c.k2 * (S + p_j + d_hat / 2.0)
        wait = 0.0
        full = len(e.running) >= self.run_cap or (S + p_j + self.reserve > c.cap_tokens)
        if full or e.waiting:
            rem = min((max(1.0, r.pred_d - r.d_done) for r in e.running), default=0.0)
            avg_d = float(np.mean([r.pred_d for r in e.running])) if e.running else d_hat
            wait = rem * iter_t + len(e.waiting) * (avg_d * iter_t / self.run_cap)
        service = c.k1 * p_j + d_hat * iter_t
        return (wait + service) / d_hat

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> dict:
        rs = self.reqs
        done = [r for r in rs if not r.dropped and r.finish >= 0]
        lat = np.array([r.lat_per_tok for r in done]) if done else np.array([0.0])
        e2e = np.array([r.finish - r.arrival for r in done]) if done else np.array([0.0])
        n_rs = max(1, len(rs))
        mean_expert_util = np.mean(self.expert_util_samples, axis=0) if self.expert_util_samples else np.zeros(self.N)

        out = dict(
            qos=float(np.mean([self.qos_of(r) for r in rs])) if rs else 0.0,
            lat_per_token_ms=float(lat.mean() * 1e3),
            lat_p50_ms=float(np.percentile(lat, 50) * 1e3),
            lat_p90_ms=float(np.percentile(lat, 90) * 1e3),
            lat_p95_ms=float(np.percentile(lat, 95) * 1e3),
            lat_p99_ms=float(np.percentile(lat, 99) * 1e3),
            e2e_s=float(e2e.mean()),
            drop_rate=float(np.mean([r.dropped for r in rs])) if rs else 0.0,
            drop_voluntary=float(sum(1 for r in rs if r.drop_cause == "voluntary") / n_rs),
            drop_queue_full=float(sum(1 for r in rs if r.drop_cause == "queue_full") / n_rs),
            drop_kv_full=float(sum(1 for r in rs if r.drop_cause == "kv_full") / n_rs),
            slo_violation=float(np.mean([r.lat_per_tok > self.L for r in done])) if done else 0.0,
            mem_util=float(np.mean(self.util_samples)) if self.util_samples else 0.0,
            mean_score_served=float(np.mean([r.score for r in done])) if done else 0.0,
            kv_overflows=int(self.kv_overflow_count),
        )
        for i in range(self.N):
            out[f"expert_{i}_util"] = float(mean_expert_util[i])
        return out
