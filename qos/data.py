"""Data preparation.

Turns Mix-Instruct (11 LLMs, each with a BERTScore per request) into one compact .npz:

    p      (n,)     input tokens of the request
    score  (n, M)   BERTScore of each model's response   (the "quality" s_{j,n})
    dlen   (n, M)   output tokens of each model's response (d_{j,n})
    emb    (n, D)   text embedding used by the predictors
    split  (n,)     0 = train, 1 = dev (checkpoint selection), 2 = test (final numbers)
    models (M,)     model names

A synthetic generator is included so the whole pipeline can be dry-run before
the real data is downloaded.

NOTE: the Mix-Instruct loader could not be tested against the real files in the
environment where this code was written (no Hugging Face access). Run
`python -m qos.cli inspect --path <jsonl>` first and fix KEYS below if needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Field names expected in each jsonl record. Adjust here if `inspect` shows otherwise.
KEYS = dict(instruction="instruction", input="input", candidates="candidates",
            model="model", text="text", scores="scores", metric="bertscore")


_TIKTOKEN_ENC = None


def count_tokens(text: str, tokenizer: str = "approx") -> int:
    """Estimate or compute exact token count.

    tokenizer in {"approx", "tiktoken", "gpt2", "bpe"}.
    'approx': max(1, int(round(len(text.split()) * 1.33)))
    'tiktoken' / 'gpt2': exact BPE tokens via tiktoken.
    """
    global _TIKTOKEN_ENC
    if tokenizer in ("tiktoken", "gpt2", "bpe"):
        if _TIKTOKEN_ENC is None:
            import tiktoken
            _TIKTOKEN_ENC = tiktoken.get_encoding("gpt2")
        return max(1, len(_TIKTOKEN_ENC.encode(text)))
    return max(1, int(round(len(text.split()) * 1.33)))


def approx_tokens(text: str) -> int:
    """Cheap token estimate (words * 1.33). Retained for backwards compatibility."""
    return count_tokens(text, "approx")


def validate_mixinstruct_record(rec: dict) -> bool:
    """Validate that a record has the expected keys and candidate BERTScores."""
    if not isinstance(rec, dict):
        return False
    candidates = rec.get(KEYS["candidates"])
    if not isinstance(candidates, list) or len(candidates) == 0:
        return False
    valid_cands = 0
    for c in candidates:
        if isinstance(c, dict) and KEYS["model"] in c and KEYS["text"] in c and KEYS["scores"] in c:
            sc = c[KEYS["scores"]]
            if isinstance(sc, dict) and KEYS["metric"] in sc and sc[KEYS["metric"]] is not None:
                valid_cands += 1
    return valid_cands > 0


# --------------------------------------------------------------------------- #
# Mix-Instruct
# --------------------------------------------------------------------------- #
def iter_jsonl(path: str | None = None, remote_file: str | None = None, limit: int | None = None):
    """Yield records from a local jsonl, or stream the first `limit` lines from the HF Hub."""
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    return
                yield json.loads(line)
    else:
        from huggingface_hub import HfFileSystem  # streams without downloading 1.4 GB
        fs = HfFileSystem()
        with fs.open(f"datasets/llm-blender/mix-instruct/{remote_file}", "r") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    return
                yield json.loads(line)


def inspect_record(path: str) -> None:
    rec = next(iter_jsonl(path=path, limit=1))
    print("top-level keys:", list(rec.keys()))
    cands = rec.get(KEYS["candidates"], [])
    print("n candidates:", len(cands))
    if cands:
        c = cands[0]
        print("candidate keys:", list(c.keys()))
        print("scores:", c.get(KEYS["scores"]))
        print("model:", c.get(KEYS["model"]), "| text[:80]:", str(c.get(KEYS["text"]))[:80])
    valid = validate_mixinstruct_record(rec)
    print("schema valid:", valid)


def read_mixinstruct(path: str | None, remote_file: str | None, limit: int | None,
                     tokenizer: str = "approx"):
    """Return list of (prompt_text, p, {model: (score, ntok)})."""
    rows = []
    for rec in iter_jsonl(path=path, remote_file=remote_file, limit=limit):
        prompt = (rec.get(KEYS["instruction"], "") or "") + "\n" + (rec.get(KEYS["input"], "") or "")
        per_model = {}
        for c in rec.get(KEYS["candidates"], []):
            sc = c.get(KEYS["scores"], {})
            val = sc.get(KEYS["metric"]) if isinstance(sc, dict) else None
            if val is None:
                continue
            per_model[c[KEYS["model"]]] = (float(val), count_tokens(str(c[KEYS["text"]]), tokenizer))
        rows.append((prompt, count_tokens(prompt, tokenizer), per_model))
    return rows


def build_npz(train_rows, val_rows, out_path: str, embed: str = "tfidf", dim: int = 128, tokenizer: str = "approx"):
    rows = train_rows + val_rows
    half = len(val_rows) // 2   # validation set -> first half = dev (checkpoint selection), second half = test
    split = np.array([0] * len(train_rows) + [1] * half + [2] * (len(val_rows) - half), dtype=np.int8)
    # keep models present in >= 99% of rows
    counts: dict[str, int] = {}
    for _, _, pm in rows:
        for m in pm:
            counts[m] = counts.get(m, 0) + 1
    models = sorted(m for m, c in counts.items() if c >= 0.99 * len(rows))
    keep = [i for i, (_, _, pm) in enumerate(rows) if all(m in pm for m in models)]
    rows = [rows[i] for i in keep]
    split = split[keep]
    print(f"{len(models)} models kept: {models}\n{len(rows)} requests with full coverage")
    p = np.array([r[1] for r in rows], dtype=np.int32)
    score = np.array([[r[2][m][0] for m in models] for r in rows], dtype=np.float32)
    dlen = np.array([[r[2][m][1] for m in models] for r in rows], dtype=np.int32)
    texts = [r[0] for r in rows]
    emb = embed_texts(texts, split, embed, dim)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, p=p, score=score, dlen=dlen, emb=emb, split=split,
                        models=np.array(models), tokenizer=np.array(tokenizer))
    print("saved", out_path)


def embed_texts(texts, split, kind: str, dim: int):
    if kind == "minilm":  # better embeddings; needs sentence-transformers (GPU helps)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model.encode(texts, batch_size=128, show_progress_bar=True).astype(np.float32)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(texts)  # fit on all text only (no labels involved -> no label leakage)
    svd = TruncatedSVD(n_components=min(dim, X.shape[1] - 1), random_state=0)
    return svd.fit_transform(X).astype(np.float32)


# --------------------------------------------------------------------------- #
# Synthetic data (dry runs / unit tests)
# --------------------------------------------------------------------------- #
def make_synthetic(out_path: str, n: int = 6000, m: int = 8, n_clusters: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    cl = rng.integers(0, n_clusters, n)
    p = np.clip(rng.lognormal(5.0, 0.6, n), 8, 900).astype(np.int32)
    expertise = rng.uniform(0.45, 0.85, (m, n_clusters))
    score = np.clip(expertise[:, cl].T + rng.normal(0, 0.05, (n, m)), 0.2, 0.95).astype(np.float32)
    verbosity = rng.lognormal(np.log(150), 0.35, m)
    cl_scale = rng.uniform(0.6, 1.6, n_clusters)
    dlen = np.clip(rng.lognormal(np.log(verbosity[None, :] * cl_scale[cl][:, None]), 0.35), 8, 600).astype(np.int32)
    emb = np.eye(n_clusters)[cl] * 2.0 + rng.normal(0, 0.6, (n, n_clusters))
    emb = np.concatenate([emb, rng.normal(0, 1, (n, 10))], axis=1).astype(np.float32)
    r = rng.random(n)
    split = np.where(r < 0.70, 0, np.where(r < 0.85, 1, 2)).astype(np.int8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, p=p, score=score, dlen=dlen, emb=emb, split=split,
                        models=np.array([f"syn-model-{i}" for i in range(m)]))
    print("saved synthetic data to", out_path)


# --------------------------------------------------------------------------- #
# Loader used by everything else
# --------------------------------------------------------------------------- #
class Data:
    """Array bundle shared by predictors, simulator and policies."""

    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.p = z["p"].astype(np.int64)
        self.score = z["score"].astype(np.float64)
        self.dlen = z["dlen"].astype(np.int64)
        self.emb = z["emb"].astype(np.float32)
        self.split = z["split"]
        self.models = [str(x) for x in z["models"]]
        self.tokenizer = str(z["tokenizer"]) if "tokenizer" in z.files else "approx"
        self.n, self.M = self.score.shape
        # predicted / oracle bucket-centre values, filled by `qos.cli predict`
        self.pred_s = np.zeros_like(self.score)
        self.pred_d = np.zeros_like(self.score)
        self.orc_s = np.zeros_like(self.score)
        self.orc_d = np.zeros_like(self.score)
        for k in ("pred_s", "pred_d", "orc_s", "orc_d"):
            if k in z.files:
                setattr(self, k, z[k].astype(np.float64))
        self.use_predictions("pred", "pred")

    def idx(self, split: str) -> np.ndarray:
        """split in {train, dev, test}. dev = checkpoint selection, test = final reported numbers."""
        code = {"train": 0, "dev": 1, "test": 2}[split]
        if code == 2 and not (self.split == 2).any():   # old/synthetic files without a test split
            code = 1
        return np.where(self.split == code)[0]

    def use_predictions(self, s_mode: str = "pred", d_mode: str = "pred"):
        """s_mode/d_mode in {pred, oracle, none}. 'none' reproduces the paper's ZS / ZL ablations."""
        z = {"pred": (self.pred_s, self.pred_d), "oracle": (self.orc_s, self.orc_d)}
        self.act_s = np.zeros_like(self.score) if s_mode == "none" else z[s_mode][0]
        self.act_d = np.zeros_like(self.score) if d_mode == "none" else z[d_mode][1]
        return self
