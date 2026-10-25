"""Unit tests for Mix-Instruct schema validation, tokenizer options, and BurstGPT parsing (Phase 1f/1g).

Tests:
1. validate_mixinstruct_record with valid, missing candidates, and corrupted score records
2. count_tokens comparing approx (words * 1.33) and tiktoken BPE
3. burstgpt_arrivals handling both numeric timestamps and datetime string formats
"""
import tempfile
import numpy as np
import pandas as pd
import pytest

from qos.data import count_tokens, approx_tokens, validate_mixinstruct_record, KEYS
from qos.workload import burstgpt_arrivals


def test_validate_mixinstruct_record():
    """Validates real schema structure of Mix-Instruct records."""
    valid_record = {
        "id": "item_0",
        "instruction": "Summarize this article.",
        "input": "Long text...",
        "candidates": [
            {
                "model": "model_a",
                "text": "This is response A.",
                "scores": {"bertscore": 0.75, "rouge1": 0.4}
            },
            {
                "model": "model_b",
                "text": "This is response B.",
                "scores": {"bertscore": 0.82, "rouge1": 0.45}
            }
        ]
    }
    assert validate_mixinstruct_record(valid_record) is True

    # Missing candidates
    invalid_no_cands = {"id": "1", "instruction": "hi", "candidates": []}
    assert validate_mixinstruct_record(invalid_no_cands) is False

    # Missing bertscore metric
    invalid_no_metric = {
        "id": "2",
        "candidates": [{"model": "m1", "text": "resp", "scores": {"other_metric": 1.0}}]
    }
    assert validate_mixinstruct_record(invalid_no_metric) is False


def test_token_counting_methods():
    """count_tokens returns positive integers for both approx and tiktoken methods."""
    sample_text = "The quick brown fox jumps over the lazy dog."
    
    cnt_approx = count_tokens(sample_text, "approx")
    assert cnt_approx > 0
    assert approx_tokens(sample_text) == cnt_approx
    
    # Tiktoken test
    try:
        import tiktoken
        cnt_bpe = count_tokens(sample_text, "tiktoken")
        assert cnt_bpe > 0
        # For simple English sentences, words*1.33 is typically within 20% of BPE
        assert abs(cnt_approx - cnt_bpe) <= 5
    except ImportError:
        pass


def test_burstgpt_numeric_and_datetime():
    """burstgpt_arrivals handles both numeric float seconds and ISO datetime string columns."""
    rng = np.random.default_rng(42)
    
    # 1. Test numeric timestamps
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df_num = pd.DataFrame({
            "Timestamp": np.linspace(0, 100, 20),
            "Model": ["gpt-3.5-turbo"] * 20
        })
        df_num.to_csv(f.name, index=False)
        arr_num = burstgpt_arrivals(f.name, lam=5.0, n=10, rng=rng)
        assert len(arr_num) == 10
        assert (np.diff(arr_num) >= 0).all()

    # 2. Test datetime string timestamps
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        dates = pd.date_range("2023-01-01 12:00:00", periods=20, freq="10s")
        df_str = pd.DataFrame({
            "Logical Time": dates.astype(str),
            "Model": ["gpt-4"] * 20
        })
        df_str.to_csv(f.name, index=False)
        arr_str = burstgpt_arrivals(f.name, lam=5.0, n=10, rng=rng)
        assert len(arr_str) == 10
        assert (np.diff(arr_str) >= 0).all()
