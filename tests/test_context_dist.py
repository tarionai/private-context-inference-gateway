"""Offline tests for the context-size distribution analyzer (no gateway, no GPU)."""

import json

from telemetry.context_dist import (
    percentile,
    read_context_tokens,
    regime_for,
)


def test_regime_boundaries_match_the_decision_table():
    assert regime_for(0).batch == 16 and regime_for(249).batch == 16
    assert regime_for(250).batch == 4 and regime_for(999).batch == 4
    assert regime_for(1_000).batch == 2 and regime_for(2_999).batch == 2
    assert regime_for(3_000).batch == 1 and regime_for(50_000).batch == 1


def test_regime_batch_is_monotonic_non_increasing_in_tokens():
    batches = [regime_for(t).batch for t in (100, 500, 1_500, 5_000)]
    assert batches == [16, 4, 2, 1]
    assert all(a >= b for a, b in zip(batches, batches[1:]))


def test_percentile_nearest_rank():
    data = list(range(1, 101))  # 1..100
    assert percentile(data, 50) == 50
    assert percentile(data, 95) == 95
    assert percentile(data, 99) == 99
    assert percentile([], 95) == 0


def test_read_context_tokens_extracts_field_and_skips_records_without_it(tmp_path):
    audit = tmp_path / "audit.jsonl"
    lines = [
        json.dumps({"response": {"included_context_tokens": 320, "prompt_tokens": 400}}),
        json.dumps({"response": {"included_context_tokens": 1500}}),
        json.dumps({"response": {"cost_usd": 0.0}}),  # legacy record, no field -> skipped
        "",                                            # blank line tolerated
    ]
    audit.write_text("\n".join(lines), encoding="utf-8")
    assert read_context_tokens(audit) == [320, 1500]


def test_read_context_tokens_missing_file_is_empty(tmp_path):
    assert read_context_tokens(tmp_path / "nope.jsonl") == []
