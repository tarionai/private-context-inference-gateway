"""Offline tests for the load tester's pure logic (no network, no GPU).

Covers percentile math, Prometheus metric scraping, sustainable-width selection,
prompt sizing, the diurnal trace shape, and capture-freeze hashing. The wire
calls and live sweep are exercised against the real endpoint, not here.
"""

import json

from serving.loadtest import (
    INTERACTIVE_P95_SLO_MS,
    RequestSample,
    SweepLevelResult,
    _build_prompt,
    _curve_point,
    _diurnal_trace,
    _freeze,
    _max_slo_meeting_width,
    _percentile,
    _scan_metric,
    _summarize_level,
    _sustainable_width,
)


def _level(concurrency: int, sustainable: bool) -> SweepLevelResult:
    return SweepLevelResult(
        context_label="full4096", concurrency=concurrency, requests=10, errors=0,
        duration_sec=20.0, prompt_tokens_measured=2800, p50_ms=100.0, p95_ms=200.0, p99_ms=300.0,
        throughput_req_s=1.0, output_tok_s=10.0, preemptions_delta=0,
        kv_cache_usage_peak=0.5, meets_slo=sustainable, preemption_free=sustainable,
        sustainable=sustainable,
    )


def test_percentile_nearest_rank():
    data = [float(n) for n in range(1, 101)]  # 1..100
    assert _percentile(data, 50) == 50.0
    assert _percentile(data, 95) == 95.0
    assert _percentile(data, 99) == 99.0
    assert _percentile([], 95) == 0.0


def test_scan_metric_sum_and_max():
    text = (
        "# HELP vllm:num_preemptions_total ...\n"
        "# TYPE vllm:num_preemptions_total counter\n"
        'vllm:num_preemptions_total{model="family-7b"} 4.0\n'
        'vllm:num_preemptions_total{model="other"} 6.0\n'
        'vllm:gpu_cache_usage_perc{model="family-7b"} 0.42\n'
        'vllm:gpu_cache_usage_perc{model="other"} 0.88\n'
    )
    assert _scan_metric(text, "num_preemptions", "sum") == 10.0
    assert _scan_metric(text, "gpu_cache_usage", "max") == 0.88
    assert _scan_metric(text, "does_not_exist", "sum") == 0.0


def test_summarize_level_flags_slo_and_preemption():
    fast = [RequestSample(latency_ms=200.0, completion_tokens=20, prompt_tokens=100, ok=True)] * 10
    good = _summarize_level("short", 8, fast, 20.0, preempt_delta=0, kv_peak=0.3)
    assert good.sustainable and good.meets_slo and good.preemption_free

    slow = [RequestSample(latency_ms=5000.0, completion_tokens=20, prompt_tokens=100, ok=True)] * 10
    breached = _summarize_level("short", 16, slow, 20.0, preempt_delta=0, kv_peak=0.9)
    assert not breached.meets_slo and not breached.sustainable

    preempted = _summarize_level("short", 16, fast, 20.0, preempt_delta=3, kv_peak=0.99)
    assert preempted.meets_slo and not preempted.preemption_free and not preempted.sustainable

    # Negative delta is a load-balanced-metrics artifact, not a preemption: must not block.
    lb_artifact = _summarize_level("short", 4, fast, 20.0, preempt_delta=-199, kv_peak=0.0)
    assert lb_artifact.preemption_free and lb_artifact.sustainable


def test_summarize_level_counts_errors_as_unmet_slo():
    mixed = (
        [RequestSample(latency_ms=200.0, completion_tokens=20, prompt_tokens=100, ok=True)] * 8
        + [RequestSample(latency_ms=0.0, completion_tokens=0, prompt_tokens=0, ok=False, error="Timeout")] * 2
    )
    result = _summarize_level("short", 8, mixed, 20.0, preempt_delta=0, kv_peak=0.3)
    assert result.errors == 2
    assert not result.meets_slo and not result.sustainable


def test_sustainable_width_is_contiguous_from_smallest():
    levels = [_level(1, True), _level(2, True), _level(4, False), _level(8, True)]
    # 8 is sustainable on its own but 4 broke first -> honest width is 2, not 8.
    assert _sustainable_width(levels) == 2
    assert _sustainable_width([_level(1, False)]) == 0
    assert _sustainable_width([_level(1, True), _level(2, True)]) == 2


def test_max_slo_meeting_width_tolerates_transient_dip():
    # c=8 is an isolated transient break; steady-state holds to 24. Strict=4, tolerant=24.
    levels = [_level(c, ok) for c, ok in [(1, True), (2, True), (4, True),
                                          (8, False), (16, True), (24, True), (32, False)]]
    assert _sustainable_width(levels) == 4
    assert _max_slo_meeting_width(levels) == 24
    assert _max_slo_meeting_width([_level(1, False)]) == 0


def test_curve_point_reports_measured_tokens_and_knee():
    levels = [_level(c, ok) for c, ok in [(1, True), (2, True), (4, True), (8, False), (16, True)]]
    point = _curve_point(levels)
    assert point["measured_tokens"] == 2800   # from the _level helper
    assert point["sustainable_batch"] == 16   # transient-tolerant: the c=8 dip is ignored
    assert point["strict_batch"] == 4         # contiguous: stops at the c=8 dip


def test_build_prompt_unique_with_seed():
    a = _build_prompt(128, seed="aaaa1111")
    b = _build_prompt(128, seed="bbbb2222")
    assert a != b  # distinct seed => distinct prefix from token 0 (defeats prefix cache)
    assert a[:20] != b[:20]


def test_build_prompt_scales_with_target_tokens():
    short = _build_prompt(128)
    full = _build_prompt(3500)
    assert len(full) > len(short)
    assert "Summarize" in short
    # FULL should be in the right order of magnitude for a 4096-ctx test.
    assert len(full) > 3000 * 4 * 0.5


def test_diurnal_trace_is_ordered_and_nonempty():
    trace = _diurnal_trace(minutes=2.0, peak_rps=8.0, base_rps=2.0)
    assert trace == sorted(trace)
    assert len(trace) > 0
    assert all(0.0 <= offset <= 120.0 for offset in trace)


def test_freeze_writes_sorted_json_and_returns_sha256(tmp_path):
    payload = {"b": 2, "a": 1, "mode": "sweep"}
    out = tmp_path / "cap.json"
    digest = _freeze(payload, str(out))
    assert len(digest) == 64
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert reloaded == payload
    # Same content -> same hash (frozen, verifiable).
    assert _freeze(payload, str(out)) == digest


def test_slo_constant_is_three_seconds():
    assert INTERACTIVE_P95_SLO_MS == 3000.0
