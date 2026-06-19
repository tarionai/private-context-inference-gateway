from gateway.contract import Route
from serving.clients import CompletionResult
from telemetry.cost import (
    CONFIGURED_BATCH_CEILING,
    MEASURED_BATCH_BY_CONTEXT,
    MEASURED_BATCH_FULL_CTX,
    MEASURED_BATCH_SHORT_CTX,
    MEASURED_BATCH_TINY_CTX,
    MEASURED_COLD_BOOT_SEC,
    FleetScenario,
    FleetServingConfig,
    PooledDiurnalFleetCostModel,
    SparseIsolatedCostModel,
    cost_for,
    project_fleet_cost,
    scenario_table,
)

WARM = CompletionResult(
    text="ok",
    model_used="family-7b",
    prompt_tokens=100,
    completion_tokens=50,
    latency_ms=930.0,
    cold_start=False,
)


def _pooled(req_per_day=4, peak_factor=3, cotenancy=True, batch=32):
    return PooledDiurnalFleetCostModel(
        families=1_000_000,
        requests_per_family_per_day=req_per_day,
        diurnal_peak_factor=peak_factor,
        warm_service_sec=0.93,
        max_concurrent_per_replica=batch,
        cotenancy_permitted=cotenancy,
    )


# --- Regression lock on the previously untested per-request rollups -----------


def test_cost_for_self_hosted_splits_active_and_amortized():
    breakdown = cost_for(Route.self_hosted, WARM)
    assert breakdown.active_serving_usd > 0
    assert breakdown.amortized_boot_idle_usd > 0
    assert breakdown.total_usd == round(
        breakdown.active_serving_usd + breakdown.amortized_boot_idle_usd, 6
    )


def test_cost_for_hosted_uses_token_price_without_amortization():
    breakdown = cost_for(Route.hosted_fast, WARM)
    assert breakdown.active_serving_usd == round(
        (100 * 1.00 + 50 * 5.00) / 1_000_000, 6
    )
    assert breakdown.amortized_boot_idle_usd == 0.0


def test_cost_for_cache_and_fallback_are_free():
    for route in (Route.cache, Route.deterministic_fallback):
        assert cost_for(route, WARM).total_usd == 0.0


def test_sparse_model_matches_legacy_upper_bound():
    projection = SparseIsolatedCostModel(0.029, 4, 1_000_000).project()
    assert projection == project_fleet_cost(0.029, 4, 1_000_000)
    assert projection["projected_fleet_day_usd"] == 116_000.0


# --- Pooled fleet model -------------------------------------------------------


def test_pooled_high_stays_below_sparse_upper_bound():
    sparse = SparseIsolatedCostModel(0.029, 4, 1_000_000).project()
    pooled = _pooled().project()
    assert pooled.fleet_cost_per_day_usd_high < sparse["projected_fleet_day_usd"]


def test_band_is_ordered_for_cost_and_container_hours():
    result = _pooled().project()
    assert (
        result.fleet_cost_per_day_usd_low
        <= result.fleet_cost_per_day_usd_mid
        <= result.fleet_cost_per_day_usd_high
    )
    assert (
        result.warm_container_hours_per_day_low
        <= result.warm_container_hours_per_day_mid
        <= result.warm_container_hours_per_day_high
    )


def test_cotenancy_gate_raises_cost_when_disabled():
    on = _pooled(cotenancy=True).project()
    off = _pooled(cotenancy=False).project()
    assert off.fleet_cost_per_day_usd_high > on.fleet_cost_per_day_usd_high
    assert off.warm_container_hours_per_day_high > on.warm_container_hours_per_day_high


def test_peak_factor_drives_replicas_and_high_but_not_low():
    base = _pooled(peak_factor=2).project()
    peaky = _pooled(peak_factor=6).project()
    assert peaky.replicas_at_peak > base.replicas_at_peak
    assert peaky.fleet_cost_per_day_usd_high > base.fleet_cost_per_day_usd_high
    assert peaky.fleet_cost_per_day_usd_low == base.fleet_cost_per_day_usd_low


def test_higher_request_density_raises_every_band_figure():
    low = _pooled(req_per_day=4).project()
    high = _pooled(req_per_day=40).project()
    assert high.fleet_cost_per_day_usd_low > low.fleet_cost_per_day_usd_low
    assert high.fleet_cost_per_day_usd_mid > low.fleet_cost_per_day_usd_mid
    assert high.fleet_cost_per_day_usd_high > low.fleet_cost_per_day_usd_high


def test_rampup_cold_start_is_a_fraction():
    assert 0.0 <= _pooled().project().rampup_cold_start_fraction <= 1.0
    stress = PooledDiurnalFleetCostModel(1_000, 1, 50, 5.0, 1, False).project()
    assert 0.0 <= stress.rampup_cold_start_fraction <= 1.0


def test_lower_batch_width_raises_cost():
    """32 is the configured ceiling; a lower sustainable batch costs strictly more."""
    wide = _pooled(batch=32).project()
    narrow = _pooled(batch=8).project()
    assert narrow.fleet_cost_per_day_usd_high > wide.fleet_cost_per_day_usd_high
    assert narrow.warm_container_hours_per_day_high > wide.warm_container_hours_per_day_high


def test_measured_inputs_locked_from_live_capture():
    """Locked from state/loadtest_*_capture.json (2026-06-18)."""
    assert MEASURED_BATCH_TINY_CTX == 16    # ~184-tok, optimistic SLO-edge
    assert MEASURED_BATCH_SHORT_CTX == 4    # ~0.5-1k-tok realistic context
    assert MEASURED_BATCH_FULL_CTX == 1     # ~3.4k-tok context, prefill-compute-bound
    assert CONFIGURED_BATCH_CEILING == 32   # --max-num-seqs, not interactively sustainable
    assert MEASURED_COLD_BOOT_SEC == 90.0   # measured 77-92s, replaces assumed 30s
    # The fleet model uses the MEASURED cold boot, not the old 30s assumption.
    assert FleetServingConfig().cold_boot_billed_sec == MEASURED_COLD_BOOT_SEC


def test_measured_batch_curve_is_monotonic_non_increasing():
    """Sustainable batch falls (never rises) as context grows — the measured cost curve."""
    tokens = [t for t, _ in MEASURED_BATCH_BY_CONTEXT]
    batches = [b for _, b in MEASURED_BATCH_BY_CONTEXT]
    assert tokens == sorted(tokens)
    assert all(earlier >= later for earlier, later in zip(batches, batches[1:]))
    assert batches[0] == MEASURED_BATCH_TINY_CTX and batches[-1] == MEASURED_BATCH_FULL_CTX


def test_smaller_batch_along_the_curve_costs_more():
    """Walking the curve toward larger context (smaller batch) raises cost monotonically."""
    highs = [_pooled(batch=b).project().fleet_cost_per_day_usd_high
             for _, b in MEASURED_BATCH_BY_CONTEXT]
    # tiny-ctx (batch 16) is cheapest; full-ctx (batch 1) is dearest, by a wide margin.
    assert highs[0] == min(highs) and highs[-1] == max(highs)
    assert highs[-1] >= highs[0] * 5


def test_expected_latency_between_warm_and_cold():
    model = _pooled()
    result = model.project()
    warm_ms = model.warm_service_sec * 1000
    cold_ms = warm_ms + model.config.cold_boot_billed_sec * 1000
    assert warm_ms <= result.expected_latency_ms <= cold_ms
    assert result.tail_latency_regime in {"warm-dominated", "cold-exposed"}


def test_scenario_table_keys_and_pooled_below_sparse_per_row():
    sparse = SparseIsolatedCostModel(0.029, 4, 1_000_000)
    scenarios = [
        FleetScenario("4 req/family/day", _pooled(req_per_day=4)),
        FleetScenario("40 req/family/day", _pooled(req_per_day=40)),
    ]
    rows = scenario_table(scenarios, sparse)
    assert len(rows) == 2
    expected_keys = {
        "label",
        "effective_batch",
        "sparse_fleet_usd_day",
        "pooled_usd_day_low",
        "pooled_usd_day_mid",
        "pooled_usd_day_high",
        "warm_container_hours_low",
        "warm_container_hours_mid",
        "warm_container_hours_high",
        "rampup_cold_start_fraction",
        "expected_latency_ms",
    }
    for row in rows:
        assert set(row.keys()) == expected_keys
        assert row["pooled_usd_day_high"] < row["sparse_fleet_usd_day"]
