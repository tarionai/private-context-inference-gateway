"""Per-request cost and rollups.

The self-hosted cost is NOT `GPU $/hr ÷ tokens-served-while-warm` — that flatters
the figure by ignoring the dominant cost of scale-to-zero serving: billed
cold-start/boot GPU-seconds at sparse traffic. Two self-hosted components are
reported separately, mirroring the cold/warm latency split:

  active serving cost = GPU $/sec × active-generation seconds, per request;
  amortized boot/idle cost = (billed cold-start + idle GPU-sec) spread across
      requests at a STATED sparse-traffic assumption (requests per warm window).

Hosted cost is token-price. All price constants are stated assumptions to verify
at build time, not hidden magic numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from gateway.contract import Route
from serving.clients import CompletionResult

# --- Stated price assumptions (verify at build time) ---------------------------
# Modal L4 GPU. Pin against Modal's current published rate before publishing.
GPU_USD_PER_HOUR = 0.80
GPU_USD_PER_SEC = GPU_USD_PER_HOUR / 3600.0

# Claude pricing per 1M tokens (pinned 2026-06-18 via claude-api skill).
HOSTED_PRICES_USD_PER_MTOK = {
    Route.hosted_fast: {"in": 1.00, "out": 5.00},    # Haiku 4.5
    Route.hosted_strong: {"in": 3.00, "out": 15.00},  # Sonnet 4.6
}

# Scale-to-zero serving assumptions.
COLD_BOOT_BILLED_SEC = 30.0       # per-request OFFLINE placeholder; LIVE measures real boot (~90s, below)
IDLE_SCALEDOWN_SEC = 120.0        # idle GPU-seconds before scale-to-zero (matches deploy)
REQUESTS_PER_WARM_WINDOW = 3.0    # sparse-traffic assumption: requests served per warm-up

# --- MEASURED on Modal L4 (state/loadtest_*_capture.json, 2026-06-18) ---
# Sustainable interactive batch width (p95 < 3s SLO) FALLS with context length -- a
# measured CURVE, not a single value (serving/loadtest.py ctxcurve). Distinct per-
# request context (ADR-0001 salting defeats prefix sharing). The realistic operating
# point depends on the assembled-context size distribution (production data needed;
# the demo fixture assembles only ~15-77 tok). See serving/vllm_config.md.
MEASURED_BATCH_BY_CONTEXT = (  # (assembled_context_tokens, sustainable_batch)
    (184, 16), (276, 4), (529, 4), (1035, 4), (1937, 2), (3367, 1),
)
MEASURED_BATCH_TINY_CTX = 16      # ~184 tok, optimistic edge (sits right at the 3s SLO knee)
MEASURED_BATCH_SHORT_CTX = 4      # ~0.5-1k tok realistic context (c=8 breaks ~3.2s across 3 sizes)
MEASURED_BATCH_FULL_CTX = 1       # ~3.4k tok; prefill-compute-bound -> refutes the 16-21 hypothesis
CONFIGURED_BATCH_CEILING = 32     # --max-num-seqs; NOT interactively sustainable at any tested context
MEASURED_COLD_BOOT_SEC = 90.0     # 77-92s across runs; replaces the fleet model's assumed 30s


@dataclass(frozen=True)
class SelfHostedCostConfig:
    """Cost inputs for the self-hosted route. Defaults model Modal L4 at sparse
    traffic; LIVE runs pass REAL measured boot/active seconds and the local
    host rate so the figure reflects the actual serve, not an assumption."""

    host_usd_per_sec: float = GPU_USD_PER_SEC
    boot_billed_sec: float = COLD_BOOT_BILLED_SEC
    idle_sec: float = IDLE_SCALEDOWN_SEC
    requests_per_window: float = REQUESTS_PER_WARM_WINDOW


@dataclass(frozen=True)
class CostBreakdown:
    route: Route
    active_serving_usd: float       # warm-path GPU cost for this request
    amortized_boot_idle_usd: float  # boot+idle GPU cost spread at the traffic assumption
    total_usd: float

    @property
    def headline_usd(self) -> float:
        """The honest cost a reviewer probes: active + amortized boot/idle."""
        return self.active_serving_usd + self.amortized_boot_idle_usd


def _self_hosted_cost(
    result: CompletionResult, config: SelfHostedCostConfig
) -> CostBreakdown:
    active_seconds = result.latency_ms / 1000.0 if not result.cold_start else 0.0
    active = config.host_usd_per_sec * active_seconds
    boot_idle_total = config.host_usd_per_sec * (config.boot_billed_sec + config.idle_sec)
    amortized = boot_idle_total / config.requests_per_window
    return CostBreakdown(
        route=Route.self_hosted,
        active_serving_usd=round(active, 6),
        amortized_boot_idle_usd=round(amortized, 6),
        total_usd=round(active + amortized, 6),
    )


def _hosted_cost(route: Route, result: CompletionResult) -> CostBreakdown:
    prices = HOSTED_PRICES_USD_PER_MTOK[route]
    cost = (
        result.prompt_tokens * prices["in"] + result.completion_tokens * prices["out"]
    ) / 1_000_000.0
    return CostBreakdown(
        route=route,
        active_serving_usd=round(cost, 6),
        amortized_boot_idle_usd=0.0,
        total_usd=round(cost, 6),
    )


def cost_for(
    route: Route,
    result: CompletionResult,
    self_hosted_config: SelfHostedCostConfig | None = None,
) -> CostBreakdown:
    if route == Route.self_hosted:
        return _self_hosted_cost(result, self_hosted_config or SelfHostedCostConfig())
    if route in HOSTED_PRICES_USD_PER_MTOK:
        return _hosted_cost(route, result)
    # cache + deterministic_fallback: no model inference, no GPU.
    return CostBreakdown(route=route, active_serving_usd=0.0, amortized_boot_idle_usd=0.0, total_usd=0.0)


def project_fleet_cost(
    cost_per_request_usd: float, requests_per_family_per_day: float, families: int
) -> dict[str, float]:
    """Sparse isolated (no-pooling) UPPER BOUND on fleet cost.

    Multiplies one per-request cost by every family. This silently assumes each
    family is an isolated tenant that boots its own GPU, serves a few requests in
    one warm window, then scales to zero — i.e. zero pooling across families. It
    therefore bounds fleet cost from ABOVE. Retained unchanged as the honest
    contrast to `PooledDiurnalFleetCostModel` (D1); a real pooled fleet shares
    warm replicas and costs 2-3 orders of magnitude less. See
    docs/IMPL_COST_MODEL_V2_fleet_economics.md.
    """
    per_family_day = cost_per_request_usd * requests_per_family_per_day
    return {
        "cost_per_family_day_usd": round(per_family_day, 4),
        "projected_fleet_day_usd": round(per_family_day * families, 2),
        "families": float(families),
        "requests_per_family_per_day": requests_per_family_per_day,
    }


# --- Fleet cost models (MODELED, NOT MEASURED) --------------------------------
# Two regimes side by side: the sparse no-pooling UPPER BOUND (above) and the
# pooled, peak-concurrency-driven fleet (below). The pooled figure rests on two
# UNMEASURED inputs — sustainable batch width on an L4 with a 7B+KV, and real
# cold-start exposure under Modal's autoscaler — so it is a defensible BAND, not
# a measured number. Live measurement is a later slice. See
# docs/IMPL_COST_MODEL_V2_fleet_economics.md.

_SECONDS_PER_DAY = 86_400.0
_HOURS_PER_DAY = 24.0


@dataclass(frozen=True)
class FleetServingConfig:
    """Serving/billing constants for the pooled fleet. Defaults model Modal L4.

    No idle/scaledown window field: this model assumes an always-warm baseline
    floor (no scale-to-zero at fleet scale), so the idle window changes only the
    trough cost of peak-following replicas — an effect that needs the diurnal-
    trough integral and a real arrival curve. That idle-window optimum is the
    live-benchmark slice, not this model (so a `scaledown_window` knob here would
    be dead)."""

    gpu_usd_per_hour: float = GPU_USD_PER_HOUR
    cold_boot_billed_sec: float = MEASURED_COLD_BOOT_SEC  # MEASURED ~90s (was assumed 30s)
    autoscaler_lag_sec: float = 30.0  # demand-rise -> new warm replica; ASSUMED (Tier A saw a
    # scale-up transient near c=8 short-ctx but did not isolate the lag) -- UNMEASURED


@dataclass(frozen=True)
class SparseIsolatedCostModel:
    """No-pooling UPPER BOUND. Each family is an isolated tenant that boots its
    own GPU, serves a few requests in one warm window, then scales to zero.
    Bounds fleet cost from above; the honest contrast to the pooled model (D1).
    This is the math behind the legacy `project_fleet_cost`."""

    per_request_usd: float  # headline per-request cost (active + amortized boot/idle)
    requests_per_family_per_day: float
    families: int

    def project(self) -> dict[str, float]:
        """-> cost_per_family_day_usd, projected_fleet_day_usd, families, requests_per_family_per_day."""
        return project_fleet_cost(
            self.per_request_usd, self.requests_per_family_per_day, self.families
        )


@dataclass(frozen=True)
class FleetCostLatency:
    """Pooled-fleet projection. Warm container-hours and cost are a band
    (low/mid/high); latency is a first-order estimate, NOT a measured P95."""

    peak_concurrent_requests: float
    replicas_at_peak: int
    baseline_replicas: int  # always-warm floor (no scale-to-zero)
    avg_replicas: float
    warm_container_hours_per_day_low: float  # demand-integral floor (perfect right-size)
    warm_container_hours_per_day_mid: float
    warm_container_hours_per_day_high: float  # peak-provisioned always-on
    fleet_cost_per_day_usd_low: float  # = container_hours_low * gpu_usd_per_hour
    fleet_cost_per_day_usd_mid: float
    fleet_cost_per_day_usd_high: float
    rampup_cold_start_fraction: float  # share hitting a DIURNAL RAMP-UP boot (NOT total cold-start prob)
    expected_latency_ms: float  # warm + rampup_exposure x boot; first-order, not a measured P95
    tail_latency_regime: str  # "warm-dominated" (<5% exposure) | "cold-exposed"
    latency_note: str  # what is and isn't modeled (scope-honest)


@dataclass(frozen=True)
class PooledDiurnalFleetCostModel:
    """Pooled, peak-concurrency-driven fleet. Families share warm GPU replicas
    (co-tenancy permitted under ADR-0001 isolation: shared base model, per-request
    private context, salted KV/prefix cache, sequence-level isolation). Replicas
    are sized at the diurnal peak (Little's law) with an always-warm baseline
    floor — no scale-to-zero at fleet scale. MODELED, NOT MEASURED."""

    families: int
    requests_per_family_per_day: float
    diurnal_peak_factor: float  # peak instantaneous rate / daily-average rate
    warm_service_sec: float  # measured warm latency == service time per request
    max_concurrent_per_replica: int  # sustainable batch width. MEASURED: 16 short-ctx / 1 full-4096-ctx
    # (state/loadtest_sweep_capture.json). 32 = configured ceiling, NOT interactively sustainable.
    cotenancy_permitted: bool  # True only under ADR-0001 isolation; False -> batch collapses to 1
    config: FleetServingConfig = FleetServingConfig()

    def project(self) -> FleetCostLatency:
        baseline, avg_replicas, replicas_at_peak = _replica_band(self)
        gpu_rate = self.config.gpu_usd_per_hour
        hours = (avg_replicas * _HOURS_PER_DAY, baseline * _HOURS_PER_DAY,
                 replicas_at_peak * _HOURS_PER_DAY)
        exposure = _rampup_cold_start_fraction(self)
        return FleetCostLatency(
            peak_concurrent_requests=_peak_concurrency(self),
            replicas_at_peak=replicas_at_peak,
            baseline_replicas=baseline,
            avg_replicas=avg_replicas,
            warm_container_hours_per_day_low=hours[0],
            warm_container_hours_per_day_mid=hours[1],
            warm_container_hours_per_day_high=hours[2],
            fleet_cost_per_day_usd_low=hours[0] * gpu_rate,
            fleet_cost_per_day_usd_mid=hours[1] * gpu_rate,
            fleet_cost_per_day_usd_high=hours[2] * gpu_rate,
            rampup_cold_start_fraction=exposure,
            expected_latency_ms=_expected_latency_ms(self, exposure),
            tail_latency_regime=_tail_latency_regime(exposure),
            latency_note=_latency_note(exposure, self.config.cold_boot_billed_sec),
        )


def _avg_rate(model: PooledDiurnalFleetCostModel) -> float:
    """Daily-average arrival rate (requests/sec) across the whole fleet."""
    return model.families * model.requests_per_family_per_day / _SECONDS_PER_DAY


def _peak_concurrency(model: PooledDiurnalFleetCostModel) -> float:
    """Little's law at the diurnal peak: L = (peak arrival rate) * service time."""
    peak_rate = _avg_rate(model) * model.diurnal_peak_factor
    return peak_rate * model.warm_service_sec


def _effective_batch(model: PooledDiurnalFleetCostModel) -> int:
    """Co-tenancy gate (D3): full batch only under ADR-0001 isolation, else 1."""
    return model.max_concurrent_per_replica if model.cotenancy_permitted else 1


def _replica_band(model: PooledDiurnalFleetCostModel) -> tuple[int, float, int]:
    """(baseline_replicas, avg_replicas, replicas_at_peak). Baseline is the
    whole-replica always-warm floor covering average demand; peak is sized to
    serve the diurnal peak concurrency. Ordering: avg <= baseline <= peak."""
    batch = _effective_batch(model)
    avg_concurrency = _avg_rate(model) * model.warm_service_sec
    peak_concurrency = avg_concurrency * model.diurnal_peak_factor
    avg_replicas = avg_concurrency / batch
    baseline = max(1, math.ceil(avg_replicas))
    replicas_at_peak = max(baseline, math.ceil(peak_concurrency / batch))
    return baseline, avg_replicas, replicas_at_peak


def _rampup_cold_start_fraction(model: PooledDiurnalFleetCostModel) -> float:
    """Share of requests hitting a DIURNAL RAMP-UP cold boot.

    Premise: an always-warm baseline floor (no scale-to-zero), so the ONLY
    cold-start source modeled is replicas added on the way up to peak. Each
    scale-up opens a (lag + boot) window in which requests at that replica's
    share of peak rate pay cold latency. It does NOT count idle-gap / overnight
    scale-to-zero cold starts (assumed away by the baseline floor) — so it stays
    roughly flat with density and is NOT a total cold-start probability. Clamped
    to [0, 1]; first-order estimate, not a measured P95."""
    baseline, _, replicas_at_peak = _replica_band(model)
    scale_ups = max(0, replicas_at_peak - baseline)
    total = model.families * model.requests_per_family_per_day
    if scale_ups == 0 or total <= 0:
        return 0.0
    window = model.config.autoscaler_lag_sec + model.config.cold_boot_billed_sec
    peak_rate = _avg_rate(model) * model.diurnal_peak_factor
    exposed = scale_ups * window * (peak_rate / replicas_at_peak)
    return max(0.0, min(1.0, exposed / total))


_TAIL_REGIME_THRESHOLD = 0.05  # exposure above this => cold-exposed tail


def _expected_latency_ms(model: PooledDiurnalFleetCostModel, exposure: float) -> float:
    """First-order expected latency: warm service plus ramp-up cold exposure.

    A mixture of two point masses (warm vs warm+boot) — an expectation, NOT a
    measured P95; the warm distribution under batch queueing needs measurement."""
    warm_ms = model.warm_service_sec * 1000.0
    boot_ms = model.config.cold_boot_billed_sec * 1000.0
    return round(warm_ms + exposure * boot_ms, 1)


def _tail_latency_regime(exposure: float) -> str:
    return "warm-dominated" if exposure < _TAIL_REGIME_THRESHOLD else "cold-exposed"


def _latency_note(exposure: float, boot_sec: float) -> str:
    """Scope-honest note: only diurnal ramp-up exposure is modeled."""
    return (
        f"~{exposure:.2%} of requests hit a ~{boot_sec:.0f}s ramp-up cold boot "
        f"(always-warm baseline assumed; idle-gap cold starts not modeled). The "
        f"idle-window optimum needs the diurnal-trough integral — live slice."
    )


@dataclass(frozen=True)
class FleetScenario:
    """One scenario-table row: a label + the pooled-model inputs."""

    label: str
    model: PooledDiurnalFleetCostModel


def scenario_table(
    scenarios: list[FleetScenario], sparse: SparseIsolatedCostModel
) -> list[dict]:
    """Render each scenario through the pooled model next to the sparse upper
    bound. Each row: label, effective_batch, sparse_fleet_usd_day (upper bound),
    pooled band (low/mid/high usd/day), warm container-hours band, ramp-up
    cold-start fraction, and first-order expected latency."""
    sparse_day = sparse.project()["projected_fleet_day_usd"]
    rows = []
    for scenario in scenarios:
        result = scenario.model.project()
        rows.append({
            "label": scenario.label,
            "effective_batch": _effective_batch(scenario.model),
            "sparse_fleet_usd_day": sparse_day,
            "pooled_usd_day_low": result.fleet_cost_per_day_usd_low,
            "pooled_usd_day_mid": result.fleet_cost_per_day_usd_mid,
            "pooled_usd_day_high": result.fleet_cost_per_day_usd_high,
            "warm_container_hours_low": result.warm_container_hours_per_day_low,
            "warm_container_hours_mid": result.warm_container_hours_per_day_mid,
            "warm_container_hours_high": result.warm_container_hours_per_day_high,
            "rampup_cold_start_fraction": result.rampup_cold_start_fraction,
            "expected_latency_ms": result.expected_latency_ms,
        })
    return rows
