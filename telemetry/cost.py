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
COLD_BOOT_BILLED_SEC = 30.0       # GPU-seconds billed for container boot + model load
IDLE_SCALEDOWN_SEC = 120.0        # idle GPU-seconds before scale-to-zero (matches deploy)
REQUESTS_PER_WARM_WINDOW = 3.0    # sparse-traffic assumption: requests served per warm-up


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
    """Roll a per-request cost up to per-family/day and per-fleet/day."""
    per_family_day = cost_per_request_usd * requests_per_family_per_day
    return {
        "cost_per_family_day_usd": round(per_family_day, 4),
        "projected_fleet_day_usd": round(per_family_day * families, 2),
        "families": float(families),
        "requests_per_family_per_day": requests_per_family_per_day,
    }
