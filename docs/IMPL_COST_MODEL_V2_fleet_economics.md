# [COST-MODEL-V2] Implementation Plan — Pooled Fleet Cost & Latency Model

**Plan ID:** IMPL_COST_MODEL_V2
**WP:** Cost-Model Correction — Pooled Fleet Economics (BUILD_PLAN §6 / Slice 3 follow-on)
**Spec refs:** BUILD_PLAN §6 (Cost & telemetry), §8 Slice 3 (cost economics), ADR-0001 (wire boundary / isolation)
**Gap refs:** Cost model presents a sparse per-family figure as the headline; no pooled-fleet model exists
**Date:** 2026-06-18
**Prerequisite:** Slice 2 (per-route `cost_usd` shipped) + Slice 3 (`telemetry/cost.py` cost model shipped) — both green; the sparse model and its sole caller (`evidence/__main__.py`) must exist to be reframed.
**Unblocks:** A defensible public cost claim (README + `python -m evidence` output) and any future live cost benchmark (gives it a model to validate against).

> **Plan-format note:** This app is a standalone proof under `apps/`, not part of the FirstLight Layer A–I framework stack. The IMPL_WP numbering and the Primitive-First (layer) Gate do not apply; the app uses BUILD_PLAN slice sequencing. The Behavior-First Gate *does* apply and is filled below. Layer-gate substitution is recorded under Primitive-First Gate.

---

## Behavior-First Gate

    behavior_gate:
      scope: "foundry"
      user_behavior: "The architect (Ed) publishes a self-hosted inference cost figure, and a hiring reviewer interrogates it. This protects that exchange: the figure must survive 'what about pooling / peak concurrency?' instead of collapsing."
      waste_reduced: "Motivated-dismissal waste — an indefensible single number ($118k/day, silently a no-pooling upper bound) lets a reviewer discard the whole cost claim. Also rework waste: re-deriving the cost story per traffic assumption by hand."
      deterministic_boundary: "The entire model is deterministic arithmetic over stated traffic + serving assumptions (Little's law at peak, replica integration over a diurnal shape, GPU $/sec). Nothing is inferred by a model. What is NOT deterministic and stays out: the sustainable batch width and real P95 under co-tenancy, which need live measurement."
      agentic_role: "none"
      promotion_risk: "foundry_only"
      capabilities:
        - fleet_cost_projection
        - cold_start_exposure_estimation
      canonical_owner_of:
        - fleet_cost_projection

Promotion class for any agent output: **none** (P0 — no agent output exists in this WP; it is deterministic arithmetic).
Cross-scope gate: **N/A** (foundry_only; the model produces a published claim artifact, not runtime behavior).

---

## Primitive-First Gate (layer-gate substitution)

The five-question Layer A–I gate is **N/A** — this app is not part of the FirstLight stack and declares no layer. Substituted check against the app's own one-way dependency direction (README: pure cost arithmetic must not import serving/runtime up the chain):

1. **Module:** `telemetry/cost.py` — the reporting/telemetry boundary. ✔ unchanged location.
2. **Imports:** new code imports only `dataclasses`, `math`, and the existing `Route` enum already imported. It adds **no** new dependency on `serving/`, `gateway/`, or any runtime module. ✔ no new cross-module edge.
3. **Pure / side-effect-free:** the model is pure functions over frozen dataclasses — no I/O, no clock, no network. ✔ (matches the repo's "no hidden magic numbers; stated assumptions" discipline).
4. **Single-writer:** `telemetry/cost.py` remains the sole writer of cost figures. The new functions are read-only consumers of their inputs. ✔
5. **Non-breaking:** `project_fleet_cost`, `cost_for`, `SelfHostedCostConfig` keep their current signatures; the sole external caller (`evidence/__main__.py:124`) keeps working. ✔

---

## Context

`telemetry/cost.py` reports two honest self-hosted components (active serving; amortized boot/idle at a stated sparse-traffic assumption). That per-request split is sound. The defect is the **fleet roll-up**: `project_fleet_cost(per_request, requests_per_family_per_day, families)` multiplies a single per-request cost by families. At `($0.029, 4, 1_000_000)` it yields the **~$118k/day** figure that the README (line 55) and the `evidence` harness present as the headline. That figure silently assumes every family is an isolated tenant that boots its own GPU, serves ~3 requests, idles, and scales to zero — i.e. **no pooling across families**. It treats `scaledown_window` and `requests_per_window` as independent knobs when both are consequences of one arrival process.

ADR-0001 establishes that the privacy boundary sits at **context assembly and KV/prefix cache**, not the container: the base model is shared, only per-request context and its cache are private (the prefix-cache section mandates per-tenant cache salting). Therefore **multi-family co-tenancy on a warm GPU is architecturally permitted** under sequence-level KV isolation + salted cache. That moves pooling from "open question" to "supported premise," and makes a pooled-fleet model the correct one.

This WP adds that model **alongside** the sparse one (it does not delete the sparse figure — that is the honest no-pooling upper bound, kept as the contrast case, mirroring the demo-grade "ship both verdicts" stance). The model is **peak-concurrency-driven** (replicas sized by Little's law at the diurnal peak, with an always-warm baseline floor — at fleet scale there is no scale-to-zero), couples **cost and latency** (a first-order cold-start-exposure estimate so the idle-window optimum is visible, not "smallest idle"), and emits a **band parameterized by traffic**, because the result is wildly traffic-sensitive: illustratively, 1M families at 4 req/family/day project to ~\$40–100/day of warm GPU; at 40 req/family/day, ~\$250–600/day. Both are 2–3 orders of magnitude below \$118k. **No single corrected headline number is produced** until live batch-width and P95 measurement (deferred); the deliverable is a defensible model + band + stated assumptions.

---

## Resolved Decisions

### D1: Retain the sparse model, relabel it as the upper bound — do not delete it

`project_fleet_cost` keeps its signature and behavior (non-breaking for `evidence/__main__.py`). Only its docstring changes, to name it the **sparse isolated (no-pooling) upper bound**. Rationale: it is a *real* bound (the cost if per-family container isolation were forced), and shipping it next to the pooled figure is the honest framing a reviewer rewards. Rejected alternative — deleting/replacing it — would erase the contrast that makes the correction legible and would break the caller.

### D2: The fleet model is peak-concurrency-driven, not average-rate

Required replicas are sized at the **diurnal peak** via Little's law (`L = λ·W`), not by daily-average rate. Rationale: a 1M-family fleet always has baseline traffic (never scales to zero), and provisioning to average under-serves the peak. "A small number of replicas" is average-rate intuition and is rejected as the basis for the figure. The model carries an explicit `diurnal_peak_factor` input.

### D3: Co-tenancy is a typed gated precondition, not a prose caveat

`FleetTrafficModel.cotenancy_permitted: bool`. When `True` (justified by ADR-0001 isolation: shared base model, per-request private context, salted KV/prefix cache, sequence-level isolation), the effective batch width is `max_concurrent_per_replica`. When `False` (strict per-family isolation), effective batch collapses to `1` and the pooled figure degrades toward the sparse upper bound. Rationale: the cheapest configs are exactly the ones that most need the privacy argument — encoding it as a flag makes the coupling explicit and testable rather than a footnote. This is the privacy-↔-cost coupling made structural.

### D4: One model emits both cost and a first-order latency (cold-start exposure) figure

`FleetCostLatency` carries `cold_start_exposure_fraction` and a qualitative `p95_inflation_note`. Rationale: cost-only optimization drives toward "smallest idle window," which maximizes cold starts and destroys P95 against a ~90s boot. Coupling them lets the model show the *idle-window optimum*. **Explicit limit:** this is a first-order analytic estimate (scale-up events × lag/boot vs ramp), **not** a measured P95. Real P95 needs the deferred load test (Out of Scope).

### D5: No live GPU spend — pure analytical model + tests this WP

Matches "model-first, not GPU-first" and the BUILD_PLAN budget posture ("don't run GPU experiments until the traffic model is set"). The model is the thing a future benchmark validates against. Rejected alternative — run a Modal batch sweep now — is premature: without the model there is nothing to compare a measurement to, and it spends the GPU budget on an unframed question.

### D6: The model is the single source of truth — the band is emitted as a band, no hand-typed headline

Numbers in the README and `evidence` output are produced by calling the model, not transcribed. The model returns `(low, mid, high)` and the docs render what it returns. Rationale (Automation Efficiency): a hand-typed corrected number would be a second drifting truth and would re-commit the original sin (an un-interrogable figure). **Lower-waste alternative considered:** replace line 55 with a one-sentence prose hedge ("$118k is a no-pooling upper bound; pooled is likely 2–3 OOM lower, pending measurement") — rejected because a prose hedge is just as un-interrogable as the number it replaces and undercuts the repo's measurable/reproducible brand; the executable model costs ~half a day more and is the higher-integrity artifact. **Waste avoided:** per-traffic-assumption re-derivation by hand; a published number that cannot be re-checked.

---

## New types and their schema/code locations

All in `telemetry/cost.py` (appended; existing functions unchanged). Two named
models + a scenario renderer. `warm_container_hours` is the primary physical
driver; cost is derived from it (`container_hours × gpu_usd_per_hour`).

```python
@dataclass(frozen=True)
class FleetServingConfig:
    """Serving/billing constants for the pooled fleet. Defaults model Modal L4."""
    gpu_usd_per_hour: float = GPU_USD_PER_HOUR
    cold_boot_billed_sec: float = COLD_BOOT_BILLED_SEC
    scaledown_window_sec: float = IDLE_SCALEDOWN_SEC
    autoscaler_lag_sec: float = 30.0    # demand-rise -> new warm replica


@dataclass(frozen=True)
class SparseIsolatedCostModel:
    """No-pooling UPPER BOUND. Each family is an isolated tenant that boots its
    own GPU, serves a few requests in one warm window, then scales to zero.
    Bounds fleet cost from above; the honest contrast to the pooled model (D1).
    This is the math behind the legacy `project_fleet_cost`."""
    per_request_usd: float              # headline per-request cost (active + amortized boot/idle)
    requests_per_family_per_day: float
    families: int

    def project(self) -> dict[str, float]:
        """-> cost_per_family_day_usd, projected_fleet_day_usd, families, requests_per_family_per_day."""


@dataclass(frozen=True)
class PooledDiurnalFleetCostModel:
    """Pooled, peak-concurrency-driven fleet. Families share warm GPU replicas
    (co-tenancy permitted under ADR-0001 isolation: shared base model, per-request
    private context, salted KV/prefix cache, sequence-level isolation). Replicas
    are sized at the diurnal peak (Little's law) with an always-warm baseline
    floor — no scale-to-zero at fleet scale. MODELED, NOT MEASURED."""
    families: int
    requests_per_family_per_day: float
    diurnal_peak_factor: float          # peak instantaneous rate / daily-average rate
    warm_service_sec: float             # measured warm latency == service time per request
    max_concurrent_per_replica: int     # vLLM continuous-batch width sustainable at acceptable P95
    cotenancy_permitted: bool           # True only under ADR-0001 isolation; False -> batch collapses to 1
    config: FleetServingConfig = FleetServingConfig()

    def project(self) -> "FleetCostLatency": ...


@dataclass(frozen=True)
class FleetCostLatency:
    """Pooled-fleet projection. Warm container-hours and cost are a band
    (low/mid/high); latency is a first-order cold-start-exposure estimate,
    NOT a measured P95."""
    peak_concurrent_requests: float
    replicas_at_peak: int
    baseline_replicas: int                    # always-warm floor (no scale-to-zero)
    avg_replicas: float
    warm_container_hours_per_day_low: float   # demand-integral floor (perfect right-size)
    warm_container_hours_per_day_mid: float
    warm_container_hours_per_day_high: float  # peak-provisioned always-on
    fleet_cost_per_day_usd_low: float         # = container_hours_low * gpu_usd_per_hour
    fleet_cost_per_day_usd_mid: float
    fleet_cost_per_day_usd_high: float
    cold_start_exposure_fraction: float       # share of requests that hit a cold boot
    p95_inflation_note: str                   # qualitative cold-exposure x boot-sec impact


@dataclass(frozen=True)
class FleetScenario:
    """One scenario-table row: a label + the pooled-model inputs."""
    label: str
    model: PooledDiurnalFleetCostModel


def scenario_table(
    scenarios: list[FleetScenario], sparse: SparseIsolatedCostModel
) -> list[dict]:
    """Render each scenario through the pooled model next to the sparse upper
    bound. Each row: label, sparse_fleet_usd_day (upper bound), pooled band
    (low/mid/high usd/day), warm_container_hours band, cold_start_fraction."""
```

---

## Implementation

### Task 1 — Relabel the sparse roll-up as the explicit upper bound

**Pre-task reads:**
- `telemetry/cost.py` §`project_fleet_cost` (lines ~106–117) — confirm signature and return keys before editing the docstring only.
- `evidence/__main__.py` §line 124 — confirm it is the sole caller and relies on the current return keys.

Change only the `project_fleet_cost` docstring to state it is the **sparse isolated (no-pooling) upper bound**: it assumes each family boots its own GPU and scales to zero, so it bounds cost from above and is retained as the honest contrast to the pooled model (D1). Do not change its signature, math, or return keys.

**Verified by:** `python -m pytest -q` — existing 25 tests still pass (no behavior change).

---

### Task 2 — Add `SparseIsolatedCostModel`, `PooledDiurnalFleetCostModel`, and `scenario_table`

**Pre-task reads:**
- `telemetry/cost.py` §module header + §price constants (lines 1–37) — reuse `GPU_USD_PER_HOUR`, `GPU_USD_PER_SEC`, `COLD_BOOT_BILLED_SEC`, `IDLE_SCALEDOWN_SEC`; do not redefine.

Add the types above plus the private helpers below. Keep every function/method ≤30 lines (decomposed); `PooledDiurnalFleetCostModel.project()` orchestrates the helpers:

- `SparseIsolatedCostModel.project()` — `per_family = per_request_usd * requests_per_family_per_day`; return the four-key dict (same numbers the legacy `project_fleet_cost` produces — this is the upper bound).
- `_peak_concurrency(model) -> float` — `avg_rate = families*req_per_day/86400`; `peak_rate = avg_rate*diurnal_peak_factor`; return `peak_rate * warm_service_sec` (Little's law).
- `_effective_batch(model) -> int` — `max_concurrent_per_replica if cotenancy_permitted else 1` (D3).
- `_replica_band(model) -> tuple[int, float, int]` — `(baseline_replicas, avg_replicas, replicas_at_peak)` via `ceil(concurrency / _effective_batch)` at average and peak; `baseline = max(1, ...)`.
- `_cold_start_exposure(model) -> float` — first-order: scale-up events/day × `(autoscaler_lag_sec + cold_boot_billed_sec)` × marginal peak rate ÷ total requests/day; clamp `[0, 1]` (D4).
- `PooledDiurnalFleetCostModel.project() -> FleetCostLatency` — warm container-hours band first (`avg_replicas*24` → low, baseline+diurnal → mid, `replicas_at_peak*24` → high), then cost = `container_hours * config.gpu_usd_per_hour` for each, plus exposure fraction + `p95_inflation_note`.
- `scenario_table(scenarios, sparse) -> list[dict]` — one row per scenario: label, sparse upper-bound `$/day`, pooled band `$/day`, warm container-hours band, cold-start fraction.

**Verified by:** `python -c "from telemetry.cost import *; print(PooledDiurnalFleetCostModel(1_000_000,4,3,0.93,32,True).project())"` — prints a band where `_high >= _mid >= _low` (cost and container-hours), all far below `$118k`, `replicas_at_peak >= baseline_replicas`.

---

### Task 3 — Add `tests/test_cost.py` (cost model currently has zero tests)

**Pre-task reads:**
- `tests/test_router.py` — match the existing test style/imports (pytest, no fixtures framework beyond stdlib) so the new file is consistent.

New file. Cover the pre-existing functions (regression-lock them — they were untested) and both models:
1. `cost_for(self_hosted, ...)` returns the active + amortized split; hosted routes return token-price; cache/fallback return 0.
2. `SparseIsolatedCostModel(0.029, 4, 1_000_000).project()["projected_fleet_day_usd"]` ≈ `$116k`, and equals the legacy `project_fleet_cost(0.029, 4, 1_000_000)` (locks the upper bound + non-breakage).
3. **Regime ordering:** `PooledDiurnalFleetCostModel(...).project().fleet_cost_per_day_usd_high` < the sparse upper-bound figure for the same traffic (pooled ≤ sparse — the core claim).
4. **Band ordering:** `_low <= _mid <= _high` for both cost and warm-container-hours.
5. **Co-tenancy gate (D3):** `cotenancy_permitted=False` yields strictly higher cost + container-hours than `=True` (batch collapses to 1).
6. **Peak-driven (D2):** raising `diurnal_peak_factor` raises `replicas_at_peak` and `_high`, leaving `_low` (demand integral) ~unchanged.
7. **Density sensitivity:** raising `requests_per_family_per_day` raises all band figures monotonically.
8. **Exposure bounds:** `0 <= cold_start_exposure_fraction <= 1`.
9. **Scenario table:** `scenario_table([...], sparse)` returns one row per scenario with the expected keys; pooled `$/day` < sparse `$/day` in every row.

**Verified by:** `python -m pytest -q tests/test_cost.py` — all new tests pass; `python -m pytest -q` — full suite green (25 + new).

---

### Task 4 — Surface both regimes in the reviewer harness

**Pre-task reads:**
- `evidence/__main__.py` §lines 32, 104–124 — the cost imports, `_live_cost_config`, and the `project_fleet_cost` call site to extend.

Replace the bare `project_fleet_cost(...)` print with `scenario_table([...], SparseIsolatedCostModel(...))` rendered as a few labeled lines: the sparse upper bound, then the pooled band (cost + warm container-hours + cold-start fraction) for 2–3 scenarios (e.g. 4 and 40 req/family/day), with the stated assumptions inline (batch width, diurnal factor, co-tenancy=ADR-0001). Output stays offline-deterministic in SIMULATED mode (the models are pure), so the determinism hash is unaffected.

**Verified by:** `python -m evidence` — exit 0; output shows the sparse upper bound AND the pooled scenario table with assumptions; a second run yields the same determinism hash.

---

### Task 5 — Correct the public cost claim (README + strategy doc)

**Pre-task reads:**
- `README.md` §line 55 (the results table row) — the headline `~$118k / 1M families/day`.
- `cost-optimization-strategy.md` §1 (the `scaledown_window` 4× table, lines ~27–60) — the table that holds `requests/window=3` fixed while shrinking the window.

README: relabel the `$118k` row as **"sparse isolated upper bound (no pooling)"** and add a row for the pooled fleet band (value rendered from `fleet_cost_latency`, per D6), with a one-line regime note and pointer to this plan. `cost-optimization-strategy.md`: add a short caveat above the §1 table noting it holds request density fixed while varying the window (the independence error) and that fleet economics are governed by the pooled model, not this table; point to this plan. Do not hand-type the pooled number — cite the harness output.

**Verified by:** `git diff README.md cost-optimization-strategy.md` — the $118k figure is explicitly labeled an upper bound; the pooled band and regime distinction are present; no un-sourced corrected headline number was introduced.

---

## File Touch List

| File | Change |
|---|---|
| `telemetry/cost.py` | Relabel `project_fleet_cost` docstring (sparse upper bound, non-breaking); add `SparseIsolatedCostModel`, `PooledDiurnalFleetCostModel`, `FleetServingConfig`, `FleetCostLatency`, `FleetScenario`, `scenario_table()` + private helpers (peak concurrency, replica band, cold-start exposure, warm container-hours) |
| `tests/test_cost.py` | **New.** Regression-lock the previously untested `cost_for`/`project_fleet_cost`; test the fleet model (regime ordering, band ordering, co-tenancy gate, peak-driven replicas, density sensitivity, exposure bounds) |
| `evidence/__main__.py` | Print the pooled fleet band with stated assumptions alongside the relabeled sparse upper bound |
| `README.md` | Relabel line 55 as the no-pooling upper bound; add pooled-fleet band row + regime note (value from the model) |
| `cost-optimization-strategy.md` | Caveat above the §1 table (fixed-density independence error); pointer to the pooled model |

**Not touched:**
- `gateway/engine.py` — calls `cost_for` per-request only; the fleet model is reporting-only, not on the routing path. No runtime change.
- `gateway/router.py` — routing thresholds are out of scope; this WP is cost *modeling*, not routing.
- `serving/deploy_modal.py`, `serving/vllm_config.md` — no serving/GPU config change (no live benchmark this WP).
- `HANDOVER_cost_perf_optimization.md` — historical handover; left as the record of how this WP was scoped.

---

## Out of Scope

- **Live GPU batch-width / P95 measurement** — the model's two load-bearing unmeasured inputs (`max_concurrent_per_replica` sustainable at acceptable P95, and real cold-start exposure under the Modal autoscaler) need a Modal run. Deferred to a future "live cost benchmark" slice; this WP gives that benchmark a model to validate. **This is the weakest assumption in the plan:** the corrected band is defensible arithmetic over assumptions, not a measured figure — which is why D6 forbids publishing a single headline number.
- **A single corrected headline cost figure** — intentionally not produced; the output is a band parameterized by traffic (the figure swings 2–3 OOM with req/family/day and batch width).
- **Modal memory snapshots / AWQ quantization / keep-warm tuning** — cold-start *reduction* levers (BUILD_PLAN Slice 3); they change the model's inputs but are separate work.
- **Routing-threshold or degradation-ladder changes** — `router.py` is untouched.
- **Historical-policy replay** — already out of scope per ADR-0001; unrelated here.

---

## Amendment — reviewer corrections (post-merge of `b8c37d3`)

A design review of the shipped model required corrections; applied as a follow-up delta. The schema block above reflects the original build; the code now differs as follows.

- **(A) Batch width is a ceiling, not sustainable capacity.** `max_concurrent_per_replica` is relabeled the *configured* `--max-num-seqs` ceiling (sustainable width on a 24 GB L4 with a 7B + 4096-ctx KV is ~16–21, UNMEASURED). The evidence harness now sweeps a 5-row matrix `(label, density, batch)` — sparse pilot (4,1) / low pooled (4,8) / moderate (12,16) / high engagement (40,16) / optimistic capacity (40,32) — so `32` appears only in the optimistic row. New test `test_lower_batch_width_raises_cost` locks lower batch → higher cost.
- **(B) Dead idle lever removed.** `FleetServingConfig.scaledown_window_sec` was declared but never read (the model assumes an always-warm baseline, so idle window only affects the trough cost of peak-following replicas — which needs the diurnal-trough integral, deferred to the live slice). Field deleted; the latency note no longer claims an idle-window effect it does not compute.
- **(C) Cold-start field relabeled with its premise.** `cold_start_exposure_fraction` → `rampup_cold_start_fraction`; it counts only diurnal ramp-up boots under the always-warm-baseline premise, **not** total cold-start probability (so it stays ~flat with density by construction — that is correct, not a bug). Helper `_cold_start_exposure` → `_rampup_cold_start_fraction`.
- **(D) Latency surfaced honestly.** Added `expected_latency_ms` (warm + ramp exposure × boot; a two-point expectation) and `tail_latency_regime` (`warm-dominated` < 5% exposure else `cold-exposed`). `p95_inflation_note` → `latency_note`. No fake P95 is produced — a real P95 needs the warm distribution under batch queueing (live measurement). New test `test_expected_latency_between_warm_and_cold`.
- **(E) README band drift fixed.** The README pooled band now states its **LIVE 0.93 s warm-service basis** and notes that `python -m evidence` (SIMULATED) prints a lower band from its faster simulated warm latency — closing the D6 "docs render what the model returns" gap.

`scenario_table` row keys changed accordingly: added `effective_batch`, `expected_latency_ms`; `cold_start_fraction` → `rampup_cold_start_fraction`. Per-request `cost_for` / `SelfHostedCostConfig` / `project_fleet_cost` signatures are unchanged (still non-breaking for the gateway runtime).
