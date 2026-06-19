# Cost Model V2 — Pooled Fleet Economics (Report)

**Status:** MODELED projection over **MEASURED** inputs · Phase 1 live benchmark **COMPLETE** (2026-06-18) · **Initial model commit:** `b8c37d3`
**Plan:** [`IMPL_COST_MODEL_V2_fleet_economics.md`](./IMPL_COST_MODEL_V2_fleet_economics.md) · **Boundary:** [`ADR-0001`](./ADR-0001-inference-gateway-boundary.md)
**Source of truth:** `telemetry/cost.py` — every figure below is emitted by the model and reproduced by `python -m evidence`. No number in this report is hand-typed.
**Measured inputs:** `serving/loadtest.py` (`sweep` batch-width, `ctxcurve` context-size sweep, `diurnal` replay); frozen captures `state/loadtest_sweep_capture.json` (sha `e5758cda…`), `state/loadtest_ctxcurve_capture.json` (sha `46c792cf…`), `state/loadtest_diurnal_capture.json` (sha `65fb994d…`); method in [`serving/vllm_config.md`](../serving/vllm_config.md).

---

## Executive summary

The published self-hosted headline — **~$113–116k/day for 1M families** — is *not wrong*, but it is a **no-pooling upper bound**, not the operating cost. It silently assumes each family is an isolated tenant that boots its own GPU, serves a few requests, and scales to zero — i.e. **zero pooling across families**.

> **The narrative in one paragraph.** The original $113–116k/day figure was a *valid sparse-isolated upper bound, not the operating estimate*. A traffic-aware pooled fleet model plus live Modal L4 measurements shows the realistic cost is **2–3 orders of magnitude lower for short-context interactive inference**, while **full-context workloads lose batching and move into the higher-cost cautionary regime**.

[ADR-0001](./ADR-0001-inference-gateway-boundary.md) places the privacy boundary at **context assembly and the salted KV/prefix cache**, not the container: the base model is shared, only per-request context and its salted cache are private. Multi-family co-tenancy on a warm GPU is therefore *architecturally permitted*. Modelling the fleet that way — replicas sized at the **diurnal peak** (Little's law) over an **always-warm baseline floor** — and feeding it the **live-measured sustainable batch width** gives:

| Regime | 1M families, 4 req/family/day | Basis |
|---|---|---|
| Sparse isolated (no pooling) — **upper bound** | **~$113–116k/day** | one GPU boot per family; legacy `project_fleet_cost` |
| Pooled, **realistic context** (~0.5–1k tok, batch **4**, MEASURED) | **~$207–634/day** | families share warm replicas; sustainable batch measured on L4 |
| Pooled, **full 4096 context** (batch **1**, MEASURED) | **~$827–2,496/day** | at full context a single L4 **cannot batch interactively**; concurrency lost |

**The headline finding: sustainable batch width is a measured *curve* that falls with context length** — not a single number. A live context-size sweep measured **16 at tiny (~184-tok) context → 4 at realistic (~0.5–1k tok) → 2 (~2k) → 1 at full 4096 context**, all at a p95 < 3 s SLO. The hypothesised "~16–21 at full context" is **refuted**: the binding limit at large context is **prefill compute + queueing, not KV exhaustion** (vLLM logs showed KV peaking at ~57 % with 12 requests queued). The deliverable is therefore a **cost curve over context size**, and the one remaining unknown is *where real family-digest traffic sits on it* (the assembled-context distribution — production data needed).

### Operating guide (decision table)

Read off the **actual assembled-context size** of a request and pick the regime. `$/day` is 1M families at 4 req/family/day, warm 0.93 s (scales ~linearly with density); see the full curve below.

| Actual assembled context | Operating recommendation | Pooled cost (1M fam, 4 req/day) |
|---|---|---:|
| **< 250 tok** | batch-16 possible — **cheapest regime** | ~$52–173/day |
| **250 – 1k tok** | batch-4 — **realistic interactive regime** | ~$207–634/day |
| **1 – 3k tok** | batch-2 — **caution regime** (latency tightens) | ~$413–1,248/day |
| **> 3k tok** | batch-1 — **avoid full-context inference unless necessary** | ~$827–2,496/day |

`python -m telemetry.context_dist` reads the gateway audit and maps real traffic's assembled-context p50/p90/p95/p99 onto these rows.

**Lever:** keep assembled context small (tighter privacy-filtered assembly, summarised history, retrieval over dump-everything) to stay in the cheap, high-batch regimes. Context size *is* the cost knob.

---

## The two regimes

**Sparse isolated cost (`SparseIsolatedCostModel`)** — retained unchanged as the honest contrast. It multiplies one per-request cost by every family: real *only if* per-family container isolation were forced. It bounds fleet cost from above. This is the math behind the original `~$118k` figure (now ~$113–116k with the measured ~85 s boot).

**Pooled diurnal fleet cost (`PooledDiurnalFleetCostModel`)** — the operating model. Families share warm GPU replicas; the number of replicas is driven by **peak concurrency**, not average rate:

- `peak concurrency = (families × req/day ÷ 86,400) × diurnal_peak_factor × warm_service_sec`  (Little's law, `L = λ·W`)
- `replicas_at_peak = ⌈peak concurrency ÷ effective_batch⌉`
- `effective_batch = max_concurrent_per_replica` **if co-tenancy permitted, else 1** (the privacy↔cost coupling, made structural — ADR-0001 is the gate)
- `baseline_replicas` = the always-warm floor (whole replicas covering average demand — at fleet scale there is **no scale-to-zero**)

**Warm container-hours is the primary physical driver; cost = container-hours × GPU $/hr.** The model emits a band (**low** = `avg_replicas × 24`, the demand integral; **mid** = `baseline_replicas × 24`, the always-warm floor; **high** = `replicas_at_peak × 24`, peak provisioned and held warm), plus a first-order **`rampup_cold_start_fraction`** and **`expected_latency_ms`** against the **measured ~90 s boot**.

---

## The cost curve (context size → sustainable batch → $/day)

This is the V2 deliverable, now anchored to measurement. Pooled fleet, 1M families, **4 req/family/day**, warm service = 0.93 s (measured Modal L4; LIVE varies 0.79–0.93 s), `diurnal_peak_factor = 3`, GPU $0.80/hr, co-tenancy = True, **cold boot = 90 s (measured)**. The `batch` column is **MEASURED** per context size (`loadtest.py ctxcurve`); the cost is the model rendered at that batch. Cost scales ~linearly with request density.

| assembled context | batch | replicas (base/avg/peak) | warm GPU-hrs/day (low–high) | **$/day (low – mid – high)** |
|---|---:|---:|---:|---:|
| ~184 tok (tiny) | 16 | 3/2.7/9 | 65–216 | **$52 – 58 – 173** |
| ~276 tok | 4 | 11/10.8/33 | 258–792 | **$207 – 211 – 634** |
| ~529 tok | 4 | 11/10.8/33 | 258–792 | **$207 – 211 – 634** |
| ~1,035 tok | 4 | 11/10.8/33 | 258–792 | **$207 – 211 – 634** |
| ~1,937 tok | 2 | 22/21.5/65 | 517–1,560 | **$413 – 422 – 1,248** |
| ~3,367 tok (full) | 1 | 44/43.1/130 | 1,033–3,120 | **$827 – 845 – 2,496** |

**Reading the curve:** the realistic interactive operating band — a few hundred to ~1k tokens of assembled context — is **~$207–634/day at 4 req/family/day** (~$2,067–6,202/day at 40). That is **~180–550× below the ~$116k sparse upper bound**, the core correction. As context grows, batching collapses: at ~2k tokens cost roughly doubles, and at full 4096 context (batch 1) it reaches ~$827–2,496/day. Expected latency stays warm-dominated (~1.18 s = warm service + ~0.28 % ramp-up exposure against the measured 90 s boot) across the curve.

> The optimistic **batch 16** only holds at the smallest (~184-tok) context, and there it sits right at the p95 < 3 s knee. The configured `--max-num-seqs 32` ceiling is **never interactively sustainable** at any tested context (short-ctx c=32 hit p95 ~3.6 s) — a config maximum, not a costable operating point.

### The co-tenancy gate is load-bearing — and full context trips it

The cheapest configurations are exactly the ones that most need the privacy argument. At 4 req/family/day:

| co-tenancy | effective batch | replicas at peak | $/day (high) |
|---|---:|---:|---:|
| **True**, realistic context | 4 (measured) | 33 | **$634** |
| **True**, full context | 1 (measured limit) | 130 | **$2,496** |
| **False** (strict per-family) | 1 (forced) | 130 | **$2,496** |

Full-context interactive traffic lands at `effective_batch = 1` **regardless of the co-tenancy flag** — at full context the hardware imposes the same batch-1 limit strict isolation would. Co-tenancy's win is the *concurrency* it permits at moderate context; it cannot manufacture concurrency the prefill path can't sustain. Even so, batch 1 stays ~47× below the $116k sparse bound, because families still time-share warm replicas; the full collapse to $116k requires the additional (false) assumption that every family boots its *own* GPU.

---

## What live measurement found (Phase 1 — COMPLETE)

The V2 model rested on two unmeasured inputs. A live Modal L4 benchmark (`serving/loadtest.py`; closed-loop concurrency sweeps at p95 < 3 s, distinct per-request context so vLLM's prefix cache cannot share KV — modelling distinct per-family contexts under ADR-0001 salting) measured both.

1. **Sustainable batch width — the dominant input — is a curve that falls with context length.** Measured points (`ctxcurve`): **~184 tok → 16, ~276/529/1,035 tok → 4, ~1,937 tok → 2, ~3,367 tok → 1.** The decline is monotonic and the realistic-context value (batch 4 across three sizes, where c=8 breaks at ~3.2 s) is robust. The hypothesised "~16–21 at full context" is **refuted**: the binding constraint at large context is **prefill compute + queueing, not KV exhaustion** — vLLM's own logs showed GPU KV-cache peaking at **~57 %** (headroom remaining) with **12 requests queued** during the cliff. So sustainable batch sits *below* the naive KV-capacity estimate (~23 sequences of 4096 tokens fit in ~5–6 GB of KV on a 24 GB L4); compute, not cache, is the wall.

2. **Real cold-start exposure.** Cold boot **measured ~77–92 s** (model load from the cache volume) — corrected from the V2 model's assumed 30 s (`MEASURED_COLD_BOOT_SEC = 90`). A diurnal replay **through the gateway** (300 requests over a compressed 2-min day, base 1 → peak 4 req/s) served **0 cold** under a warm baseline: ramp-up cold-start fraction **0.00 %** measured vs ~0.28 % modelled (both negligible). No full scale-to-zero occurred under continuous traffic — **validating the always-warm-baseline premise** at the regime the model targets (fleet scale always has baseline traffic).

**Measurement caveat (recorded in the capture):** Modal load-balances the `/metrics` endpoint across replicas, so under autoscaling the aggregate preemption *counter* is unreliable (it resets when a replica is added — visible as negative deltas). The **p95 SLO cliff is the binding per-replica signal**; vLLM's per-replica logs corroborate via KV % + waiting-queue depth.

**A bug the load test surfaced (and fixed):** firing 300 concurrent requests through the gateway corrupted the hash-chained audit (51/300 records, chain broken) — `AuditStore.append` was not thread-safe. A per-store lock (`audit/store.py`) was added; the chain now stays intact under concurrent load (300/300, verified), with a concurrency regression test.

### The remaining gap: context-distribution realism (not cost-model structure)

The cost curve is measured; the open question is **where real family-digest traffic sits on it.** The cost depends on the assembled-context-token distribution — and that is the one input still unmeasured against production data. The demo fixture is small by design: the gateway's assembler produces **15–77 tokens** (p50 = 27, p90/95/99 = 77) across members × tasks over the 7-item synthetic family — i.e. it lands at the **tiny-context, cheapest end** of the curve (batch 16). A real family with accumulated history (months of location pings, messages, calendar, documents) would assemble far larger context and move up the curve toward the batch-4/2/1 regimes.

**Next proof (named, not done):** measure `assembled_context_tokens` p50/p90/p95/p99 on **production-shaped** family data, then read the operating cost straight off this curve. Until then, the headline band is credible but the *operating point on it* is an assumption. This is the residual uncertainty — it is about input realism, **not** cost-model structure. (Secondary: the per-context knee carries ±noise because Modal's autoscaler added replicas mid-sweep and the endpoint could not be pinned to one replica; the curve's monotonic shape is robust, the small-context absolute value 4–16 is SLO-boundary-sensitive.)

> **Note on `python -m evidence`:** the offline harness runs in **SIMULATED** mode with a *faster* simulated warm latency, so its printed curve sits **below** this report's 0.93 s-basis figures. Set `MODAL_BASE_URL` (LIVE GPU) to render the measured warm-service basis.

---

## Reproduce

```
# the live benchmark that measured the inputs (needs MODAL_BASE_URL + proxy-auth):
python -m serving.loadtest sweep        # batch-width sweep      -> state/loadtest_sweep_capture.json
python -m serving.loadtest ctxcurve     # context-size cost curve -> state/loadtest_ctxcurve_capture.json
python -m serving.loadtest diurnal      # diurnal replay through the gateway

python -m evidence                      # section 4 prints the sparse upper bound + pooled cost curve
python -m pytest -q tests/test_cost.py  # cost-model tests (regime/band ordering, co-tenancy gate, measured curve, latency)
python -m pytest -q                     # full suite
```

## Decision provenance

| Decision | Statement |
|---|---|
| D1 | Retain the sparse model, relabel it the upper bound — do not delete it |
| D2 | Fleet model is peak-concurrency-driven (Little's law at the diurnal peak), not average-rate |
| D3 | Co-tenancy is a typed gated precondition (`cotenancy_permitted`), not a prose caveat |
| D4 | One model emits both cost and a first-order latency (ramp-up cold-start + expected latency) figure |
| D5 | No live GPU spend *in the V2 modelling WP* — the model was built first so the benchmark had something to validate; the live benchmark (this update) is the deferred Phase 1, now executed |
| D6 | The model is the single source of truth — emit a curve, no hand-typed headline; this report's tables are rendered from `telemetry/cost.py` at the measured inputs |
