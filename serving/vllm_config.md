# vLLM serve config & rationale

Self-hosted route: `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) served by **vLLM**
(OpenAI-compatible) on Modal serverless GPU.

```bash
# As deployed (serving/deploy_modal.py); --enforce-eager added there for L4 cold-start stability.
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name family-7b \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 32
```

| Flag | Why |
|---|---|
| `--enable-prefix-caching` | Reuse the family context prefix across requests. **Note:** ADR-0001 mandates per-tenant cache salting, so it does NOT share KV *across families* — only within a family's repeated prefix. The batch-width measurement below uses distinct per-request contexts to reflect this. |
| `--enable-chunked-prefill` | Overlap prefill with decode; smoother latency under mixed traffic. |
| `--max-num-seqs 32` | Configured continuous-batching ceiling. **Measured sustainable width is far lower** (see below). |
| `--gpu-memory-utilization 0.85` | Leave KV-cache headroom on a 24GB L4. |
| `--max-model-len 4096` | Bounded context; enough for assembled family context + query. |

## Measured sustainable batch width (Modal L4, 2026-06-18)

Measured by `serving/loadtest.py` — closed-loop concurrency sweeps at an interactive
SLO of **p95 < 3 s**, with a **distinct per-request context** (a unique prefix per
request, so vLLM's prefix cache cannot share KV — modelling distinct per-family
contexts under ADR-0001 salting). The `ctxcurve` mode sweeps context size to produce
the curve below; frozen captures `state/loadtest_sweep_capture.json` and
`state/loadtest_ctxcurve_capture.json`.

**Sustainable batch width is a CURVE that falls with context length** (not a single value):

| Assembled context | **Sustainable interactive batch (p95<3s)** |
|---|---:|
| ~184 tok (tiny) | **16** (sits right at the SLO knee) |
| ~276 / ~529 / ~1,035 tok (realistic) | **4** (c=8 breaks at ~3.2 s across all three) |
| ~1,937 tok | **2** |
| ~3,367 tok (full 4096) | **1** (c=2 already ~5 s; ~2 req/s saturation) |

Key findings:

- The hypothesised KV-bound width of ~16–21 at full 4096 context is **refuted**: a
  single L4 sustains **batch 1** at full context under an interactive SLO.
- The binding constraint at full context is **prefill compute + queueing, not KV
  exhaustion**: vLLM's own logs showed GPU KV-cache usage peaking at **~57%** (not
  full) with **12 requests queued** (`Waiting: 12`) during the cliff. So sustainable
  batch (1) is *below* the naive KV-capacity estimate (~23 sequences of 4096 tokens
  fit in ~5–6 GB of KV on a 24 GB L4) — compute, not cache, is the wall.
- **Cold boot measured ~77–92 s** (model load from the cache volume), vs the cost
  model's previously assumed 30 s — corrected in `telemetry/cost.py`.
- **`--max-num-seqs 32` is never interactively sustainable** at any tested context
  (short-ctx c=32 hits p95 ~3.6 s); it is a configured ceiling, not a capacity.

**Measurement caveat:** Modal load-balances the `/metrics` endpoint across replicas,
so under autoscaling the aggregate preemption *counter* is unreliable (it resets when
a replica is added). The **p95 SLO cliff is the binding per-replica signal**; vLLM's
per-replica logs corroborate via KV% + waiting-queue depth.

**The remaining gap is context-distribution realism, not cost-model structure:** the
curve is measured, but *where real family-digest traffic sits on it* depends on the
assembled-context-token distribution. The demo fixture assembles only ~15–77 tok
(p50 27, p99 77) — the cheapest end. A real p50/p90/p95/p99 needs production data; the
named next proof is to measure it and read cost straight off this curve.

These measured numbers feed `telemetry/cost.py` (`MEASURED_BATCH_BY_CONTEXT` curve,
`MEASURED_BATCH_TINY_CTX`/`SHORT_CTX`/`FULL_CTX`, `MEASURED_COLD_BOOT_SEC`); the
README pooled cost curve and `python -m evidence` render from them.

## Engine choice

- **vLLM** — PagedAttention, continuous batching, chunked prefill, prefix caching,
  structured outputs, tool calling. Chosen.
- **TGI** — maintenance mode; comparison row only, not built on.
- **SGLang** — credible alternative; RadixAttention is the comparison point for
  prefix reuse. Not built twice.

## Quantization (Slice 3 only)

Measure the fp16 route first. Then add `--quantization awq` with AWQ weights to
fit a cheaper GPU and demonstrate the quantization lever. Do not quantize before
the fp16 baseline is measured.

## Tool calling

Optional. Plain chat completion through the typed router is sufficient for the
self-hosted serving proof. Treat the Hermes tool-call parser as a test assertion
to be made green at build time, not a standing claim.
