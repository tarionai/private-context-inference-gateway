# Private Context Inference Gateway

[![tests](https://github.com/tarionai/private-context-inference-gateway/actions/workflows/tests.yml/badge.svg)](https://github.com/tarionai/private-context-inference-gateway/actions/workflows/tests.yml)

**I own the seam between deterministic context data and probabilistic LLM behavior — and I make that seam safe, measurable, cost-aware, and shippable.**

A live, hash-verifiable inference gateway that pairs a deterministic multi-principal context store with self-hosted and hosted LLMs. Built to demonstrate end-to-end ownership of an LLM inference pipeline: model routing, self-hosted serving, privacy-filtered context assembly, eval-gated release, and cost telemetry — over synthetic multi-user (family-shaped) data.

> **Hard rule:** synthetic data only. No real PII, no real credentials. The synthetic multi-principal graph *is* the proprietary-shaped data ("data as the moat").

**Write-up:** [Owning the Seam Between Private Data and the Model](https://www.tarion.ai/blog/private-context-inference-gateway) — the architecture, the assembly-time privacy boundary, and the honest measured-vs-modeled cost story, on tarion.ai.

## What it demonstrates (minimum verifiable set — confirmable in <10 min)

| Capability | How it's shown |
|---|---|
| **Typed inference boundary** | Live `/infer` returns a typed, audited response |
| **Self-hosted serving** | A request route can resolve to self-hosted vLLM (Qwen) on Modal — not only a hosted API |
| **Model routing** | Self-hosted vs hosted (Claude) decided on cost/quality/health; route logged |
| **Privacy at assembly time** | Per-member visibility enforced *before* prompt assembly; every exclusion audited with `policy_version` |
| **Eval-gated release** | A deliberately leaky context change is **blocked** by CI |
| **Cost discipline** | Per-route `cost_usd`, cold-start vs warm P50/P95/P99, cache-hit rate reported separately |

## Reviewer path — one command

```
python -m evidence
```

Prints the whole verifiable set in order: a typed `/infer` response · `route = self_hosted` proof · cold-start **and** warm latency (separate) · per-route cost split (active-serving **and** amortized boot/idle) · the context included/excluded table with `policy_version` · the eval gate **blocking** a deliberately leaky fixture · a tamper-evident audit-chain check · a determinism hash (re-run yields the same hash). Exit 0 means the verifiable set was produced.

Run the tests with `python -m pytest -q` (20 passing). Watch the gate block a regression directly with `python -m eval.gate --inject-leak` (exits non-zero).

## Status

🟢 **All three target competencies proven LIVE; spine shipped and tested (25 tests).** The deterministic pipeline — typed boundary, privacy-filtered assembly, model routing + degradation ladder, active/amortized cost split, blocking eval gate, hash-chained audit — is built and reproducible via `python -m evidence`.

**Self-hosted serving is live, not simulated.** The `self_hosted` route speaks the OpenAI-compatible wire, so it runs against a real local model with **zero cloud credentials**:

```
ollama pull deepseek-r1:14b
SELF_HOSTED_BASE_URL="http://localhost:11434/v1" SELF_HOSTED_MODEL="deepseek-r1:14b" \
  SELF_HOSTED_MAX_TOKENS=512 python -m evidence     # LIVE: real cold/warm latency + measured cost
```

A real 14.8B Q4 model served the request (`route=self_hosted`, real cold/warm latency, measured tokens/sec); the cost split is computed from **measured** boot/active seconds (Ollama's 5-min keep-alive is the real scale-to-zero idle window); a hash-verifiable capture is frozen to `state/evidence_live_capture.json`. See [`serving/local_ollama.md`](./serving/local_ollama.md).

**Honest model-fit finding:** a heavy reasoning model at ~27 s warm is batch/background-grade, not interactive — the self-hosted route is configured accordingly (the router treats self-hosted quality/latency as a runtime input). Interactive traffic targets a small fast model.

**Production scale — proven LIVE on GPU.** The same route runs against `Qwen2.5-7B` on vLLM/Modal (serverless L4, scale-to-zero). Measured on the deployed endpoint:

| Metric | Modal L4 + Qwen2.5-7B (exact $0.7992/hr) | Local Ollama (CPU, reasoning model) |
|---|---|---|
| Warm latency | **0.93 s (interactive-grade)** | ~27 s (batch-grade) |
| Cold (scale-from-zero: spin + load 7B from cache volume) | **~85 s (measured 77–92 s)** | ~40 s |
| Active serving cost | $0.000206/req | — |
| Amortized boot/idle (measured ~85 s boot + 300 s idle, sparse traffic) | ~$0.028/req → **~$113k / 1M families/day** *(sparse isolated upper bound — no pooling)* | ~$0.028/req |
| **Pooled fleet** (1M families, peak-concurrency, warm basis ~0.93 s) | A **measured cost curve by context size** (sustainable batch falls as context grows), at 4 req/family/day: **~$52–173/day** at tiny ~184-tok context (batch 16) → **~$207–634/day** at realistic ~0.5–1k-tok context (batch 4) → **~$827–2,496/day** at full 4096-tok context (batch 1). Cost scales ~linearly with density. MODELED projection over **MEASURED** batch + cold-boot inputs. | — |

> **Regime distinction — read before quoting the cost.** The **~$113k/day** figure is the *sparse isolated upper bound*: it assumes every family boots its own GPU, serves a few requests, then scales to zero — i.e. **zero pooling across families**. ADR-0001 permits multi-family co-tenancy on a warm GPU (shared base model, per-request private context, **salted** KV/prefix cache, sequence-level isolation), so the real fleet pools families onto warm replicas sized at the **diurnal peak** (Little's law) with an always-warm baseline floor — a band **2–3 orders of magnitude lower**. The two load-bearing inputs are now **MEASURED on Modal L4** ([`serving/loadtest.py`](./serving/loadtest.py); frozen capture `state/loadtest_sweep_capture.json`, 4 runs):
>
> - **Sustainable batch width is a measured CURVE that falls with context length.** A context-size sweep (`serving/loadtest.py ctxcurve`) measured **16 at tiny (~184-tok) context → 4 at realistic (~0.5–1k tok) → 2 (~2k) → 1 at full 4096 context**, all at a p95 < 3 s interactive SLO. The earlier "~16–21 at full context" KV estimate is **refuted**: the binding limit at large context is **prefill compute + queueing, not KV exhaustion** (vLLM logs showed KV peaking at ~57% with 12 requests queued). The "16" holds only at the smallest context, right at the SLO knee. `--max-num-seqs 32` is a *configured ceiling*, never interactively sustainable at any tested context.
> - **Cold boot measured ~77–92 s** — vs the model's previously assumed 30 s. Corrected in `telemetry/cost.py`.
>
> So the pooled cost is now a **MODELED-arithmetic-over-MEASURED-inputs curve**: at warm ~0.93 s, 4 req/family/day, it runs **~$52/day (tiny ~184-tok ctx, batch 16) → ~$207–634/day (realistic ~0.5–1k tok, batch 4) → ~$827–2,496/day (full 4096 ctx, batch 1)**; cost scales ~linearly with request density. The one remaining gap is **context-distribution realism**: the demo fixture assembles only ~15–77 tok (cheapest end), so the real operating point needs a production assembled-context p50–p99. `python -m evidence` (LIVE) prints the curve per run (timings vary); SIMULATED mode uses a faster simulated warm latency, so its band sits lower. Math and decisions: [`docs/IMPL_COST_MODEL_V2_fleet_economics.md`](./docs/IMPL_COST_MODEL_V2_fleet_economics.md); measurement method: [`serving/vllm_config.md`](./serving/vllm_config.md).

```
pip install modal && python -m modal setup            # one-time browser auth
modal deploy serving/deploy_modal.py                  # builds image, deploys scale-to-zero
# The endpoint requires proxy auth (requires_proxy_auth=True). Create a token at
# modal.com/settings/proxy-auth-tokens, then pass it via Modal-Key / Modal-Secret:
MODAL_BASE_URL="https://<workspace>--…-serve.modal.run/v1" \
  MODAL_KEY="<token-id>" MODAL_SECRET="<token-secret>" python -m evidence        # LIVE GPU numbers
```

The demo endpoint is kept live but **locked**: unauthenticated requests get HTTP 401; the gateway client forwards `Modal-Key`/`Modal-Secret` from the environment.

The GPU path resolves the build plan's weakest assumption: on real hardware the self-hosted route **is** interactive-viable (0.83 s warm), unlike a CPU reasoning model. The Modal serving stack is a **frozen dependency envelope** — vLLM is a fragile composition of CUDA, PyTorch, FastAPI, Starlette, Prometheus middleware and OpenAI routing; the deploy pins `fastapi==0.136.3` (FastAPI 0.137's router refactor breaks vLLM's metrics middleware — vLLM #45597) and guards it at boot. See `serving/deploy_modal.py`.

Run `python -m evidence` with no env vars for the **SIMULATED** offline mode (byte-deterministic, credential-free) used in CI. Tests: `python -m pytest -q` (25 passing). Watch the gate block a regression: `python -m eval.gate --inject-leak`.

## Latency honesty

This runs serverless scale-to-zero — it has cold starts. Metrics are always reported as **cold-start latency**, **warm P50/P95/P99**, **tokens/sec**, and **cost/request** *separately*. No flat "interactive SLO" claim is made.
