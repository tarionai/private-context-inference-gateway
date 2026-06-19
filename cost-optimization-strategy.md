<text>

## Yes — but only with the right claim

Cost optimization **will make the story stronger** if the proof says:

> “We measured the private self-hosted route, found cold/idle waste, then reduced that waste with explicit serving economics.”

That is a much stronger engineering story than pretending self-hosting is automatically cheaper than hosted APIs.

The unspoken truth: **your current result is a governance/control proof, not a cost/performance proof.** The next benchmark should show that you can inspect the cost structure, identify the waste term, change one variable at a time, and publish the before/after evidence.

---

## Published context

Your current host cost of about `$0.80/hr` matches Modal’s published L4 price: Modal lists Nvidia L4 at `$0.000222/sec`, about `$0.799/hr`. ([Modal][1])

For low-volume serverless GPU, idle policy is load-bearing. Modal says `scaledown_window` defaults to 60s and can be set from 2 seconds to 20 minutes; it also warns that longer warm windows reduce cold starts but bill idle GPU reservation. ([Modal][2]) RunPod’s serverless docs are even more explicit: workers are billed from start until full stop, and idle time is billable; its documented default idle timeout is 5 seconds. ([Runpod Documentation][3]) ([Runpod Documentation][4])

The hosted API comparison is harsh for tiny requests. Groq publishes Llama 3.1 8B at **840 tokens/sec**, `$0.05/M input`, and `$0.08/M output`. ([Groq][5]) Together lists small/8B-class options such as Gemma 3n E4B at `$0.06/M input`, `$0.12/M output`, Qwen2.5 7B at `$0.30/M input/output`, and Llama 3 8B Lite at `$0.14/M input/output`. ([Together AI][6])

---

## Highest-ROI actions you can take today

### 1. Reduce `scaledown_window` from `300s` to `5s` and `2s`

> **Caveat — fixed-density independence error.** The table below holds **requests/window = 3 fixed** while shrinking the idle window. But `scaledown_window` and `requests_per_window` are **not independent** — both fall out of one arrival process: at fixed traffic, a shorter window serves *fewer* requests per warm-up. So these per-request figures are a *single-tenant, sparse-traffic* sensitivity, **not** fleet economics. At fleet scale, cost is governed by the **pooled, peak-concurrency model** (`telemetry/cost.py` → `PooledDiurnalFleetCostModel`), not by this table — see [`docs/IMPL_COST_MODEL_V2_fleet_economics.md`](./docs/IMPL_COST_MODEL_V2_fleet_economics.md).

This is the best immediate proof because it changes only the billing shape.

Using your numbers:

```text
hourly = $0.7992/hr
boot = 89.2s
warm active = 0.939s/request
requests/window = 3
```

| Idle window | Estimated cost/request | Improvement vs current |
| ----------: | ---------------------: | ---------------------: |
|        300s |              `$0.0290` |               baseline |
|         60s |              `$0.0112` |          ~2.6× cheaper |
|         30s |              `$0.0090` |          ~3.2× cheaper |
|         10s |              `$0.0075` |          ~3.8× cheaper |
|          5s |              `$0.0072` |          ~4.0× cheaper |
|          2s |              `$0.0070` |          ~4.1× cheaper |

**Recommendation:** run three captures today:

```text
scaledown_window=300
scaledown_window=30
scaledown_window=5
```

Publish them as a cost-ablation table.

**Why this strengthens the story:** it proves you understand the dominant cost term and can optimize it without weakening privacy, routing, audit, or eval gates.

---

### 2. Add request-density sweep: `3 / 10 / 30 / 100 requests per warm window`

At your current request density, boot dominates. At higher density, cost falls quickly.

Assuming `scaledown_window=5s`:

| Requests per warm window | Estimated cost/request |
| -----------------------: | ---------------------: |
|                        3 |             `$0.00718` |
|                       10 |             `$0.00230` |
|                       30 |             `$0.00091` |
|                      100 |             `$0.00042` |
|                      300 |             `$0.00028` |

**Recommendation:** implement a simple benchmark mode:

```text
python -m evidence --burst 3
python -m evidence --burst 10
python -m evidence --burst 30
python -m evidence --burst 100
```

Report:

```text
cold_start_s
warm_latency_p50
warm_latency_p95
tokens/sec
cost/request_active_only
cost/request_amortized
requests_per_warm_window
scaledown_window_s
```

**Why this strengthens the story:** it turns the result from “one lucky demo run” into an actual cost curve.

---

### 3. Move model loading out of request path

Modal’s cold-start guide says model downloads during boot should be moved ahead of time into the image or a Modal Volume, and that for models in the tens of GB this can reduce boot time from minutes to seconds. ([Modal][2]) It also says initialization can be moved into global scope or a container `enter` method so containers are not marked warm before initialization completes. ([Modal][2])

**Today action:**

* Store weights in a Modal Volume or baked image layer.
* Start the inference server in `@modal.enter`.
* Run one synthetic warmup request during enter.
* Do not route external requests until `/health` passes.

Expected proof target:

| Metric          | Current |                   Target |
| --------------- | ------: | -----------------------: |
| Cold start      | `90.1s` |      `<30s` first target |
| Warm latency    | `939ms` |      preserve or improve |
| Warm tokens/sec |  `11.7` | improve with longer test |

**Why this matters:** it attacks the biggest non-idle cost term: the 89.2s boot.

---

### 4. Test Modal Memory Snapshots, but label it experimental if using GPU snapshots

Modal documents Memory Snapshots as a way to reduce cold start latency by reusing initialized memory state, with practical initialization-heavy functions often starting **3–10× faster**. ([Modal][7]) GPU Memory Snapshots exist, but Modal labels them **Alpha**, and warns that they do not speed model loading from storage when storage bandwidth is the bottleneck. ([Modal][7])

**Today action:**

Run one controlled branch:

```python
@app.cls(
    gpu="L4",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
```

Only publish it if it is stable.

**Expected proof target:**

| Scenario              |               Cold start target |
| --------------------- | ------------------------------: |
| No snapshot           |                            ~90s |
| CPU snapshot only     |                    maybe modest |
| GPU snapshot + warmup | target `<30s`; strong if `<15s` |

**Caution:** do not make this the main story. Because GPU snapshots are alpha, the reliable story is still “boot/idle amortization discipline.”

---

### 5. Enable concurrency / continuous batching

Modal supports `@modal.concurrent`, and its docs explicitly call out GPU inference servers such as vLLM as a use case because continuous batching can batch inputs even when they do not arrive simultaneously. ([Modal][8]) vLLM itself lists continuous batching, chunked prefill, prefix caching, CUDA graphs, quantization, and optimized kernels among its serving features. ([vLLM][9])

**Today action:**

Try:

```python
@modal.concurrent(max_inputs=8, target_inputs=4)
```

Then run:

```text
concurrency = 1, 2, 4, 8
```

Report:

```text
P50 latency
P95 latency
requests/sec
output_tokens/sec
cost/request
GPU utilization
```

**What good looks like:**

* P95 does not explode.
* Tokens/sec increases.
* Cost/request drops at concurrency `4` or `8`.

**Why this strengthens the story:** it proves the private route is not just functional; it can amortize GPU time under realistic burst traffic.

---

### 6. Add prefix caching for repeated policy/system context

Your gateway probably has repeated system/policy scaffolding across requests. vLLM’s prefix caching avoids recomputing repeated prompt prefixes and supports per-request cache salting for privacy isolation. ([vLLM][10]) SGLang’s runtime also emphasizes RadixAttention prefix caching, continuous batching, structured outputs, chunked prefill, and quantization. ([GitHub][11])

**Today action:**

Create two benchmark modes:

```text
cache_mode=off
cache_mode=on
```

Use 20 repeated requests with the same policy prefix but different private context.

Report:

```text
TTFT
prefill_tokens/sec
end_to_end_latency
cache_hit_rate
```

**Privacy caveat:** use cache salt by family/account/tenant. Do not let cross-family private context share cache.

---

### 7. Compare vLLM against TGI or SGLang only after the billing fix

Hugging Face claims TGI v3 processes **3× more tokens** and is **13× faster than vLLM on long prompts**, and specifically says a single L4 can handle about **30k tokens** on Llama 3.1 8B versus roughly **10k** for vLLM in that comparison. ([Hugging Face][12]) SGLang’s public repo highlights RadixAttention, continuous batching, prefill/decode disaggregation, quantization, and structured-output optimizations. ([GitHub][11])

**Recommendation:** do not switch engines first. First fix idle/window economics. Then run an engine bakeoff:

```text
vLLM baseline
vLLM + prefix caching
TGI v3
SGLang
```

Use the same model, same prompt set, same GPU, same concurrency.

**Best story:** “We kept the governance contract stable while swapping the serving engine behind it.”

That shows clean architecture.

---

### 8. Right-size hardware after you have the benchmark harness

RunPod lists L4 from `$0.39/hr`, materially below your current `$0.799/hr` Modal L4 rate. ([runpod.io][13]) But moving platforms before fixing idle/cold-start evidence risks creating a messy comparison.

**Today action:** add a provider-cost parameter to the evidence output:

```text
SELF_HOSTED_USD_PER_HOUR=0.7992
SELF_HOSTED_USD_PER_HOUR=0.39
```

Then show projected cost under both rates.

At `scaledown_window=5s`, 3 requests/window:

| Hourly GPU rate | Estimated cost/request |
| --------------: | ---------------------: |
|     `$0.799/hr` |             `$0.00718` |
|      `$0.39/hr` |             `$0.00350` |

**Recommendation:** use provider right-sizing as the second proof, not the first. The first proof should be platform-independent cost discipline.

---

## What not to do first

### Do not keep `min_containers=1` for the cost story

`min_containers=1` improves latency but destroys low-volume cost. Modal explicitly says `min_containers` prevents scale-to-zero, while idle warm containers increase consumed resources. ([Modal][2])

At Modal L4 pricing, one always-warm L4 is roughly:

```text
$0.000222/sec × 3600 × 24 × 30 ≈ $575/month
```

That may be acceptable for production SLA, but it weakens the “scale-to-zero cost” claim.

### Do not compare only against Groq on 42-token prompts

Groq’s Llama 3.1 8B pricing and 840 TPS make tiny-prompt hosted inference almost impossible to beat on raw economics. ([Groq][5]) Your better comparison is not “cheaper than Groq.” It is:

> “For private context that cannot leave the control boundary, we measured and reduced the self-hosted privacy premium.”

---

## Recommended benchmark sequence

### Phase 1 — Same system, billing optimization

```text
A0: current baseline
A1: scaledown_window=30
A2: scaledown_window=5
A3: scaledown_window=2
```

Gate:

```text
privacy filter unchanged
leaky fixture still blocked
audit chain intact
answer unchanged or semantically equivalent
```

### Phase 2 — Same system, density optimization

```text
burst=3
burst=10
burst=30
burst=100
```

Gate:

```text
P95 latency acceptable
cost/request decreases monotonically
no policy/audit regression
```

### Phase 3 — Same contract, serving optimization

```text
vLLM baseline
vLLM + prefix cache
TGI v3
SGLang
```

Gate:

```text
same /infer contract
same policy-stamped context table
same eval gate result
same audit chain verification
```

---

## Recommended public claim after optimization

Use this framing:

> “Initial live benchmark proved self-hosted private inference, policy-stamped context exclusion, leaky-regression blocking, and hash-chained audit. A follow-up cost ablation reduced amortized request cost by tuning serverless idle policy and request density, while preserving the same privacy and governance gates.”

That is a strong proof story because it shows:

1. **Control boundary**
2. **Measurable baseline**
3. **Identified bottleneck**
4. **Targeted optimization**
5. **No governance regression**

## Bottom line

Yes, optimize for cost — but optimize the **waste terms** first:

1. `scaledown_window`
2. request density
3. cold-start initialization
4. concurrency / batching
5. prefix caching
6. serving-engine bakeoff
7. hardware/provider right-sizing

The best immediate result would be a second evidence run showing:

```text
amortized cost/request: $0.0288 → ~$0.0072
cold/warm/audit/privacy gates: unchanged
```

That is a credible **4× same-day cost optimization proof**.

</text>

<probability>0.9</probability>

[1]: https://modal.com/pricing "Plan Pricing | Modal"
[2]: https://modal.com/docs/guide/cold-start "Cold start performance | Modal Docs"
[3]: https://docs.runpod.io/serverless/pricing "Pricing - Runpod Documentation"
[4]: https://docs.runpod.io/serverless/endpoints/endpoint-configurations "Endpoint settings - Runpod Documentation"
[5]: https://groq.com/pricing "Groq On-Demand Pricing for Tokens-as-a-Service | Groq is fast, low cost inference."
[6]: https://www.together.ai/pricing "Pricing | Together AI"
[7]: https://modal.com/docs/guide/memory-snapshots "Memory Snapshots | Modal Docs"
[8]: https://modal.com/docs/guide/concurrent-inputs "Input concurrency | Modal Docs"
[9]: https://docs.vllm.ai/en/latest/ "vLLM"
[10]: https://docs.vllm.ai/en/stable/design/prefix_caching/ "Automatic Prefix Caching - vLLM"
[11]: https://github.com/sgl-project/sglang "GitHub - sgl-project/sglang: SGLang is a high-performance serving framework for large language models and multimodal models. · GitHub"
[12]: https://huggingface.co/docs/text-generation-inference/en/conceptual/chunking "TGI v3 overview · Hugging Face"
[13]: https://www.runpod.io/gpu-models/l4 "L4 GPU Cloud | Runpod"
