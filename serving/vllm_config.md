# vLLM serve config & rationale

Self-hosted route: `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) served by **vLLM**
(OpenAI-compatible) on Modal serverless GPU.

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name family-7b \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 64 \
  --enable-auto-tool-choice --tool-call-parser hermes   # optional; tool calling not required
```

| Flag | Why |
|---|---|
| `--enable-prefix-caching` | Reuse the family context prefix across requests — the key cost lever for repeated context assembly. Drives "prompt tokens avoided" in the cost rollup. |
| `--enable-chunked-prefill` | Overlap prefill with decode; smoother latency under mixed traffic. |
| `--max-num-seqs 64` | Continuous-batching width; throughput under concurrent family requests. |
| `--gpu-memory-utilization 0.90` | Leave KV-cache headroom on a 24GB L4. |
| `--max-model-len 8192` | Bounded context; enough for assembled family context + query. |

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
