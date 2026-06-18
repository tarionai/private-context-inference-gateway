# Live self-hosted serving — two hosts, one wire

The gateway's `self_hosted` route speaks the **OpenAI-compatible** wire, so any
OpenAI-compatible server backs it unchanged. Two real hosts:

| Host | Use | Hardware | How to point the gateway at it |
|---|---|---|---|
| **Ollama (local)** | credential-free LIVE proof | local CPU/GPU | `SELF_HOSTED_BASE_URL=http://localhost:11434/v1` |
| **vLLM on Modal** | production scale | serverless L4, scale-to-zero | `MODAL_BASE_URL=https://<app>--serve.modal.run/v1` (see `deploy_modal.py`) |

Both are genuinely "self-hosted serving" — *we* host the weights and serve
completions; only the host and model size differ. The local path needs no cloud
account, so the LIVE proof is reproducible by any reviewer with Ollama.

## Reproduce the LIVE evidence run (local, zero credentials)

```bash
ollama pull deepseek-r1:14b          # or any instruct/reasoning model
SELF_HOSTED_BASE_URL="http://localhost:11434/v1" \
SELF_HOSTED_MODEL="deepseek-r1:14b" \
SELF_HOSTED_MAX_TOKENS=512 \
SELF_HOSTED_USD_PER_HOUR=0.80 \
python -m evidence
```

`python -m evidence` then runs in **LIVE** mode:
- forces a true cold start (`ollama stop <model>`) before measuring,
- reports **real** cold-start vs warm latency and tokens/sec,
- computes the cost split from **measured** boot/active seconds (Ollama's 5-min
  keep-alive is the real scale-to-zero idle window) at a stated host `$/hr`,
- freezes a hash-verifiable capture to `state/evidence_live_capture.json`.

## Honest finding — model fit drives route class

A measured warm latency of ~27 s on a 14.8B Q4 **reasoning** model
(`deepseek-r1`) is **not** interactive-grade — the model spends its budget
thinking (reasoning scratch is stripped before the answer reaches the user). The
correct reading, exactly as the build plan pre-authorized: a heavy self-hosted
reasoning model fits the **batch / background / hard-reasoning** route, while
latency-sensitive interactive traffic belongs on a small fast model (hosted-fast,
or a small quantized self-hosted model on a GPU). The router treats self-hosted
health/quality as a runtime input, so this is a configuration choice, not a code
change. The production Modal path (`Qwen2.5-7B`, fp16/AWQ on an L4) targets the
interactive route; the local Ollama path proves the serving + cost mechanics
end-to-end without a cloud account.
