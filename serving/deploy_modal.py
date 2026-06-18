"""vLLM serving Qwen2.5-7B-Instruct on Modal (serverless, scale-to-zero).

The production GPU path. Windows cannot host vLLM (Linux/CUDA), so Modal owns the
container + CUDA and scales to zero when idle — idle families cost $0, which is
itself the cost story. The gateway's self_hosted route speaks the same
OpenAI-compatible wire, so nothing in the app changes between the local Ollama
proof and this production serve.

Run (one-time auth is interactive — do it yourself):

    pip install modal
    modal token new                       # opens a browser; authenticates your account
    modal deploy serving/deploy_modal.py  # builds the image, deploys scale-to-zero

The deploy prints a base URL like  https://<workspace>--<app>-serve.modal.run .
Point the gateway at it and run the LIVE evidence:

    export MODAL_BASE_URL="https://<workspace>--<app>-serve.modal.run/v1"
    python -m evidence                    # LIVE: real cold/warm latency + real cost

Model weights persist in a Modal Volume, so only the FIRST cold start downloads
them; later cold starts just load from the volume. Budget ceiling: $50-$100.
Stop the endpoint when done capturing numbers:  modal app stop private-context-inference-gateway
"""

from __future__ import annotations

import subprocess

import modal

# Frozen dependency envelope. vLLM serving is a fragile composition of CUDA,
# PyTorch, FastAPI, Starlette, Prometheus middleware and OpenAI-compatible
# routing — not "just a model server". FastAPI 0.137 refactored router internals
# so `router.routes` can contain `_IncludedRouter` objects with no `.path`,
# which crashes vLLM's bundled Prometheus middleware (500 on every request,
# vLLM issue #45597). 0.136.3 is the last pre-refactor line and still satisfies
# vLLM 0.21's starlette>=0.49.1 floor.
FASTAPI_PIN = "fastapi[standard]==0.136.3"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Apache-2.0, no HF token required
SERVED_NAME = "family-7b"                # the gateway's default self_hosted model id
GPU = "L4"                               # 24GB; fits 7B fp16 + KV headroom; cheapest viable tier
VLLM_PORT = 8000
MINUTES = 60

# Persist weights + compiled artifacts across cold starts (avoids re-downloading
# ~15GB on every scale-from-zero).
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


def _download_model() -> None:
    """Pull weights into the HF cache volume at BUILD time, so the first request
    only loads from cache (never downloads) — keeps cold start under the timeout."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)


# Pin a current vLLM. If the build fails on this version, bump it — the serve
# flags below are stable across recent releases.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    # vLLM 0.21 transitively requires starlette>=0.49.1, whose `_IncludedRouter`
    # route type crashes the bundled Prometheus metrics middleware
    # (`route.path` AttributeError) on every request. FastAPI can't be pinned
    # down (starlette floor), so upgrade the metrics lib to a release that
    # tolerates the new route types instead.
    .pip_install("vllm==0.21.0", FASTAPI_PIN, "huggingface_hub[hf_transfer]")
    # VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer JIT-compiles its sampling kernel
    # at engine init, which fails on this image and crashes the engine core. Use
    # vLLM's native PyTorch sampler instead. FLASH_ATTN avoids FlashInfer's JIT
    # attention path too.
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "VLLM_USE_V1": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    })
    .run_function(_download_model, volumes={"/root/.cache/huggingface": hf_cache})
)

app = modal.App("private-context-inference-gateway")


@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=5 * MINUTES,   # scale to zero after 5 min idle (the real idle window)
    timeout=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=32)        # continuous-batching width per replica
@modal.web_server(
    port=VLLM_PORT,
    startup_timeout=20 * MINUTES,
    # Lock the public endpoint: callers must send Modal-Key / Modal-Secret
    # headers from a proxy-auth token (Modal dashboard -> Settings -> Proxy Auth
    # Tokens). Keeps the demo endpoint live without exposing an open GPU.
    requires_proxy_auth=True,
)
def serve() -> None:
    # Boot guard: fail loudly if the FastAPI pin ever drifts past the router
    # refactor, rather than 500-ing every request from inside vLLM's middleware.
    from importlib.metadata import version

    fastapi_version = version("fastapi")
    assert tuple(int(p) for p in fastapi_version.split(".")[:2]) < (0, 137), (
        f"vLLM serving requires FastAPI < 0.137 (got {fastapi_version}); "
        "0.137 router refactor breaks Prometheus route instrumentation."
    )

    # --enforce-eager skips CUDA-graph capture (a common cause of engine-core
    # init OOM/crash on a 24GB L4) and cuts cold-start time; lower memory util +
    # context length leave KV headroom. --enable-prefix-caching reuses the family
    # context prefix across requests. See serving/vllm_config.md for rationale.
    subprocess.Popen(
        [
            "vllm", "serve", MODEL_NAME,
            "--served-model-name", SERVED_NAME,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.85",
            "--enforce-eager",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--max-num-seqs", "32",
        ]
    )
