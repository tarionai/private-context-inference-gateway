"""Live load tester for the self-hosted vLLM route (BUILD_PLAN §7).

Two measurements the pooled cost model (telemetry/cost.py) leaves UNMEASURED:

  sweep   -- Unknown 1: sustainable batch width. A closed-loop concurrency sweep
             {1,2,4,8,16,24,32} at SHORT and FULL-4096 context, hitting the raw
             OpenAI-compatible endpoint. Per level: p50/p95/p99 latency, throughput
             (req/s + output tok/s), and whether vLLM preempted / hit KV-cache
             pressure (scraped from the /metrics Prometheus endpoint). Sustainable
             width = highest concurrency holding the interactive p95 SLO with ZERO
             preemption -- the number fed back into max_concurrent_per_replica.

  diurnal -- Unknown 2 / Tier B: a compressed synthetic diurnal arrival trace
             replayed END-TO-END THROUGH THE GATEWAY (so the privacy-filter + audit
             path is exercised under load), measuring the real ramp-up cold-start
             fraction. Cold is detected by latency threshold (a freshly-booted Modal
             replica pays ~90s vs ~1s warm), which captures real autoscaler boots,
             not a per-client first-call flag.

Endpoint resolution mirrors serving/clients.py: MODAL_BASE_URL (or
SELF_HOSTED_BASE_URL), with optional Modal-Key/Modal-Secret proxy-auth headers
sent only when both are present. Results are frozen to a hash-verifiable JSON
capture (mirrors state/evidence_live_capture.json).

    MODAL_BASE_URL=https://<ws>--<app>-serve.modal.run/v1 python -m serving.loadtest sweep
    MODAL_BASE_URL=...                                     python -m serving.loadtest diurnal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# --- Stated load-test assumptions (the SLO and shapes are explicit) -----------
CONCURRENCY_LEVELS = (1, 2, 4, 8, 16, 24, 32)
INTERACTIVE_P95_SLO_MS = 3000.0     # stated interactive SLO: p95 < 3 s
DEFAULT_DURATION_SEC = 20.0         # closed-loop hold time per concurrency level
MAX_OUTPUT_TOKENS = 128             # fixed output size so token load is comparable
SHORT_PROMPT_TOKENS = 128          # SHORT context row
FULL_PROMPT_TOKENS = 2800          # FULL row: ~2.8k-token target, conservative vs max-model-len 4096
                                   # (real tokenization is denser than 4 chars/token; stay clear of the
                                   # 4096 ceiling so requests aren't rejected -- measured tokens recorded)
COLD_LATENCY_THRESHOLD_MS = 10_000.0  # latency above this => a cold-booted replica served it
_CHARS_PER_TOKEN = 4               # rough char->token factor for prompt sizing only
_SYSTEM = "Answer concisely using only the provided family context."
_CONTEXT_SIZES = {"short": SHORT_PROMPT_TOKENS, "full4096": FULL_PROMPT_TOKENS}


@dataclass(frozen=True)
class RequestSample:
    latency_ms: float
    completion_tokens: int
    prompt_tokens: int
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class SweepLevelResult:
    context_label: str
    concurrency: int
    requests: int
    errors: int
    duration_sec: float
    prompt_tokens_measured: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_req_s: float
    output_tok_s: float
    preemptions_delta: int
    kv_cache_usage_peak: float
    meets_slo: bool
    preemption_free: bool
    sustainable: bool


# --- Endpoint resolution (mirrors serving/clients.py) -------------------------


def _resolve_endpoint() -> tuple[str, dict[str, str], str]:
    base = os.environ.get("MODAL_BASE_URL") or os.environ.get("SELF_HOSTED_BASE_URL")
    if not base:
        raise SystemExit("set MODAL_BASE_URL (or SELF_HOSTED_BASE_URL) to the /v1 endpoint")
    headers: dict[str, str] = {}
    if os.environ.get("MODAL_KEY") and os.environ.get("MODAL_SECRET"):
        headers = {"Modal-Key": os.environ["MODAL_KEY"], "Modal-Secret": os.environ["MODAL_SECRET"]}
    root = base[:-3] if base.endswith("/v1") else base.rstrip("/")
    return base, headers, root.rstrip("/") + "/metrics"


def _model_name() -> str:
    return os.environ.get("MODAL_MODEL_NAME") or os.environ.get("SELF_HOSTED_MODEL", "family-7b")


# --- Pure helpers (offline-testable) ------------------------------------------


def _build_prompt(target_tokens: int, seed: str = "") -> str:
    """A family-context prompt padded to roughly target_tokens.

    `seed` makes the prompt unique from token 0 so vLLM's prefix cache cannot share
    KV across requests -- modelling DISTINCT per-family contexts (ADR-0001 mandates
    per-tenant cache salting, so cross-family prefix sharing is disabled in
    production). Without a seed, identical prompts share KV and understate KV
    pressure -- the wrong regime for a sustainable-batch-width measurement."""
    tag = f"[family {seed}] " if seed else ""
    unit = f"{tag}household note: groceries, schedules, and a pet update. "
    reps = max(1, (target_tokens * _CHARS_PER_TOKEN) // len(unit))
    # Seed at token 0 so the very first cache block differs -> no prefix-cache sharing.
    return f"{tag}Summarize the private updates for this family.\n" + unit * reps


def _percentile(sorted_ms: list[float], q: float) -> float:
    """Nearest-rank percentile (rank = ceil(q/100 * N)) over sorted latencies (ms)."""
    if not sorted_ms:
        return 0.0
    rank = math.ceil(q / 100.0 * len(sorted_ms))
    idx = min(len(sorted_ms) - 1, max(0, rank - 1))
    return sorted_ms[idx]


def _scan_metric(text: str, needle: str, reducer: str) -> float:
    """Sum or max the sample values of every non-comment metric line containing needle."""
    values = []
    for line in text.splitlines():
        if line.startswith("#") or needle not in line:
            continue
        try:
            values.append(float(line.rsplit(" ", 1)[-1]))
        except ValueError:
            continue
    if not values:
        return 0.0
    return sum(values) if reducer == "sum" else max(values)


def _summarize_level(
    label: str, concurrency: int, samples: list[RequestSample],
    duration_sec: float, preempt_delta: int, kv_peak: float,
) -> SweepLevelResult:
    ok = [s for s in samples if s.ok]
    lat = sorted(s.latency_ms for s in ok)
    errors = len(samples) - len(ok)
    p95 = _percentile(lat, 95)
    meets = bool(lat) and p95 < INTERACTIVE_P95_SLO_MS and errors == 0
    # Only a POSITIVE delta is a real preemption. A negative delta is an artifact of
    # Modal load-balancing /metrics across replicas (the per-replica counter resets
    # when the autoscaler adds a replica), so it must not count as a preemption.
    free = preempt_delta <= 0
    prompt_tokens = sorted(s.prompt_tokens for s in ok)
    return SweepLevelResult(
        context_label=label, concurrency=concurrency, requests=len(samples), errors=errors,
        duration_sec=round(duration_sec, 1),
        prompt_tokens_measured=int(_percentile(prompt_tokens, 50)),
        p50_ms=round(_percentile(lat, 50), 1),
        p95_ms=round(p95, 1), p99_ms=round(_percentile(lat, 99), 1),
        throughput_req_s=round(len(ok) / duration_sec, 2) if duration_sec else 0.0,
        output_tok_s=round(sum(s.completion_tokens for s in ok) / duration_sec, 1) if duration_sec else 0.0,
        preemptions_delta=int(preempt_delta), kv_cache_usage_peak=round(kv_peak, 3),
        meets_slo=meets, preemption_free=free, sustainable=meets and free,
    )


def _sustainable_width(levels: list[SweepLevelResult]) -> int:
    """Highest concurrency that is sustainable with every lower level also sustainable.

    Contiguous from the smallest level: once p95 breaks the SLO or a preemption
    appears it does not 'recover' at higher load, so a contiguous run is the honest
    width (no cherry-picking a lucky high level)."""
    width = 0
    for level in sorted(levels, key=lambda x: x.concurrency):
        if not level.sustainable:
            break
        width = level.concurrency
    return width


def _max_slo_meeting_width(levels: list[SweepLevelResult]) -> int:
    """Highest concurrency meeting the SLO, ignoring contiguity.

    Reported alongside the strict contiguous width so an isolated transient dip
    (e.g. an autoscaler scale-up event mid-sweep) does not masquerade as the
    capacity ceiling. The gap between the two flags transient noise."""
    meeting = [level.concurrency for level in levels if level.sustainable]
    return max(meeting) if meeting else 0


def _freeze(payload: dict, path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(out.read_bytes()).hexdigest()


# --- Wire calls + closed-loop sweep -------------------------------------------


def _fire_one(client, model: str, prompt: str, max_tokens: int) -> RequestSample:
    started = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=max_tokens,
            messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = completion.usage
        return RequestSample(
            latency_ms=latency_ms, completion_tokens=getattr(usage, "completion_tokens", 0),
            prompt_tokens=getattr(usage, "prompt_tokens", 0), ok=True,
        )
    except Exception as exc:  # noqa: BLE001 -- a load tool must record and continue, not die
        return RequestSample((time.perf_counter() - started) * 1000.0, 0, 0, False, type(exc).__name__)


def _worker_loop(client, model, prompt_tokens, max_tokens, deadline, sink: list) -> None:
    while time.perf_counter() < deadline:
        prompt = _build_prompt(prompt_tokens, seed=uuid.uuid4().hex[:8])  # unique => no prefix-cache share
        sink.append(_fire_one(client, model, prompt, max_tokens))


def _run_level(client, model, prompt_tokens, concurrency, duration_sec, max_tokens) -> list[RequestSample]:
    deadline = time.perf_counter() + duration_sec
    buckets: list[list[RequestSample]] = [[] for _ in range(concurrency)]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_worker_loop, client, model, prompt_tokens, max_tokens, deadline, buckets[i])
            for i in range(concurrency)
        ]
        for future in futures:
            future.result()
    return [sample for bucket in buckets for sample in bucket]


def _warmup(client, model: str) -> RequestSample:
    """First contact: pays the cold boot if the replica is scaled to zero. Reconfirms boot sec."""
    return _fire_one(client, model, "Say OK.", 8)


def _poll_kv(metrics_url: str, headers: dict[str, str], stop: threading.Event, out: list[float]) -> None:
    """Sample the KV-cache usage gauge DURING the level (it drains to ~0 between levels)."""
    peak = 0.0
    while not stop.is_set():
        peak = max(peak, _scan_metric(_fetch_metrics(metrics_url, headers), "gpu_cache_usage", "max"))
        stop.wait(0.5)
    out.append(peak)


def _measure_level(client, model, prompt_tokens, label, concurrency, duration, metrics_url, headers) -> SweepLevelResult:
    before = _scan_metric(_fetch_metrics(metrics_url, headers), "num_preemptions", "sum")
    stop = threading.Event()
    kv_out: list[float] = []
    poller = threading.Thread(target=_poll_kv, args=(metrics_url, headers, stop, kv_out), daemon=True)
    poller.start()
    samples = _run_level(client, model, prompt_tokens, concurrency, duration, MAX_OUTPUT_TOKENS)
    stop.set()
    poller.join(timeout=5)
    after = _scan_metric(_fetch_metrics(metrics_url, headers), "num_preemptions", "sum")
    return _summarize_level(label, concurrency, samples, duration, int(after - before),
                            max(kv_out) if kv_out else 0.0)


def _sweep_context(client, model, label, prompt_tokens, metrics_url, headers, duration,
                   concurrency_levels=CONCURRENCY_LEVELS) -> list[SweepLevelResult]:
    levels: list[SweepLevelResult] = []
    for concurrency in concurrency_levels:
        level = _measure_level(client, model, prompt_tokens, label, concurrency, duration, metrics_url, headers)
        levels.append(level)
        _print_level(level)
    return levels


def _fetch_metrics(metrics_url: str, headers: dict[str, str]) -> str:
    import httpx  # wrapped at the boundary; openai already pulls httpx

    try:
        return httpx.get(metrics_url, headers=headers or None, timeout=10.0).text
    except Exception:  # noqa: BLE001 -- metrics are best-effort; absence => 0 preemptions recorded
        return ""


def _print_level(level: SweepLevelResult) -> None:
    flag = "OK " if level.sustainable else "BREAK"
    print(
        f"  [{level.context_label:8s}] c={level.concurrency:2d} ptok={level.prompt_tokens_measured:5d}  "
        f"p50={level.p50_ms:7.0f} p95={level.p95_ms:7.0f} p99={level.p99_ms:7.0f} ms  "
        f"{level.throughput_req_s:5.1f} req/s  {level.output_tok_s:6.0f} tok/s  "
        f"preempt+{level.preemptions_delta} kv={level.kv_cache_usage_peak:.2f}  {flag}"
    )


def run_sweep(args: argparse.Namespace) -> int:
    base, headers, metrics_url = _resolve_endpoint()
    from openai import OpenAI  # wrapped at the boundary; lazy import

    client = OpenAI(base_url=base, api_key="x", timeout=900.0, default_headers=headers or None)
    model = _model_name()
    print(f"load-test SWEEP -> {base}  model={model}  SLO p95<{INTERACTIVE_P95_SLO_MS:.0f}ms")
    warm = _warmup(client, model)
    if not warm.ok:
        raise SystemExit(f"warmup failed ({warm.error}) -- check endpoint/auth before spending GPU")
    cold_boot_sec = round(warm.latency_ms / 1000.0, 1)
    print(f"  cold-boot (first contact): {cold_boot_sec:.1f}s")
    levels: list[SweepLevelResult] = []
    sustainable: dict[str, int] = {}
    max_slo_width: dict[str, int] = {}
    for label, prompt_tokens in _CONTEXT_SIZES.items():
        ctx_levels = _sweep_context(client, model, label, prompt_tokens, metrics_url, headers, args.duration)
        levels.extend(ctx_levels)
        sustainable[label] = _sustainable_width(ctx_levels)
        max_slo_width[label] = _max_slo_meeting_width(ctx_levels)
    payload = {
        "mode": "sweep", "model": model, "slo_p95_ms": INTERACTIVE_P95_SLO_MS,
        "duration_sec_per_level": args.duration, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "concurrency_levels": list(CONCURRENCY_LEVELS), "cold_boot_sec": cold_boot_sec,
        "context_prompt_tokens": _CONTEXT_SIZES,
        "sustainable_width": sustainable,            # strict: contiguous from c=1
        "max_slo_meeting_width": max_slo_width,      # highest SLO-meeting level (transient-tolerant)
        "unique_prefix_per_request": True,  # distinct per-family context; defeats prefix cache (ADR-0001 salting)
        "preemption_metric_caveat": (
            "preemptions_delta is unreliable under autoscaling: Modal load-balances /metrics "
            "across replicas, so the aggregate counter resets/jumps (negative deltas). The p95 "
            "SLO cliff is the binding per-replica signal; vLLM logs report per-replica KV% + waiting reqs."
        ),
        "levels": [asdict(level) for level in levels],
    }
    digest = _freeze(payload, args.out)
    print(f"\n  sustainable batch width (strict contiguous): {sustainable}")
    print(f"  max SLO-meeting width (transient-tolerant):  {max_slo_width}")
    print(f"  frozen: {args.out}\n  sha256(capture) = {digest}")
    return 0


# --- Context-size sweep: turn the batch-width bracket into a cost curve --------
# Target prompt tokens -> measured ~256/512/1k/2k/3.5k (tokenizer runs ~1.25x denser
# than 4 chars/token). The top stays clear of max-model-len 4096 (prompt + output).
_CTX_CURVE_TARGETS = (200, 400, 800, 1600, 2800)
_CTX_CURVE_LADDER = (1, 2, 4, 8, 16, 32)


def _curve_point(levels: list[SweepLevelResult]) -> dict:
    measured = levels[0].prompt_tokens_measured if levels else 0
    return {
        "measured_tokens": measured,
        "sustainable_batch": _max_slo_meeting_width(levels),  # transient-tolerant knee
        "strict_batch": _sustainable_width(levels),           # contiguous from c=1
    }


def run_ctxcurve(args: argparse.Namespace) -> int:
    base, headers, metrics_url = _resolve_endpoint()
    from openai import OpenAI  # wrapped at the boundary; lazy import

    client = OpenAI(base_url=base, api_key="x", timeout=900.0, default_headers=headers or None)
    model = _model_name()
    print(f"load-test CTXCURVE -> {base}  model={model}  SLO p95<{INTERACTIVE_P95_SLO_MS:.0f}ms")
    warm = _warmup(client, model)
    if not warm.ok:
        raise SystemExit(f"warmup failed ({warm.error}) -- check endpoint/auth before spending GPU")
    cold_boot_sec = round(warm.latency_ms / 1000.0, 1)
    print(f"  cold-boot (first contact): {cold_boot_sec:.1f}s")
    curve, levels = [], []
    for target in args.context_tokens:
        ctx_levels = _sweep_context(client, model, f"ctx{target}", target, metrics_url, headers,
                                    args.duration, _CTX_CURVE_LADDER)
        levels.extend(asdict(level) for level in ctx_levels)
        point = {"target_tokens": target, **_curve_point(ctx_levels)}
        curve.append(point)
        print(f"  -> ~{point['measured_tokens']:>4d} tok: sustainable batch = {point['sustainable_batch']}")
    payload = {
        "mode": "ctxcurve", "model": model, "slo_p95_ms": INTERACTIVE_P95_SLO_MS,
        "duration_sec_per_level": args.duration, "ladder": list(_CTX_CURVE_LADDER),
        "cold_boot_sec": cold_boot_sec, "unique_prefix_per_request": True,
        "curve": curve, "levels": levels,
    }
    digest = _freeze(payload, args.out)
    print("\n  context->batch curve: " + ", ".join(f"{c['measured_tokens']}tok:{c['sustainable_batch']}" for c in curve))
    print(f"  frozen: {args.out}\n  sha256(capture) = {digest}")
    return 0


# --- Diurnal replay through the gateway (Tier B) ------------------------------


def _diurnal_trace(minutes: float, peak_rps: float, base_rps: float) -> list[float]:
    """Arrival offsets (sec) over a compressed day: rate = base + (peak-base)*sin^2(pi*t)."""
    total = minutes * 60.0
    offsets: list[float] = []
    carry = 0.0
    step = 1.0
    elapsed = 0.0
    while elapsed < total:
        rate = base_rps + (peak_rps - base_rps) * math.sin(math.pi * elapsed / total) ** 2
        carry += rate * step
        while carry >= 1.0:
            offsets.append(elapsed)
            carry -= 1.0
        elapsed += step
    return offsets


@dataclass(frozen=True)
class DiurnalSample:
    offset_sec: float
    latency_ms: float
    route: str
    cold: bool


def _replay_trace(gateway, trace: list[float], anchor: datetime) -> list[DiurnalSample]:
    from gateway.contract import InferenceRequest, RequestClass, TaskKind

    def _fire(index: int, offset: float) -> DiurnalSample:
        req = InferenceRequest(
            request_id=f"diurnal_{index}", family_id="fam_riveras", requesting_member_id="m_mom",
            task=TaskKind.family_digest, request_class=RequestClass.interactive,
            query=f"What changed today for my family? (#{index})", policy_version="policy-2026-06-18",
        )
        resp = gateway.infer(req, anchor).response
        return DiurnalSample(offset, resp.latency_ms, resp.route.value,
                             resp.latency_ms > COLD_LATENCY_THRESHOLD_MS)

    start = time.perf_counter()
    samples: list[DiurnalSample] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = []
        for index, offset in enumerate(trace):
            sleep_for = offset - (time.perf_counter() - start)
            if sleep_for > 0:
                time.sleep(sleep_for)
            futures.append(pool.submit(_fire, index, offset))
        for future in futures:
            samples.append(future.result())
    return samples


def run_diurnal(args: argparse.Namespace) -> int:
    base, _headers, _metrics = _resolve_endpoint()
    from gateway.composition import build_gateway

    anchor = datetime(2026, 6, 18, 17, 0, tzinfo=timezone.utc)
    gateway = build_gateway(anchor, "state/loadtest_diurnal_audit.jsonl", offline=False)
    trace = _diurnal_trace(args.minutes, args.peak_rps, args.base_rps)
    print(f"load-test DIURNAL -> {base}  {len(trace)} arrivals over {args.minutes:.1f} min "
          f"(base {args.base_rps}/s -> peak {args.peak_rps}/s)")
    samples = _replay_trace(gateway, trace, anchor)
    served = [s for s in samples if s.route == "self_hosted"]
    cold = [s for s in served if s.cold]
    frac = round(len(cold) / len(served), 4) if served else 0.0
    payload = {
        "mode": "diurnal", "arrivals": len(trace), "self_hosted_served": len(served),
        "cold_served": len(cold), "rampup_cold_start_fraction_measured": frac,
        "cold_latency_threshold_ms": COLD_LATENCY_THRESHOLD_MS,
        "minutes": args.minutes, "base_rps": args.base_rps, "peak_rps": args.peak_rps,
        "routes": sorted({s.route for s in samples}),
    }
    digest = _freeze(payload, args.out)
    print(f"  self-hosted served={len(served)} cold={len(cold)} "
          f"measured ramp-up cold-start fraction={frac:.2%}")
    print(f"  frozen: {args.out}\n  sha256(capture) = {digest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live load tester for the self-hosted vLLM route")
    sub = parser.add_subparsers(dest="command", required=True)
    sweep = sub.add_parser("sweep", help="Unknown 1: sustainable batch-width concurrency sweep")
    sweep.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC)
    sweep.add_argument("--out", default="state/loadtest_sweep_capture.json")
    sweep.set_defaults(func=run_sweep)
    curve = sub.add_parser("ctxcurve", help="Context-size sweep: sustainable batch per context size")
    curve.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC)
    curve.add_argument("--context-tokens", type=int, nargs="+",
                       default=list(_CTX_CURVE_TARGETS), dest="context_tokens")
    curve.add_argument("--out", default="state/loadtest_ctxcurve_capture.json")
    curve.set_defaults(func=run_ctxcurve)
    diurnal = sub.add_parser("diurnal", help="Tier B: diurnal replay through the gateway")
    diurnal.add_argument("--minutes", type=float, default=6.0)
    diurnal.add_argument("--base-rps", type=float, default=2.0, dest="base_rps")
    diurnal.add_argument("--peak-rps", type=float, default=8.0, dest="peak_rps")
    diurnal.add_argument("--out", default="state/loadtest_diurnal_capture.json")
    diurnal.set_defaults(func=run_diurnal)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
