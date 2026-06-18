"""`python -m evidence` - the one command that produces the whole verifiable set.

Prints, in order:
  1. one typed /infer response
  2. route = self_hosted proof (not only hosted Claude)
  3. a cold-start AND a warm latency sample (reported separately)
  4. per-route cost - active-serving AND amortized boot/idle (the honest figure)
  5. the context included/excluded table with reasons + policy_version
  6. the eval gate BLOCKING a deliberately leaky fixture

Runs offline by default (credential-free, reproducible) so a reviewer can verify
it in under 10 minutes. Simulated numbers are labelled `[SIMULATED]`; set
MODAL_BASE_URL to exercise the LIVE self-hosted route. A determinism hash over
the structured payload is printed last - re-running yields the same hash.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import subprocess

from audit.store import AuditStore
from eval.gate import run_gate
from gateway.composition import build_gateway
from gateway.contract import InferenceRequest, RequestClass, Route, TaskKind
from serving.clients import self_hosted_from_env
from telemetry.cost import SelfHostedCostConfig, cost_for, project_fleet_cost

ANCHOR = datetime(2026, 6, 18, 17, 0, tzinfo=timezone.utc)
POLICY_VERSION = "policy-2026-06-18"
STATE = Path("state")
_BASE_URL = os.environ.get("SELF_HOSTED_BASE_URL") or os.environ.get("MODAL_BASE_URL")
_LIVE = bool(_BASE_URL)
_LABEL = "LIVE" if _LIVE else "SIMULATED"
_SELF_HOSTED_MODEL = os.environ.get("SELF_HOSTED_MODEL", "family-7b")
_HOST_USD_PER_HOUR = float(os.environ.get("SELF_HOSTED_USD_PER_HOUR", "0.80"))
_OLLAMA_KEEPALIVE_SEC = 300.0  # Ollama default keep-alive == the real scale-to-zero idle window


def _unload_self_hosted() -> None:
    """Force a true cold start: unload the model so the next call pays load latency."""
    if "11434" not in (_BASE_URL or ""):
        return  # only the local Ollama path supports forced unload
    try:
        subprocess.run(["ollama", "stop", _SELF_HOSTED_MODEL], timeout=30,
                       capture_output=True, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def _rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _section_1_typed_response(audit_path: Path) -> dict:
    _rule("1. Typed /infer response (privacy-filtered, audited)")
    gateway = build_gateway(ANCHOR, audit_path, offline=not _LIVE)
    req = InferenceRequest(
        request_id="evidence_digest", family_id="fam_riveras",
        requesting_member_id="m_mom", task=TaskKind.family_digest,
        request_class=RequestClass.interactive, query="What changed today for my family?",
        policy_version=POLICY_VERSION,
    )
    outcome = gateway.infer(req, ANCHOR)
    resp = outcome.response
    print(f"  request_id : {resp.request_id}")
    print(f"  route      : {resp.route.value}")
    print(f"  model_used : {resp.model_used}")
    print(f"  cost_usd   : {resp.cost_usd}")
    print(f"  eval_flags : {resp.eval_flags}")
    print(f"  answer     : {resp.text.splitlines()[0] if resp.text else '(empty)'}")
    return {"outcome": outcome, "response": resp}


def _section_2_self_hosted_route(route: str) -> str:
    _rule("2. Self-hosted route proof (not only hosted Claude)")
    model = _SELF_HOSTED_MODEL if _LIVE else "family-7b (simulated)"
    print(f"  digest request routed to: {route}  [{_LABEL}]  model={model}")
    if _LIVE:
        print("  -> a real self-hosted model served this request, not hosted Claude.")
    return route


def _section_3_latency_split() -> dict:
    _rule("3. Cold-start vs warm latency (reported separately)")
    if _LIVE:
        _unload_self_hosted()  # force a true cold start (next call pays model load)
    client = self_hosted_from_env()
    cold = client.complete("Answer concisely.", "- [household] groceries restocked. Summarize.")
    warm = client.complete("Answer concisely.", "- [household] groceries restocked. Summarize.")
    print(f"  cold-start latency : {cold.latency_ms:>9.1f} ms  [{_LABEL}] cold_start={cold.cold_start}")
    print(f"  warm latency       : {warm.latency_ms:>9.1f} ms  [{_LABEL}] cold_start={warm.cold_start}")
    if warm.latency_ms > 0:
        print(f"  warm tokens/sec    : {warm.completion_tokens / (warm.latency_ms / 1000):>9.1f}")
    print(f"  tokens (warm)      : prompt={warm.prompt_tokens} completion={warm.completion_tokens}")
    return {"cold": cold, "warm": warm}


def _live_cost_config(cold, warm) -> SelfHostedCostConfig:
    measured_boot = max(0.0, (cold.latency_ms - warm.latency_ms) / 1000.0)
    return SelfHostedCostConfig(
        host_usd_per_sec=_HOST_USD_PER_HOUR / 3600.0,
        boot_billed_sec=measured_boot,
        idle_sec=_OLLAMA_KEEPALIVE_SEC,
        requests_per_window=3.0,
    )


def _section_4_cost(cold, warm) -> dict:
    _rule("4. Per-route cost - active serving AND amortized boot/idle")
    config = _live_cost_config(cold, warm) if _LIVE else None
    breakdown = cost_for(Route.self_hosted, warm, self_hosted_config=config)
    basis = (f"MEASURED: boot={(cold.latency_ms - warm.latency_ms)/1000:.1f}s, "
             f"active={warm.latency_ms/1000:.1f}s, idle=300s, host ${_HOST_USD_PER_HOUR:.2f}/hr, 3 req/window"
             if _LIVE else "ASSUMED: 30s boot + 120s idle, Modal L4 $0.80/hr, 3 req/window")
    print(f"  self-hosted active serving      : ${breakdown.active_serving_usd:.6f}/request")
    print(f"  self-hosted amortized boot/idle : ${breakdown.amortized_boot_idle_usd:.6f}/request")
    print(f"    ({basis})")
    fleet = project_fleet_cost(breakdown.headline_usd, requests_per_family_per_day=4, families=1_000_000)
    print(f"  cost/family/day                 : ${fleet['cost_per_family_day_usd']:.4f}")
    print(f"  projected 1M families/day       : ${fleet['projected_fleet_day_usd']:,.2f}")
    return {"breakdown": breakdown, "fleet": fleet}


def _section_5_context_table(outcome) -> None:
    _rule("5. Context included/excluded table (policy-stamped)")
    print(f"  {'item_id':22s} {'incl':4s} {'exclusion_reason':20s} {'policy_version':18s} hash")
    for ref in outcome.context.refs:
        incl = "yes" if ref.included else "no"
        reason = ref.exclusion_reason or "-"
        print(f"  {ref.item_id:22s} {incl:4s} {reason:20s} {ref.policy_version:18s} {ref.source_hash}")


def _section_6_eval_gate() -> dict:
    _rule("6. Eval gate BLOCKING a deliberately leaky fixture")
    clean = run_gate(STATE / "evidence_gate_clean.jsonl", inject_leak=False)
    leaked = run_gate(STATE / "evidence_gate_leak.jsonl", inject_leak=True)
    print(f"  clean assembler  : {'PASSED' if not clean.blocked else 'BLOCKED'} "
          f"({clean.total_violations} violations)")
    print(f"  leaky assembler  : {'BLOCKED (release denied)' if leaked.blocked else 'PASSED'} "
          f"({leaked.total_violations} violations)")
    print("  -> the gate blocks the leaky regression; a clean release passes.")
    return {"clean_blocked": clean.blocked, "leak_blocked": leaked.blocked,
            "leak_violations": leaked.total_violations}


def _verify_audit(audit_path: Path) -> tuple[bool, int]:
    _rule("Audit chain verification (tamper-evident, hash-chained)")
    intact, count = AuditStore(audit_path).verify_chain()
    print(f"  records: {count}  chain_intact: {intact}")
    return intact, count


def main() -> int:
    STATE.mkdir(exist_ok=True)
    audit_path = STATE / "evidence_audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()

    print(f"Private Context Inference Gateway - evidence run  [{_LABEL} mode]")
    # In LIVE mode the latency split runs FIRST (after a forced unload) so the
    # cold-start measurement is genuine, before any other call warms the model.
    s3 = _section_3_latency_split() if _LIVE else None
    s1 = _section_1_typed_response(audit_path)
    route = _section_2_self_hosted_route(s1["outcome"].response.route.value)
    if s3 is None:
        s3 = _section_3_latency_split()
    s4 = _section_4_cost(s3["cold"], s3["warm"])
    _section_5_context_table(s1["outcome"])
    s6 = _section_6_eval_gate()
    intact, count = _verify_audit(audit_path)

    payload = {
        "mode": _LABEL,
        "route": route,
        "self_hosted_model": s1["outcome"].response.model_used,
        "self_hosted_active_usd": s4["breakdown"].active_serving_usd,
        "self_hosted_amortized_usd": s4["breakdown"].amortized_boot_idle_usd,
        "fleet_day_usd": s4["fleet"]["projected_fleet_day_usd"],
        "included": sum(1 for r in s1["outcome"].context.refs if r.included),
        "excluded": sum(1 for r in s1["outcome"].context.refs if not r.included),
        "leak_blocked": s6["leak_blocked"],
        "leak_violations": s6["leak_violations"],
        "audit_chain_intact": intact,
    }
    if _LIVE:
        payload["cold_latency_ms"] = round(s3["cold"].latency_ms, 1)
        payload["warm_latency_ms"] = round(s3["warm"].latency_ms, 1)
        capture = STATE / "evidence_live_capture.json"
        capture.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        digest = hashlib.sha256(capture.read_bytes()).hexdigest()
        _rule("Hash-verifiable capture (LIVE timings vary; the frozen capture is hashed)")
        print(f"  frozen: {capture}")
        print(f"  sha256(capture) = {digest}")
    else:
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        _rule("Determinism hash (re-run yields the same hash in SIMULATED mode)")
        print(f"  sha256(evidence) = {digest}")

    ok = s6["leak_blocked"] and not s6["clean_blocked"] and intact and route == Route.self_hosted.value
    proven = "self-hosted serving + cost + privacy ALL LIVE" if _LIVE else "privacy+cost real; self-hosted SIMULATED"
    print(f"\nEVIDENCE {'COMPLETE' if ok else 'INCOMPLETE'} [{_LABEL} mode] - {proven}.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
