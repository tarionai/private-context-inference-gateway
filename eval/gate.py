"""Regression gate. Blocks the release on any hard privacy violation.

Runs every product scenario through the gateway (offline deterministic clients,
no GPU/keys), applies the deterministic leakage asserts, and fails closed. The
demonstrable artifact: run with `--inject-leak` to swap in the deliberately
leaky assembler and watch the gate BLOCK it (non-zero exit). That blocked run is
worth more than the passing one.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from audit.store import AuditStore
from context.assembly import assemble_context
from data.synthetic_family import build_family
from eval.leaky_assembly import leaky_assemble_context
from eval.privacy_asserts import Violation, check_no_leakage
from gateway.contract import InferenceRequest, RequestClass, Route, TaskKind
from gateway.engine import Gateway
from gateway.router import Router
from serving.clients import DeterministicClient

ANCHOR = datetime(2026, 6, 18, 17, 0, tzinfo=timezone.utc)
POLICY_VERSION = "policy-2026-06-18"
_SCENARIOS = Path(__file__).parent / "scenarios" / "product_scenarios.json"


@dataclass
class ScenarioResult:
    scenario_id: str
    route: str
    route_expected: bool
    violations: list[Violation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


@dataclass
class GateResult:
    results: list[ScenarioResult]

    @property
    def blocked(self) -> bool:
        return any(not r.passed for r in self.results)

    @property
    def total_violations(self) -> int:
        return sum(len(r.violations) for r in self.results)


def _offline_router() -> Router:
    det = DeterministicClient()
    return Router(self_hosted=det, hosted_fast=det, hosted_strong=det, deterministic=det)


def _build_gateway(audit_path: Path, *, inject_leak: bool) -> tuple[Gateway, list]:
    family = build_family(ANCHOR)
    store = AuditStore(audit_path)
    assembler = leaky_assemble_context if inject_leak else assemble_context
    gateway = Gateway(
        candidates=family.items,
        router=_offline_router(),
        audit_append=store.append,
        assembler=assembler,
    )
    return gateway, family.items


def run_gate(audit_path: Path, *, inject_leak: bool = False) -> GateResult:
    gateway, candidates = _build_gateway(audit_path, inject_leak=inject_leak)
    scenarios = json.loads(_SCENARIOS.read_text(encoding="utf-8"))
    results: list[ScenarioResult] = []
    for spec in scenarios:
        task = TaskKind(spec["task"])
        req = InferenceRequest(
            request_id=f"eval_{spec['id']}",
            family_id="fam_riveras",
            requesting_member_id=spec["requesting_member_id"],
            task=task,
            request_class=RequestClass(spec["request_class"]),
            query=spec["query"],
            policy_version=POLICY_VERSION,
        )
        outcome = gateway.infer(req, ANCHOR)
        violations = check_no_leakage(
            outcome,
            candidates=candidates,
            requesting_member_id=req.requesting_member_id,
            task=task,
            now_utc=ANCHOR,
        )
        results.append(
            ScenarioResult(
                scenario_id=spec["id"],
                route=outcome.response.route.value,
                route_expected=outcome.response.route.value in spec["expect_routes"],
                violations=violations,
            )
        )
    return GateResult(results=results)


def _print_report(result: GateResult) -> None:
    for r in result.results:
        status = "PASS" if r.passed else "BLOCK"
        print(f"  [{status}] {r.scenario_id:24s} route={r.route}")
        for v in r.violations:
            print(f"          ! {v.kind} {v.item_id}: {v.detail}")
    verdict = "BLOCKED (release denied)" if result.blocked else "PASSED"
    print(f"\nEval gate: {verdict} - {result.total_violations} privacy violation(s)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Privacy regression gate")
    parser.add_argument("--inject-leak", action="store_true", help="Swap in the leaky assembler to prove the gate blocks it")
    parser.add_argument("--audit", default="state/eval_audit.jsonl")
    args = parser.parse_args()
    result = run_gate(Path(args.audit), inject_leak=args.inject_leak)
    _print_report(result)
    return 1 if result.blocked else 0


if __name__ == "__main__":
    sys.exit(main())
