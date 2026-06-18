from datetime import datetime, timezone

from context.assembly import assemble_context
from data.synthetic_family import MOM, build_family
from gateway.contract import (
    InferenceRequest,
    RequestClass,
    Route,
    TaskKind,
)
from gateway.fallback import LADDER, next_route
from gateway.router import HealthSnapshot, Router, decide_route
from serving.clients import DeterministicClient

ANCHOR = datetime(2026, 6, 18, 17, 0, tzinfo=timezone.utc)
FAM = build_family(ANCHOR)


def _req(task, request_class):
    return InferenceRequest(
        request_id="r1",
        family_id=FAM.family_id,
        requesting_member_id=MOM,
        task=task,
        request_class=request_class,
        query="q",
        policy_version="p1",
    )


def _ctx(task, member=MOM):
    return assemble_context(
        candidates=FAM.items,
        requesting_member_id=member,
        task=task,
        policy_version="p1",
        now_utc=ANCHOR,
    )


HEALTHY = HealthSnapshot(self_hosted_healthy=True, self_hosted_saturated=False)
SATURATED = HealthSnapshot(self_hosted_healthy=True, self_hosted_saturated=True)
DOWN = HealthSnapshot(self_hosted_healthy=False, self_hosted_saturated=False)


def test_easy_interactive_goes_self_hosted_when_healthy():
    # family_digest for MOM includes only the normal grocery item -> easy, no
    # safety-critical content to escalate.
    ctx = _ctx(TaskKind.family_digest)
    route = decide_route(_req(TaskKind.family_digest, RequestClass.interactive), ctx, HEALTHY)
    assert route is Route.self_hosted


def test_saturated_self_hosted_falls_to_hosted_fast():
    ctx = _ctx(TaskKind.family_digest)
    route = decide_route(_req(TaskKind.family_digest, RequestClass.interactive), ctx, SATURATED)
    assert route is Route.hosted_fast


def test_safety_critical_context_escalates_to_strong():
    # lost_item is normally easy, but an included safety_critical item escalates.
    ctx = _ctx(TaskKind.lost_item)
    assert any(r.included and r.sensitivity == "safety_critical" for r in ctx.refs)
    route = decide_route(_req(TaskKind.lost_item, RequestClass.interactive), ctx, HEALTHY)
    assert route is Route.hosted_strong


def test_hard_task_goes_hosted_strong():
    ctx = _ctx(TaskKind.notify_decision)
    route = decide_route(_req(TaskKind.notify_decision, RequestClass.interactive), ctx, HEALTHY)
    assert route is Route.hosted_strong


def test_no_eligible_context_goes_deterministic_fallback():
    # A member with no visible items on a summary task -> nothing eligible.
    ctx = _ctx(TaskKind.family_digest, member="stranger")
    route = decide_route(_req(TaskKind.family_digest, RequestClass.interactive), ctx, HEALTHY)
    assert route is Route.deterministic_fallback


def test_ladder_order_and_termination():
    assert LADDER[0] is Route.self_hosted
    assert next_route(Route.self_hosted) is Route.hosted_fast
    assert next_route(Route.deterministic_fallback) is Route.deterministic_fallback


def test_router_walks_ladder_on_failure():
    class Boom:
        name = "boom"

        def healthy(self):
            return True

        def saturated(self):
            return False

        def complete(self, system, prompt):
            raise RuntimeError("route down")

    router = Router(
        self_hosted=Boom(),
        hosted_fast=Boom(),
        hosted_strong=Boom(),
        deterministic=DeterministicClient(),
    )
    outcome = router.run(Route.self_hosted, "sys", "- [x] hi")
    assert outcome.route is Route.deterministic_fallback
    assert "update" in outcome.result.text
