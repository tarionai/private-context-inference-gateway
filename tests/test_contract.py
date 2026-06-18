from gateway.contract import (
    ContextRef,
    InferenceRequest,
    InferenceResponse,
    RequestClass,
    Route,
    TaskKind,
)


def test_request_roundtrips_and_defaults():
    req = InferenceRequest(
        request_id="r1",
        family_id="fam1",
        requesting_member_id="m1",
        task=TaskKind.family_digest,
        request_class=RequestClass.interactive,
        query="what changed today",
        policy_version="policy-2026-06-18",
    )
    assert req.schema_version == "v1"
    assert req.trace_id is None
    dumped = req.model_dump_json()
    assert InferenceRequest.model_validate_json(dumped) == req


def test_response_carries_full_audit():
    resp = InferenceResponse(
        request_id="r1",
        text="3 updates today",
        route=Route.deterministic_fallback,
        model_used="template",
        latency_ms=1.2,
        cost_usd=0.0,
        context_used=[
            ContextRef(
                item_id="i1",
                subject_id="teen",
                sensitivity="sensitive",
                included=False,
                exclusion_reason="not_in_visible_to",
                policy_version="policy-2026-06-18",
                source_hash="abc",
            )
        ],
        eval_flags=[],
    )
    assert resp.route is Route.deterministic_fallback
    assert resp.context_used[0].included is False
    assert resp.context_used[0].exclusion_reason == "not_in_visible_to"
