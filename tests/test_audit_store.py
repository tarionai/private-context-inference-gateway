from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from audit.store import AuditStore
from gateway.contract import (
    ContextRef,
    InferenceRequest,
    InferenceResponse,
    RequestClass,
    Route,
    TaskKind,
)

ANCHOR = datetime(2026, 6, 18, 17, 0, tzinfo=timezone.utc)


def _req():
    return InferenceRequest(
        request_id="r1",
        family_id="fam",
        requesting_member_id="m1",
        task=TaskKind.family_digest,
        request_class=RequestClass.interactive,
        query="q",
        policy_version="p1",
    )


def _resp():
    return InferenceResponse(
        request_id="r1",
        text="t",
        route=Route.deterministic_fallback,
        model_used="template",
        latency_ms=1.0,
        cost_usd=0.0,
        context_used=[
            ContextRef(
                item_id="i1",
                subject_id="s",
                sensitivity="normal",
                included=True,
                exclusion_reason=None,
                policy_version="p1",
                source_hash="abc",
            )
        ],
        eval_flags=[],
    )


def test_chain_verifies_and_detects_tampering(tmp_path):
    store = AuditStore(tmp_path / "audit.jsonl")
    store.append(_req(), _resp(), ANCHOR)
    store.append(_req(), _resp(), ANCHOR)
    intact, count = store.verify_chain()
    assert intact is True and count == 2

    # Tamper with a record body; chain must break.
    path = tmp_path / "audit.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = lines[0].replace('"text":"t"', '"text":"LEAKED"')
    path.write_text("\n".join([tampered, lines[1]]) + "\n", encoding="utf-8")
    intact_after, _ = store.verify_chain()
    assert intact_after is False


def test_concurrent_appends_keep_chain_intact(tmp_path):
    """Threaded serving appends concurrently; the chain must stay intact and lose no
    records (a load test corrupted the chain before the append lock)."""
    store = AuditStore(tmp_path / "audit.jsonl")
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(store.append, _req(), _resp(), ANCHOR) for _ in range(200)]
        hashes = [future.result() for future in futures]
    intact, count = store.verify_chain()
    assert intact is True
    assert count == 200
    assert len(set(hashes)) == 200  # every append produced a distinct chained hash
