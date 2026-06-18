import os

os.environ["GATEWAY_OFFLINE"] = "1"
os.environ["AUDIT_PATH"] = "state/test_app_audit.jsonl"

from fastapi.testclient import TestClient  # noqa: E402

from gateway.app import app  # noqa: E402

client = TestClient(app)


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_infer_returns_typed_audited_response():
    body = {
        "request_id": "t1",
        "family_id": "fam_riveras",
        "requesting_member_id": "m_mom",
        "task": "family_digest",
        "request_class": "interactive",
        "query": "what changed today",
        "policy_version": "policy-2026-06-18",
    }
    resp = client.post("/infer", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_id"] == "t1"
    assert data["route"] in {"self_hosted", "hosted_fast", "deterministic_fallback"}
    # Full context audit is present, with at least one excluded item carrying a reason.
    assert any(not ref["included"] and ref["exclusion_reason"] for ref in data["context_used"])
    assert all(ref["policy_version"] == "policy-2026-06-18" for ref in data["context_used"])
