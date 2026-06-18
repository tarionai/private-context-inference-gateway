"""FastAPI inference gateway — the typed /infer boundary (Slice 0/1).

Validates an InferenceRequest, runs the privacy-filtered assembly + routing +
cost + audit pipeline, and returns the typed, audited InferenceResponse. The
clock read is confined here (a boundary). Run:

    uvicorn gateway.app:app --reload
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI

from gateway.composition import build_gateway
from gateway.contract import InferenceRequest, InferenceResponse

_AUDIT_PATH = os.environ.get("AUDIT_PATH", "state/gateway_audit.jsonl")
_OFFLINE = os.environ.get("GATEWAY_OFFLINE", "0") == "1"

app = FastAPI(title="Private Context Inference Gateway", version="0.1.0")


def _gateway():
    # Built per process start; family graph + audit store are stable for the run.
    return build_gateway(datetime.now(timezone.utc), _AUDIT_PATH, offline=_OFFLINE)


_GATEWAY = _gateway()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    outcome = _GATEWAY.infer(request, datetime.now(timezone.utc))
    return outcome.response
