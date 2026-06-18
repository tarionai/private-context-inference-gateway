"""Wiring: build a Gateway + Router from the environment.

Real clients are used when their credentials are present (MODAL_BASE_URL for the
self-hosted route, ANTHROPIC_API_KEY for hosted routes); otherwise credential-free
stand-ins keep the gateway runnable offline. This is the only module that decides
which concrete clients back each route.
"""

from __future__ import annotations

import os
from pathlib import Path

from audit.store import AuditStore
from data.synthetic_family import build_family
from datetime import datetime
from gateway.engine import Gateway
from gateway.router import Router
from serving.clients import (
    DeterministicClient,
    HostedAnthropicClient,
    LLMClient,
    SimulatedSelfHostedClient,
    self_hosted_from_env,
)


def _hosted(model_id: str, name: str) -> LLMClient:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return HostedAnthropicClient(model_id=model_id, name=name)
    return DeterministicClient(f"{name}-simulated")


def build_router(*, offline: bool = False) -> Router:
    if offline:
        sh: LLMClient = SimulatedSelfHostedClient()
        fast: LLMClient = DeterministicClient("hosted_fast-simulated")
        strong: LLMClient = DeterministicClient("hosted_strong-simulated")
    else:
        sh = self_hosted_from_env()
        fast = _hosted("claude-haiku-4-5-20251001", "hosted_fast")
        strong = _hosted("claude-sonnet-4-6", "hosted_strong")
    return Router(
        self_hosted=sh, hosted_fast=fast, hosted_strong=strong,
        deterministic=DeterministicClient(),
    )


def build_gateway(anchor: datetime, audit_path: str | Path, *, offline: bool = False) -> Gateway:
    family = build_family(anchor)
    store = AuditStore(audit_path)
    return Gateway(
        candidates=family.items,
        router=build_router(offline=offline),
        audit_append=store.append,
    )
