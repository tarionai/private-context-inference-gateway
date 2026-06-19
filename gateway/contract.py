"""The typed inference gateway wire contract.

This is the one knowable-in-advance boundary (Build-Sequence governance): a
designed wire fixed up front, depending on nothing. Every vertical slice plugs
into these types. Do not change casually after slice 1 ships.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class TaskKind(str, Enum):
    family_digest = "family_digest"      # "what changed today for my family"
    notify_decision = "notify_decision"  # "should I notify a parent"
    lost_item = "lost_item"              # "where is the lost item likely"
    context_persist = "context_persist"  # "what should be persisted/discarded"
    freeform = "freeform"


class RequestClass(str, Enum):
    interactive = "interactive"  # tight latency budget, user is waiting
    batch = "batch"              # nightly digest, latency-relaxed
    background = "background"     # pre-computation, no user waiting


class Route(str, Enum):
    cache = "cache"
    self_hosted = "self_hosted"
    hosted_fast = "hosted_fast"
    hosted_strong = "hosted_strong"
    deterministic_fallback = "deterministic_fallback"


class InferenceRequest(BaseModel):
    request_id: str
    family_id: str
    requesting_member_id: str
    task: TaskKind
    request_class: RequestClass
    query: str
    policy_version: str                   # which privacy ruleset governs this request
    schema_version: str = "v1"
    trace_id: str | None = None           # observability join key
    latency_budget_ms: int | None = None  # explicit budget the router honors


class ContextRef(BaseModel):
    item_id: str
    subject_id: str
    sensitivity: str
    included: bool
    exclusion_reason: str | None          # why a member could NOT see it — audited
    policy_version: str                   # which ruleset admitted/excluded this item
    source_hash: str | None = None        # content hash — makes the audit verifiable


class InferenceResponse(BaseModel):
    request_id: str
    text: str
    route: Route
    model_used: str
    latency_ms: float
    cost_usd: float
    context_used: list[ContextRef]        # full audit of what entered/was-excluded
    eval_flags: list[str]
    # Observability for the cost curve (counts only, never content). included_context_tokens
    # is the privacy-filtered assembled context (the assembly lever you control); prompt_tokens
    # is the full prompt the model KV-caches (the batch-width driver). Default 0 = not recorded.
    included_context_tokens: int = 0
    prompt_tokens: int = 0
