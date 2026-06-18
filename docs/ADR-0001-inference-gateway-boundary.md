# ADR-0001 — The inference gateway wire contract is the fixed boundary

**Status:** Accepted
**Date:** 2026-06-18

## Context

Build-Sequence governance permits exactly one bottom-up move: fix the single
external wire/boundary contract first, because it is the one primitive whose
interface is knowable in advance — *we* designed the boundary. Everything after
it is a vertical slice built simplest-demonstrable-first, where the consumer is
the spec.

This artifact serves a self-hosted model, routes to hosted on cost/quality,
enforces per-principal visibility before context assembly, gates releases on a
product-shaped eval suite, and reports a per-route cost figure. Those slices all
need a shared, stable type to plug into. Their *internal* interfaces (ontology
shape, policy conditions, router thresholds, cost-rollup fields) are empirically
discovered and must not be fixed ahead of their first consumer. The wire is
different: it is the seam between callers and the gateway, and it is designed,
not discovered.

## Decision

`gateway/contract.py` is the fixed boundary, committed before any application
logic. It defines:

- `TaskKind`, `RequestClass`, `Route` — closed enums (structure before semantics).
- `InferenceRequest` — what a caller sends, including `policy_version` (the
  governing privacy ruleset) and an explicit `latency_budget_ms` the router honors.
- `ContextRef` — one row of the context-assembly audit: included/excluded, the
  `exclusion_reason` when excluded, the `policy_version` that decided it, and a
  `source_hash` that makes the audit verifiable.
- `InferenceResponse` — the typed, audited result: `route`, `model_used`,
  `latency_ms`, `cost_usd`, the full `context_used` audit, and `eval_flags`.

The typed contract is committed and an ADR recorded **before** Spike -1 runs.
Spike -1 (Modal/vLLM proof) gates application logic — router, assembly,
dashboard, audit — **not** the contract. A red spike therefore never strands
the boundary work.

## policy_version scope (explicit non-goal for the weekend)

Every request and every `ContextRef` carries the *current* policy version (one
string). That makes the audit **structurally** replayable and lets us say
"policy-stamped". Building a policy-version store plus machinery to re-run a
historical request under its recorded ruleset is multi-day work and is **out of
scope** — stretch goal only. We record the field; we do not build historical
replay. The README and resume must say "policy-stamped", never "replayable",
until an actual replay command exists.

## Consequences

- Slices can be built and verified independently against a stable type surface.
- Changing the contract after slice 1 ships is a breaking change requiring a new
  `schema_version` and a follow-up ADR.
- The contract is intentionally complete (API-for-the-future): `trace_id`,
  `latency_budget_ms`, and `source_hash` are present even though the simplest
  slice does not populate all of them — an incomplete API is a trap, not a draft.
