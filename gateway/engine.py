"""The gateway composite: assemble → route → run → cost → audit.

Single entry point `Gateway.infer`. Pure-logic primitives (assembly, routing,
cost) are composed here; the only side effects (audit write) are delegated to the
declared boundary module. The assembler is injectable so the eval gate can swap
in a deliberately-leaky assembler to prove the gate blocks a regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from context.assembly import AssembledContext, assemble_context
from context.schema import ContextItem
from gateway.contract import (
    InferenceRequest,
    InferenceResponse,
    Route,
)
from gateway.router import Router, decide_route
from serving.clients import CompletionResult
from telemetry.cost import cost_for

Assembler = Callable[..., AssembledContext]

_SYSTEM_PROMPT = (
    "You answer questions about a family using only the context provided. "
    "Never reveal information not present in the context. Be concise."
)


@dataclass
class InferenceOutcome:
    response: InferenceResponse
    context: AssembledContext
    cost_route: Route


class Gateway:
    def __init__(
        self,
        *,
        candidates: list[ContextItem],
        router: Router,
        audit_append: Callable[[InferenceRequest, InferenceResponse, datetime], str],
        assembler: Assembler = assemble_context,
    ):
        self._candidates = candidates
        self._router = router
        self._audit_append = audit_append
        self._assembler = assembler

    def _deterministic(self, ctx: AssembledContext) -> CompletionResult:
        text = "No shareable updates are available for you right now."
        if ctx.included_items:
            lines = "\n".join(f"- {item.content}" for item in ctx.included_items)
            text = f"{ctx.included_count} update(s) for you today:\n{lines}"
        return CompletionResult(
            text=text, model_used="template", prompt_tokens=0,
            completion_tokens=0, latency_ms=1.0, cold_start=False, simulated=True,
        )

    def infer(self, req: InferenceRequest, now_utc: datetime) -> InferenceOutcome:
        ctx = self._assembler(
            candidates=self._candidates,
            requesting_member_id=req.requesting_member_id,
            task=req.task,
            policy_version=req.policy_version,
            now_utc=now_utc,
        )
        route = decide_route(req, ctx, self._router.health())
        if route == Route.deterministic_fallback:
            result, final_route = self._deterministic(ctx), Route.deterministic_fallback
        else:
            outcome = self._router.run(route, _SYSTEM_PROMPT, ctx.prompt_text)
            result, final_route = outcome.result, outcome.route

        breakdown = cost_for(final_route, result)
        response = InferenceResponse(
            request_id=req.request_id,
            text=result.text,
            route=final_route,
            model_used=result.model_used,
            latency_ms=result.latency_ms,
            cost_usd=breakdown.total_usd,
            context_used=ctx.refs,
            eval_flags=["simulated_model"] if result.simulated else [],
        )
        self._audit_append(req, response, now_utc)
        return InferenceOutcome(response=response, context=ctx, cost_route=final_route)
