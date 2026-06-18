"""Model routing decision + execution.

`decide_route` is a pure function of the request, the assembled context, and a
health snapshot — fully testable with no clients. `Router.run` executes the chosen
route and walks the degradation ladder on failure. The self_hosted route is a
runtime input (health/saturation), never a baked-in constant: if the self-hosted
model is unhealthy or its quality is judged too low for interactive traffic, the
router routes elsewhere without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from context.assembly import AssembledContext
from gateway.contract import InferenceRequest, RequestClass, Route, TaskKind
from gateway.fallback import next_route
from serving.clients import CompletionResult, LLMClient

_HARD_TASKS = frozenset({TaskKind.notify_decision})


@dataclass(frozen=True)
class HealthSnapshot:
    self_hosted_healthy: bool
    self_hosted_saturated: bool


def classify_difficulty(req: InferenceRequest, ctx: AssembledContext) -> str:
    """Deterministic difficulty heuristic. easy | medium | hard."""
    if req.task in _HARD_TASKS:
        return "hard"
    if any(r.included and r.sensitivity == "safety_critical" for r in ctx.refs):
        return "hard"
    if req.task in (TaskKind.family_digest, TaskKind.lost_item):
        return "easy" if ctx.included_count <= 6 else "medium"
    return "medium"


def decide_route(
    req: InferenceRequest, ctx: AssembledContext, health: HealthSnapshot
) -> Route:
    if ctx.included_count == 0:
        return Route.deterministic_fallback

    difficulty = classify_difficulty(req, ctx)
    if difficulty == "hard":
        return Route.hosted_strong

    if (
        req.request_class == RequestClass.interactive
        and difficulty == "easy"
        and health.self_hosted_healthy
        and not health.self_hosted_saturated
    ):
        return Route.self_hosted

    if health.self_hosted_saturated or not health.self_hosted_healthy:
        return Route.hosted_fast
    return Route.self_hosted


@dataclass
class RouteOutcome:
    route: Route
    result: CompletionResult


class Router:
    """Resolves a Route to a client and executes it, walking the ladder on error."""

    def __init__(
        self,
        *,
        self_hosted: LLMClient,
        hosted_fast: LLMClient,
        hosted_strong: LLMClient,
        deterministic: LLMClient,
    ):
        self._clients = {
            Route.self_hosted: self_hosted,
            Route.hosted_fast: hosted_fast,
            Route.hosted_strong: hosted_strong,
            Route.deterministic_fallback: deterministic,
        }

    def health(self) -> HealthSnapshot:
        sh = self._clients[Route.self_hosted]
        return HealthSnapshot(
            self_hosted_healthy=sh.healthy(),
            self_hosted_saturated=sh.saturated(),
        )

    def run(self, route: Route, system: str, prompt: str) -> RouteOutcome:
        current = route
        last_error: Exception | None = None
        for _ in range(len(self._clients)):
            client = self._clients[current]
            try:
                return RouteOutcome(route=current, result=client.complete(system, prompt))
            except Exception as exc:  # noqa: BLE001 — ladder must catch all route failures
                last_error = exc
                stepped = next_route(current)
                if stepped == current:
                    break
                current = stepped
        raise RuntimeError(f"degradation ladder exhausted: {last_error}")
