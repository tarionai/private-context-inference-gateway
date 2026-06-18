"""Degradation ladder. On any route failure, step down predictably.

    self_hosted -> hosted_fast -> hosted_strong -> deterministic_fallback

The deterministic fallback is non-LLM and always available, so the ladder always
terminates in a graceful answer — never a raw error. Ties straight back to
"deterministic before agentic": the floor of the system is deterministic.
"""

from __future__ import annotations

from gateway.contract import Route

LADDER: list[Route] = [
    Route.self_hosted,
    Route.hosted_fast,
    Route.hosted_strong,
    Route.deterministic_fallback,
]


def next_route(failed: Route) -> Route:
    """Return the next route down the ladder from a failed one."""
    if failed not in LADDER:
        return Route.deterministic_fallback
    idx = LADDER.index(failed)
    return LADDER[min(idx + 1, len(LADDER) - 1)]
