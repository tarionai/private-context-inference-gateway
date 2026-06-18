"""The single, fully-audited authorization decision.

`can_member_see` is the only place that decides whether a context item may enter
a prompt for a requesting member. It is pure (no I/O, no clock read passed in
from outside) and returns the exclusion reason on denial so every rejection is
auditable. This is privacy enforced *before* prompt assembly — the differentiator.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from context.schema import ConsentState, ContextItem

_RETENTION_DAYS = re.compile(r"^(\d+)d$")


def _parse_timestamp(raw: str) -> datetime:
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _expired(item: ContextItem, now_utc: datetime) -> bool:
    policy = item.retention_policy
    if policy in ("never", "session"):
        return False
    match = _RETENTION_DAYS.match(policy)
    if not match:
        # Unknown retention policy fails closed: treat as expired.
        return True
    horizon = _parse_timestamp(item.timestamp) + timedelta(days=int(match.group(1)))
    return now_utc > horizon


def can_member_see(
    item: ContextItem, member_id: str, now_utc: datetime
) -> tuple[bool, str | None]:
    """Return (allowed, exclusion_reason). reason is None iff allowed."""
    if member_id not in item.visible_to:
        return False, "not_in_visible_to"
    if item.consent_state != ConsentState.granted:
        return False, f"consent_{item.consent_state.value}"
    if not item.prompt_eligible:
        return False, "not_prompt_eligible"
    if _expired(item, now_utc):
        return False, "retention_expired"
    return True, None
