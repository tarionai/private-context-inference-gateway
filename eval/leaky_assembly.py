"""A deliberately leaky assembler — the regression the gate must block.

This is the demonstrable artifact: swap this in for the real assembler and the
eval gate must FAIL (block the release). It mirrors a realistic bug — the
visibility check (`visible_to`) is dropped, so every consent-granted item is
admitted regardless of which member is asking. Never wired into production; used
only by the gate's blocking-regression demonstration.
"""

from __future__ import annotations

from datetime import datetime

from context.assembly import AssembledContext
from context.schema import ConsentState, ContextItem
from gateway.contract import ContextRef, TaskKind


def leaky_assemble_context(
    *,
    candidates: list[ContextItem],
    requesting_member_id: str,
    task: TaskKind,
    policy_version: str,
    now_utc: datetime,
) -> AssembledContext:
    refs: list[ContextRef] = []
    survivors: list[ContextItem] = []
    for item in candidates:
        # BUG: visibility (`visible_to`) is not checked — only consent is.
        admitted = item.consent_state == ConsentState.granted
        refs.append(
            ContextRef(
                item_id=item.item_id,
                subject_id=item.subject_id,
                sensitivity=item.sensitivity.value,
                included=admitted,
                exclusion_reason=None if admitted else "consent",
                policy_version=policy_version,
                source_hash=item.source_hash(),
            )
        )
        if admitted:
            survivors.append(item)
    prompt_text = "\n".join(f"- [{i.subject_id}] {i.content}" for i in survivors)
    return AssembledContext(prompt_text=prompt_text, refs=refs, included_items=survivors)
