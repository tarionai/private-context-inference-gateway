"""Privacy-filtered context assembly.

Every candidate item is run through `can_member_see`. Survivors enter the prompt;
every exclusion is recorded as a `ContextRef` with its reason. A second gate
applies to generic-summary tasks: an item flagged `summarizable=False` must not
enter a shared summary even when the requester is otherwise authorized — this is
leakage class 2 (child-sensitive data leaking into generic summarization).

Pure transformation: it reads a clock value passed in, performs no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from context.authorization import can_member_see
from context.schema import ContextItem
from gateway.contract import ContextRef, TaskKind

# Tasks that produce a shared/generic summary rather than a member-targeted answer.
_SUMMARY_TASKS = frozenset({TaskKind.family_digest})


@dataclass(frozen=True)
class AssembledContext:
    prompt_text: str                 # the privacy-safe text handed to the model
    refs: list[ContextRef]           # full audit: every included AND excluded item
    included_items: list[ContextItem]  # survivors, for deterministic fallback rendering

    @property
    def included_count(self) -> int:
        return sum(1 for ref in self.refs if ref.included)

    @property
    def excluded_count(self) -> int:
        return sum(1 for ref in self.refs if not ref.included)


def assemble_context(
    *,
    candidates: list[ContextItem],
    requesting_member_id: str,
    task: TaskKind,
    policy_version: str,
    now_utc: datetime,
) -> AssembledContext:
    refs: list[ContextRef] = []
    survivors: list[ContextItem] = []
    is_summary = task in _SUMMARY_TASKS

    for item in candidates:
        allowed, reason = can_member_see(item, requesting_member_id, now_utc)
        if allowed and is_summary and not item.summarizable:
            allowed, reason = False, "not_summarizable"
        refs.append(
            ContextRef(
                item_id=item.item_id,
                subject_id=item.subject_id,
                sensitivity=item.sensitivity.value,
                included=allowed,
                exclusion_reason=reason,
                policy_version=policy_version,
                source_hash=item.source_hash(),
            )
        )
        if allowed:
            survivors.append(item)

    prompt_text = "\n".join(
        f"- [{item.subject_id}] {item.content}" for item in survivors
    )
    return AssembledContext(prompt_text=prompt_text, refs=refs, included_items=survivors)
