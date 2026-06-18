"""Multi-principal context types — the proprietary-shaped (synthetic) data.

A ContextItem is one fact about a subject (a teen, a child, a pet, a tile, a
place). Whether it may enter an LLM prompt for a given requesting member is a
deterministic function of these fields — see context/authorization.py.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel


class Sensitivity(str, Enum):
    normal = "normal"
    sensitive = "sensitive"
    safety_critical = "safety_critical"


class ConsentState(str, Enum):
    granted = "granted"
    revoked = "revoked"
    pending = "pending"


class ContextItem(BaseModel):
    item_id: str
    subject_id: str                 # who/what the item is about
    owner_id: str
    visible_to: list[str]           # member ids permitted to see it
    sensitivity: Sensitivity
    retention_policy: str           # "30d" | "session" | "never" | "Nd"
    consent_state: ConsentState
    summarizable: bool              # may a generic summary include it
    prompt_eligible: bool           # may it ever enter an LLM prompt
    content: str                    # the synthetic payload
    source: str
    timestamp: str                  # UTC ISO-8601

    def source_hash(self) -> str:
        """Stable content hash — makes the audit verifiable."""
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        return digest[:16]


class FamilyMember(BaseModel):
    member_id: str
    display_name: str
    role: str                       # parent | teen | child | guardian
    family_id: str


class Family(BaseModel):
    family_id: str
    members: list[FamilyMember]
    items: list[ContextItem]
