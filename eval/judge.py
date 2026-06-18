"""LLM judge — subjective product quality only (relevance, groundedness).

Privacy is NEVER judged by an LLM; it is a deterministic hard assert (see
privacy_asserts.py). This judge is optional, needs an Anthropic key, and is
bypassed when no key is configured — it never gates a release on its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeVerdict:
    available: bool
    grounded: bool | None
    relevant: bool | None
    note: str


def judge_quality(query: str, answer: str, context_text: str) -> JudgeVerdict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return JudgeVerdict(
            available=False, grounded=None, relevant=None,
            note="judge skipped: no ANTHROPIC_API_KEY (subjective quality not scored)",
        )
    from anthropic import Anthropic  # wrapped at boundary; lazy import

    client = Anthropic()
    prompt = (
        "You judge an assistant answer for a family-context app. Return JSON only: "
        '{"grounded": bool, "relevant": bool}. grounded = every claim is supported by '
        "CONTEXT. relevant = the answer addresses QUERY.\n\n"
        f"QUERY: {query}\n\nCONTEXT:\n{context_text}\n\nANSWER:\n{answer}"
    )
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in message.content if b.type == "text")
    import json

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return JudgeVerdict(True, None, None, f"judge returned non-JSON: {text[:80]}")
    return JudgeVerdict(
        available=True,
        grounded=bool(parsed.get("grounded")),
        relevant=bool(parsed.get("relevant")),
        note="judged by claude-haiku-4-5",
    )
