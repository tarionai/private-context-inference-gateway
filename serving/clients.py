"""One uniform client interface over self-hosted and hosted models.

Because vLLM is OpenAI-compatible, the self-hosted path points an `openai` client
at the Modal URL; the hosted paths call Anthropic. All paths return the same
`CompletionResult`, so the router and gateway never branch on provider.

Live providers are wrapped at this boundary (never leaked into core logic). A
`DeterministicClient` provides a credential-free, network-free path used both as
the genuine deterministic fallback and as a clearly-labeled stand-in when no live
endpoint is configured, so the evidence run is runnable offline.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.DOTALL)


def _strip_reasoning(text: str) -> str:
    """Reasoning models (deepseek-r1, qwen-r) emit <think>..</think> scratch.

    Removes complete blocks, then any unterminated trailing <think> (the model
    ran out of budget mid-thought) so a raw '<think>' never reaches the user.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _UNCLOSED_THINK.sub("", cleaned)
    return cleaned.strip()


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cold_start: bool
    simulated: bool = False


class LLMClient(Protocol):
    name: str

    def healthy(self) -> bool: ...
    def saturated(self) -> bool: ...
    def complete(self, system: str, prompt: str) -> CompletionResult: ...


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class DeterministicClient:
    """Network-free templated responder. Genuine fallback + offline stand-in."""

    def __init__(self, model_label: str = "deterministic-template"):
        self.name = model_label

    def healthy(self) -> bool:
        return True

    def saturated(self) -> bool:
        return False

    def complete(self, system: str, prompt: str) -> CompletionResult:
        lines = [ln for ln in prompt.splitlines() if ln.strip().startswith("-")]
        if lines:
            body = f"{len(lines)} update(s) in your family today:\n" + "\n".join(lines)
        else:
            body = "No shareable updates are available for you right now."
        return CompletionResult(
            text=body,
            model_used=self.name,
            prompt_tokens=_approx_tokens(prompt),
            completion_tokens=_approx_tokens(body),
            latency_ms=1.0,
            cold_start=False,
            simulated=True,
        )


class SimulatedSelfHostedClient:
    """Stand-in for the live vLLM/Modal route when no endpoint is configured.

    Reproduces the cold-start vs warm distinction deterministically so the
    evidence run can show the routing + latency split offline. Every result is
    marked `simulated=True` so no claim is made on simulated numbers.
    """

    def __init__(self, model_label: str = "family-7b"):
        self.name = model_label
        self._warmed = False

    def healthy(self) -> bool:
        return True

    def saturated(self) -> bool:
        return False

    def complete(self, system: str, prompt: str) -> CompletionResult:
        cold = not self._warmed
        self._warmed = True
        latency = 9000.0 if cold else 320.0  # illustrative split only
        inner = DeterministicClient(self.name).complete(system, prompt)
        return CompletionResult(
            text=inner.text,
            model_used=self.name,
            prompt_tokens=inner.prompt_tokens,
            completion_tokens=inner.completion_tokens,
            latency_ms=latency,
            cold_start=cold,
            simulated=True,
        )


class SelfHostedOpenAIClient:
    """Live self-hosted route: an OpenAI-compatible client pointed at a server I host.

    Works against any OpenAI-compatible self-hosted endpoint — vLLM on Modal
    (production), or a local Ollama serve (`http://localhost:11434/v1`) for a
    credential-free live proof. Reasoning-model scratch (<think>..</think>) is
    stripped so the returned text is the answer. `cold_start` marks the first
    call after construction, which pays model-load latency.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str = "family-7b",
        api_key: str = "x",
        max_tokens: int = 256,
        timeout_sec: float = 900.0,
        extra_headers: dict[str, str] | None = None,
    ):
        from openai import OpenAI  # wrapped at the boundary; lazy import

        self.name = model_name
        # Generous timeout: the first serverless cold start downloads model weights.
        # extra_headers carries Modal proxy-auth (Modal-Key / Modal-Secret) when
        # the endpoint is locked with requires_proxy_auth=True.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_sec,
            default_headers=extra_headers or None,
        )
        self._max_tokens = max_tokens
        self._warmed = False

    def healthy(self) -> bool:
        return True

    def saturated(self) -> bool:
        return False

    def complete(self, system: str, prompt: str) -> CompletionResult:
        cold = not self._warmed
        started = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=self._max_tokens,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._warmed = True
        usage = completion.usage
        raw = completion.choices[0].message.content or ""
        return CompletionResult(
            text=_strip_reasoning(raw) or "(reasoning-model output; answer budget exhausted)",
            model_used=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", _approx_tokens(prompt)),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            latency_ms=latency_ms,
            cold_start=cold,
            simulated=False,
        )


class HostedAnthropicClient:
    """Hosted route (fast or strong) — Anthropic, wrapped at the boundary."""

    def __init__(self, model_id: str, name: str):
        from anthropic import Anthropic  # lazy import; needs ANTHROPIC_API_KEY

        self.name = name
        self.model_id = model_id
        self._client = Anthropic()

    def healthy(self) -> bool:
        return True

    def saturated(self) -> bool:
        return False

    def complete(self, system: str, prompt: str) -> CompletionResult:
        started = time.perf_counter()
        message = self._client.messages.create(
            model=self.model_id,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        text = "".join(block.text for block in message.content if block.type == "text")
        return CompletionResult(
            text=text,
            model_used=self.model_id,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
            latency_ms=latency_ms,
            cold_start=False,
            simulated=False,
        )


def self_hosted_from_env() -> LLMClient:
    """Live self-hosted client if an endpoint is configured; else the simulated stand-in.

    Honors SELF_HOSTED_BASE_URL (local Ollama / any OpenAI-compatible serve) or
    MODAL_BASE_URL (production vLLM). When neither is set, returns the labelled
    simulated stand-in so the gateway stays runnable offline.
    """
    base_url = os.environ.get("SELF_HOSTED_BASE_URL") or os.environ.get("MODAL_BASE_URL")
    if base_url:
        headers: dict[str, str] = {}
        if os.environ.get("MODAL_KEY") and os.environ.get("MODAL_SECRET"):
            headers = {
                "Modal-Key": os.environ["MODAL_KEY"],
                "Modal-Secret": os.environ["MODAL_SECRET"],
            }
        return SelfHostedOpenAIClient(
            base_url=base_url,
            model_name=os.environ.get("SELF_HOSTED_MODEL") or os.environ.get("MODAL_MODEL_NAME", "family-7b"),
            api_key=os.environ.get("SELF_HOSTED_API_KEY", "x"),
            max_tokens=int(os.environ.get("SELF_HOSTED_MAX_TOKENS", "256")),
            timeout_sec=float(os.environ.get("SELF_HOSTED_TIMEOUT", "900")),
            extra_headers=headers or None,
        )
    return SimulatedSelfHostedClient()
