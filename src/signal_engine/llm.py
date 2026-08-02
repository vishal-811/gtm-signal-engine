"""Thin wrapper around the OpenAI SDK.

Every model call in the pipeline goes through :func:`structured_call`, so this
is the only module that knows which provider is in use. Swapping providers
means rewriting this file and nothing else.

Four concerns are centralized here:

* **Structured output** — responses are validated against a pydantic model by
  ``client.chat.completions.parse()``, so no stage parses or repairs JSON.
* **Prompt caching** — OpenAI caches automatically at or above 1024 prompt
  tokens; there is no per-block opt-in to set. What matters is that the system
  prompt stays byte-identical across calls, which is why prompts are loaded
  from files and never interpolated with per-request values. Cache behaviour is
  asserted, not assumed.
* **Refusals** — a refused structured-output call returns HTTP 200 with
  ``message.refusal`` set and ``message.parsed`` as None. Reading ``.parsed``
  without checking would crash the run on an unremarkable input.
* **Cost** — token usage is accumulated per run and written to the `runs` sheet
  tab. Dollar figures require per-model rates, which you set in .env.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import openai
from pydantic import BaseModel

from .config import settings

log = logging.getLogger(__name__)

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# The pipeline's five effort levels collapse onto OpenAI's three. Both "xhigh"
# and "max" map to "high" — OpenAI exposes no tier above it.
_EFFORT_MAP: dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}

T = TypeVar("T", bound=BaseModel)


class RefusalError(RuntimeError):
    """The model declined the request. Raised so the caller can skip one item
    rather than abort the whole run."""


@dataclass
class Usage:
    """Running token and cost totals for one pipeline run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    _cache_checked: bool = field(default=False, repr=False)

    def add(self, raw: Any) -> None:
        if raw is None:
            return
        self.calls += 1
        self.input_tokens += getattr(raw, "prompt_tokens", 0) or 0
        self.output_tokens += getattr(raw, "completion_tokens", 0) or 0
        details = getattr(raw, "prompt_tokens_details", None)
        if details is not None:
            self.cached_tokens += getattr(details, "cached_tokens", 0) or 0

    @property
    def cost_is_estimated(self) -> bool:
        cfg = settings()
        return bool(cfg.openai_input_cost_per_mtok or cfg.openai_output_cost_per_mtok)

    @property
    def cost_usd(self) -> float:
        """Estimated spend, or 0.0 when per-model rates are not configured.

        Rates live in .env because the model is configurable: hardcoding a
        price for a model the operator may have swapped out would report a
        confidently wrong number, which is worse than reporting none. Cached
        input bills at a discount — OpenAI's published rate is 25% of the
        uncached input price.
        """
        cfg = settings()
        if not self.cost_is_estimated:
            return 0.0
        uncached = max(self.input_tokens - self.cached_tokens, 0)
        return (
            uncached * cfg.openai_input_cost_per_mtok
            + self.cached_tokens * cfg.openai_input_cost_per_mtok * 0.25
            + self.output_tokens * cfg.openai_output_cost_per_mtok
        ) / 1_000_000

    def summary(self) -> str:
        money = (
            f" · ${self.cost_usd:.3f}"
            if self.cost_is_estimated
            else " · cost unknown (set OPENAI_*_COST_PER_MTOK)"
        )
        return (
            f"{self.calls} calls · in {self.input_tokens:,} "
            f"(cached {self.cached_tokens:,}) · out {self.output_tokens:,}{money}"
        )


# One tracker per process. The CLI resets it at the start of each run.
usage = Usage()


def reset_usage() -> None:
    global usage
    usage = Usage()


_client: openai.OpenAI | None = None

# Latched off after the first rejection so a non-reasoning model produces one
# warning per run instead of one per call.
_effort_supported = True


def reset_effort_probe() -> None:
    global _effort_supported
    _effort_supported = True


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        key = settings().openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add a "
                "key from https://platform.openai.com/api-keys"
            )
        # The SDK retries 429/5xx with backoff; 4 gives a daily batch job room
        # to ride out a transient overload without failing the whole run.
        _client = openai.OpenAI(api_key=key, max_retries=4)
    return _client


def model() -> str:
    return settings().openai_model


def _check_cache_health(raw_usage: Any, label: str) -> None:
    """Warn once per run when caching cannot possibly engage.

    OpenAI caches automatically above 1024 prompt tokens, so a prompt below
    that threshold will never hit — silently, and for the whole run.
    """
    if usage._cache_checked:
        return
    prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
    if prompt_tokens and prompt_tokens < 1024:
        log.warning(
            "%s prompt is %d tokens, under OpenAI's 1024-token caching "
            "threshold — caching will never engage for this stage.",
            label,
            prompt_tokens,
        )
    usage._cache_checked = True


def _is_unsupported_effort_error(exc: Exception) -> bool:
    """Does this error mean the chosen model has no reasoning_effort knob?

    Matched on message text rather than on a model-name allowlist: OpenAI
    renames and retires models faster than any hardcoded list stays correct,
    and a stale list would silently drop the effort setting on a model that
    does support it.
    """
    text = str(exc).lower()
    return "reasoning_effort" in text and (
        "unsupported" in text or "not supported" in text or "unknown" in text
    )


def structured_call(
    *,
    system: str,
    user: str,
    schema: type[T],
    effort: Effort = "medium",
    max_tokens: int = 8000,
    label: str = "llm",
) -> T:
    """Run one call and return a validated instance of ``schema``.

    Raises :class:`RefusalError` if the model declines, so callers can skip the
    offending item and continue the run.
    """
    global _effort_supported

    kwargs: dict[str, Any] = {
        "model": model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": schema,
        # `max_completion_tokens` supersedes `max_tokens`, which reasoning
        # models reject outright.
        "max_completion_tokens": max_tokens,
    }
    if _effort_supported:
        kwargs["reasoning_effort"] = _EFFORT_MAP[effort]

    try:
        response = client().chat.completions.parse(**kwargs)
    except openai.BadRequestError as exc:
        if not (_effort_supported and _is_unsupported_effort_error(exc)):
            raise
        # Non-reasoning model. Effort is a cost/quality hint, not a correctness
        # requirement, so drop it for the remainder of the run.
        log.warning(
            "model %s does not accept reasoning_effort; continuing without it "
            "for the rest of this run.",
            model(),
        )
        _effort_supported = False
        kwargs.pop("reasoning_effort")
        response = client().chat.completions.parse(**kwargs)

    usage.add(response.usage)
    _check_cache_health(response.usage, label)

    choice = response.choices[0]

    if choice.message.refusal:
        raise RefusalError(f"{label}: model declined — {choice.message.refusal}")

    if choice.finish_reason == "length":
        raise RuntimeError(
            f"{label}: hit max_completion_tokens ({max_tokens}) before finishing. "
            "The response is truncated — raise max_tokens for this stage."
        )

    parsed = choice.message.parsed
    if parsed is None:
        raise RuntimeError(
            f"{label}: structured output failed to parse against "
            f"{schema.__name__} (finish_reason={choice.finish_reason})"
        )
    return parsed


def ping() -> str:
    """Minimal round-trip used by `verify-credentials`. Returns the model id."""
    response = client().chat.completions.create(
        model=model(),
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_completion_tokens=16,
    )
    return response.model


def available_models(limit: int = 40) -> list[str]:
    """Model ids this API key can actually reach, with obvious non-chat
    families filtered out.

    Printed by `verify-credentials` so OPENAI_MODEL is chosen from reality
    rather than from a docs page or a guess.
    """
    ids = sorted(m.id for m in client().models.list())
    noise = ("whisper", "tts", "dall-e", "embedding", "moderation", "audio", "realtime")
    return [i for i in ids if not any(n in i for n in noise)][:limit]
