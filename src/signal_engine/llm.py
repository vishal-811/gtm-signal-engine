"""Thin wrapper around the Anthropic SDK.

Everything the pipeline needs from Claude goes through :func:`structured_call`,
which centralizes four concerns that are easy to get wrong if scattered:

* **Structured output** — responses are validated against a pydantic model by
  ``client.messages.parse()``, so no stage ever parses or repairs JSON.
* **Prompt caching** — the system prompt (which carries the rubric, a large and
  byte-stable prefix reused across every company in a run) is marked cacheable.
  Cache hits are asserted, not assumed: a silent invalidator would otherwise
  quietly multiply the bill.
* **Refusals** — Claude Opus 5 returns ``stop_reason == "refusal"`` as a
  successful HTTP 200 with empty or partial content. Reading ``.content[0]``
  without checking would crash the run on an unremarkable input.
* **Cost** — token usage is accumulated per run and written to the `runs` sheet
  tab, so decisions about cheaper models are made on real numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

import anthropic
from pydantic import BaseModel

from .config import settings

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Claude Opus 5, USD per million tokens. Cache writes bill at 1.25x input for
# the default 5-minute TTL; cache reads at 0.1x.
_INPUT_PER_MTOK = 5.00
_OUTPUT_PER_MTOK = 25.00
_CACHE_WRITE_MULTIPLIER = 1.25
_CACHE_READ_MULTIPLIER = 0.10

T = TypeVar("T", bound=BaseModel)


class RefusalError(RuntimeError):
    """Claude declined the request. Raised so the caller can skip one item
    rather than abort the whole run."""


@dataclass
class Usage:
    """Running token and cost totals for one pipeline run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0
    _cache_checked: bool = field(default=False, repr=False)

    def add(self, raw: Any) -> None:
        self.calls += 1
        self.input_tokens += getattr(raw, "input_tokens", 0) or 0
        self.output_tokens += getattr(raw, "output_tokens", 0) or 0
        self.cache_creation_tokens += getattr(raw, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(raw, "cache_read_input_tokens", 0) or 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * _INPUT_PER_MTOK
            + self.cache_creation_tokens * _INPUT_PER_MTOK * _CACHE_WRITE_MULTIPLIER
            + self.cache_read_tokens * _INPUT_PER_MTOK * _CACHE_READ_MULTIPLIER
            + self.output_tokens * _OUTPUT_PER_MTOK
        ) / 1_000_000

    def summary(self) -> str:
        return (
            f"{self.calls} calls · in {self.input_tokens:,} "
            f"(cache write {self.cache_creation_tokens:,}, "
            f"read {self.cache_read_tokens:,}) · "
            f"out {self.output_tokens:,} · ${self.cost_usd:.3f}"
        )


# One tracker per process. The CLI resets it at the start of each run.
usage = Usage()


def reset_usage() -> None:
    global usage
    usage = Usage()


_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = settings().anthropic_api_key
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                "a key from https://console.anthropic.com/settings/keys"
            )
        # The SDK retries 429/5xx with backoff; 4 gives a daily batch job room
        # to ride out a transient overload without failing the whole run.
        _client = anthropic.Anthropic(api_key=key, max_retries=4)
    return _client


def _cacheable_system(text: str) -> list[dict[str, Any]]:
    """System prompt as a single cacheable block.

    The prefix must be byte-identical across calls to hit the cache, which is
    why prompts are loaded from files and never interpolated with timestamps,
    company names, or run IDs. Per-item content belongs in the user turn.
    """
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _check_cache_health(raw_usage: Any, label: str) -> None:
    """Warn once per run if caching never engaged.

    A cache miss on every call is silent and expensive — it usually means a
    prompt file picked up a dynamic value, or the prefix fell under the
    512-token minimum. Surfacing it beats discovering it on the invoice.
    """
    if usage._cache_checked:
        return
    wrote = (getattr(raw_usage, "cache_creation_input_tokens", 0) or 0) > 0
    read = (getattr(raw_usage, "cache_read_input_tokens", 0) or 0) > 0
    if not (wrote or read):
        log.warning(
            "Prompt caching did not engage on the first %s call. The system "
            "prompt may be under the 512-token minimum, or something dynamic "
            "leaked into it.",
            label,
        )
    usage._cache_checked = True


def structured_call(
    *,
    system: str,
    user: str,
    schema: type[T],
    effort: Effort = "medium",
    max_tokens: int = 8000,
    label: str = "llm",
) -> T:
    """Run one Claude call and return a validated instance of ``schema``.

    Raises :class:`RefusalError` if Claude declines, so callers can skip the
    offending item and continue the run.
    """
    # `output_format` and `output_config` are safe to pass together: the SDK
    # merges them as `{**output_config, "format": <schema>}` (verified against
    # anthropic 0.120.2, Messages.parse), so the effort hint is preserved and
    # the schema still wins the `format` key.
    response = client().messages.parse(
        model=MODEL,
        max_tokens=max_tokens,
        system=_cacheable_system(system),
        messages=[{"role": "user", "content": user}],
        output_format=schema,
        output_config={"effort": effort},
    )

    usage.add(response.usage)
    _check_cache_health(response.usage, label)

    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        category = getattr(detail, "category", None) if detail else None
        raise RefusalError(f"{label}: Claude declined this request (category={category})")

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            f"{label}: hit max_tokens ({max_tokens}) before finishing. The "
            "response is truncated — raise max_tokens for this stage."
        )

    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError(
            f"{label}: structured output failed to parse against {schema.__name__}"
        )
    return parsed


def ping() -> str:
    """Minimal round-trip used by `verify-credentials`. Returns the model id."""
    response = client().messages.create(
        model=MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
    )
    return response.model
