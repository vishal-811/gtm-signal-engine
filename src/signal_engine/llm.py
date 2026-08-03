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


# Latched on when the endpoint proves it ignores response_format, or forced by
# LLM_STRUCTURED_MODE=prompt. Initialised lazily on first use so importing this
# module never requires config to be loadable.
_prompt_mode: bool | None = None


def _in_prompt_mode() -> bool:
    global _prompt_mode
    if _prompt_mode is None:
        _prompt_mode = settings().llm_structured_mode == "prompt"
    return _prompt_mode


def reset_effort_probe() -> None:
    global _effort_supported, _prompt_mode
    _effort_supported = True
    _prompt_mode = settings().llm_structured_mode == "prompt"


def structured_mode() -> str:
    """Which structured-output strategy is currently in force."""
    return "prompt" if _in_prompt_mode() else "native"


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        cfg = settings()
        if not cfg.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add a "
                "key from https://platform.openai.com/api-keys"
            )
        # The SDK retries 429/5xx with backoff; 4 gives a daily batch job room
        # to ride out a transient overload without failing the whole run.
        kwargs: dict[str, Any] = {
            "api_key": cfg.openai_api_key,
            "max_retries": 4,
            "timeout": cfg.llm_request_timeout,
        }
        if cfg.openai_base_url.strip():
            kwargs["base_url"] = cfg.openai_base_url.strip()
        if cfg.openai_user_agent.strip():
            kwargs["default_headers"] = {"User-Agent": cfg.openai_user_agent.strip()}
        _client = openai.OpenAI(**kwargs)
    return _client


def reset_client() -> None:
    """Drop the cached client. Needed when settings change within a process."""
    global _client
    _client = None


def endpoint() -> str:
    """Human-readable description of where calls are going."""
    base = settings().openai_base_url.strip()
    return base or "https://api.openai.com/v1 (OpenAI direct)"


def model() -> str:
    return settings().openai_model


def normalize_model_id(model_id: str) -> str:
    """Strip a provider's namespace prefix for comparison purposes.

    Gemini's OpenAI-compatible ``/models`` endpoint returns ids as
    ``models/gemini-3.6-flash`` while happily *accepting* the bare
    ``gemini-3.6-flash`` on completion calls. Comparing the two literally makes
    a perfectly valid model look unavailable.
    """
    return model_id.split("/", 1)[-1] if model_id.startswith("models/") else model_id


def _check_cache_health(raw_usage: Any, label: str) -> None:
    """Warn once per run when caching cannot possibly engage.

    OpenAI caches automatically above 1024 prompt tokens, so a prompt below
    that threshold will never hit — silently, and for the whole run.
    """
    # The capability probe is deliberately tiny; warning that it is too short
    # to cache is noise, and it would burn the once-per-run flag before a real
    # stage got the chance to report.
    if label == "probe" or usage._cache_checked:
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


# ── prompt-mode structured output ─────────────────────────────────────────────
#
# Some OpenAI-compatible endpoints accept `response_format` and ignore it,
# answering in prose. AgentRouter does this on every model it offers. For those,
# the schema is described in the prompt and the reply is parsed here.
#
# This is strictly worse than native enforcement — the provider guarantees
# nothing — so it is only used where native does not work.


def _schema_instruction(schema: type[BaseModel]) -> str:
    """Describe the required JSON shape for an endpoint that ignores schemas."""
    import json

    return (
        "\n\n---\n\n"
        "# OUTPUT FORMAT — MANDATORY\n\n"
        "Reply with a single raw JSON object and nothing else. No prose before "
        "or after it, no explanation, no markdown code fences.\n\n"
        "It must validate against this JSON Schema exactly. Include every "
        "required key. Use null for values you do not know — never omit a key "
        "and never invent a value to fill it.\n\n"
        f"```json\n{json.dumps(schema.model_json_schema(), indent=2)}\n```"
    )


def extract_json(text: str) -> str | None:
    """Pull the first complete JSON object out of a model reply.

    Handles the three things models actually do when asked for raw JSON: wrap
    it in ``` fences, prefix it with a sentence, or both. Brace counting is
    string-aware, because a ``}`` inside a quoted value would otherwise
    terminate the object early — which is not hypothetical here, where the
    payloads contain company descriptions and email bodies.

    Returns None if no balanced object is present.
    """
    if not text:
        return None

    cleaned = text.strip()

    # Strip a fenced block, keeping only its contents.
    if cleaned.startswith("```"):
        newline = cleaned.find("\n")
        if newline != -1:
            cleaned = cleaned[newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return None


def _parse_prompt_mode(text: str, schema: type[T], label: str) -> T:
    payload = extract_json(text)
    if payload is None:
        raise ValueError(
            f"{label}: no JSON object found in the reply "
            f"(began {text[:120]!r})"
        )
    return schema.model_validate_json(payload)


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

    Uses the provider's native ``response_format`` where it works, and falls
    back to describing the schema in the prompt where it does not. See
    ``LLM_STRUCTURED_MODE``.

    Raises :class:`RefusalError` if the model declines, so callers can skip the
    offending item and continue the run.
    """
    global _prompt_mode

    if not _in_prompt_mode():
        try:
            return _call_native(system, user, schema, effort, max_tokens, label)
        except _NativeSchemaIgnored as exc:
            if settings().llm_structured_mode == "native":
                raise RuntimeError(
                    f"{label}: this endpoint ignores response_format and "
                    "LLM_STRUCTURED_MODE=native forbids the fallback. Set it to "
                    "'auto' or 'prompt'."
                ) from exc
            # Latch for the rest of the run: one wasted call, not one per item.
            log.warning(
                "endpoint ignored response_format and replied in prose; "
                "switching to prompt-described JSON for the rest of this run."
            )
            _prompt_mode = True

    return _call_prompt_mode(system, user, schema, effort, max_tokens, label)


class _NativeSchemaIgnored(RuntimeError):
    """The endpoint accepted response_format and answered in prose anyway."""


def _base_kwargs(
    system: str, user: str, effort: Effort, max_tokens: int
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # `max_completion_tokens` supersedes `max_tokens`, which reasoning
        # models reject outright.
        "max_completion_tokens": max_tokens,
    }
    if _effort_supported:
        kwargs["reasoning_effort"] = _EFFORT_MAP[effort]
    return kwargs


def _send(call, kwargs: dict[str, Any], label: str):
    """Issue a request, retrying once without reasoning_effort if rejected."""
    global _effort_supported
    try:
        return call(**kwargs)
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
        kwargs.pop("reasoning_effort", None)
        return call(**kwargs)


class UpstreamPayloadError(RuntimeError):
    """The endpoint answered 200 with something that is not a completion."""


def _require_completion(response: Any, label: str) -> Any:
    """Fail loudly when the endpoint returns 200 with a non-completion body.

    The SDK does not enforce the response shape: it hands the JSON to pydantic's
    lenient ``construct_type``, so a bare string body comes back as ``str``
    rather than raising. The first symptom was
    ``'str' object has no attribute 'usage'`` on every batch — accurate, but it
    named the SDK's internals instead of the gateway that misbehaved, and a
    scheduled run reported success while extracting nothing.

    Observed against an OpenAI-compatible gateway from a datacenter IP, where
    the same request succeeds from a laptop.
    """
    if hasattr(response, "choices") and hasattr(response, "usage"):
        return response
    body = response if isinstance(response, str) else repr(response)
    raise UpstreamPayloadError(
        f"{label}: {endpoint()} returned HTTP 200 but the body is not a chat "
        f"completion (got {type(response).__name__}). "
        f"First 300 characters: {body[:300]!r}"
    )


def _guard(choice, label: str, max_tokens: int) -> None:
    if choice.message.refusal:
        raise RefusalError(f"{label}: model declined — {choice.message.refusal}")
    if choice.finish_reason == "length":
        raise RuntimeError(
            f"{label}: hit max_completion_tokens ({max_tokens}) before finishing. "
            "The response is truncated — raise max_tokens for this stage."
        )


def _call_native(
    system: str, user: str, schema: type[T], effort: Effort, max_tokens: int, label: str
) -> T:
    kwargs = _base_kwargs(system, user, effort, max_tokens)
    kwargs["response_format"] = schema

    try:
        response = _send(client().chat.completions.parse, kwargs, label)
    except Exception as exc:  # noqa: BLE001
        # The SDK raises a pydantic ValidationError when the reply is not JSON,
        # which is what an endpoint that ignores response_format produces. That
        # is a capability signal, not a bad response — distinguish it so `auto`
        # can fall back rather than failing the item.
        if "json_invalid" in str(exc) or "Invalid JSON" in str(exc):
            raise _NativeSchemaIgnored(str(exc)[:200]) from exc
        raise

    usage.add(response.usage)
    _check_cache_health(response.usage, label)
    choice = response.choices[0]
    _guard(choice, label, max_tokens)

    parsed = choice.message.parsed
    if parsed is None:
        raise RuntimeError(
            f"{label}: structured output failed to parse against "
            f"{schema.__name__} (finish_reason={choice.finish_reason})"
        )
    return parsed


def _call_prompt_mode(
    system: str, user: str, schema: type[T], effort: Effort, max_tokens: int, label: str
) -> T:
    """Describe the schema in the prompt and parse the reply ourselves.

    Retries once with the parse error fed back, which recovers the common
    near-misses (a trailing comment, a stray sentence) without a second full
    round of the pipeline.
    """
    instructed = system + _schema_instruction(schema)
    kwargs = _base_kwargs(instructed, user, effort, max_tokens)

    response = _require_completion(
        _send(client().chat.completions.create, kwargs, label), label
    )
    usage.add(response.usage)
    _check_cache_health(response.usage, label)
    choice = response.choices[0]
    _guard(choice, label, max_tokens)

    content = choice.message.content or ""
    try:
        return _parse_prompt_mode(content, schema, label)
    except Exception as exc:  # noqa: BLE001
        # Bind the *message* here, not the exception. Python unbinds the `as`
        # name when the except block ends, so reading it below raised NameError
        # and killed the retry that is the whole point of this branch — a
        # batch of articles was discarded every time the model fumbled its JSON.
        first_error = str(exc)[:200]
        log.info("%s: reply was not valid JSON, retrying once", label)

    retry = _base_kwargs(instructed, user, effort, max_tokens)
    retry["messages"].append({"role": "assistant", "content": content})
    retry["messages"].append(
        {
            "role": "user",
            "content": (
                f"That reply could not be parsed as JSON: {first_error}\n\n"
                "Reply again with only the raw JSON object. No prose, no code "
                "fences."
            ),
        }
    )
    response = _require_completion(
        _send(client().chat.completions.create, retry, label), label
    )
    usage.add(response.usage)
    choice = response.choices[0]
    _guard(choice, label, max_tokens)
    return _parse_prompt_mode(choice.message.content or "", schema, label)


def ping() -> str:
    """Minimal round-trip used by `verify-credentials`. Returns the model id."""
    response = client().chat.completions.create(
        model=model(),
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_completion_tokens=16,
    )
    return response.model


class _ProbeAnswer(BaseModel):
    """Trivial schema used only by :func:`probe_structured_output`."""

    city: str
    population_is_over_one_million: bool


def probe_structured_output() -> tuple[bool, str]:
    """Verify the endpoint really implements strict JSON-schema output.

    Every stage of this pipeline depends on it, and it is the capability that
    OpenAI-compatible gateways most often omit or implement partially — a
    gateway can pass a chat-completion smoke test and still ignore
    ``response_format``, which would surface later as parse failures on every
    single article rather than as a clear setup error.

    Costs one tiny call. Returns (ok, detail).
    """
    try:
        result = structured_call(
            system="You answer factual questions. Be accurate.",
            user="What is the capital of France, and does it have over 1 million people?",
            schema=_ProbeAnswer,
            effort="low",
            max_tokens=2000,
            label="probe",
        )
    except RefusalError as exc:
        return False, f"model refused the probe: {exc}"
    except Exception as exc:  # noqa: BLE001 - any failure here is disqualifying
        return False, f"{type(exc).__name__}: {str(exc)[:220]}"

    if not result.city:
        return False, "returned a valid shape but empty content"
    return True, f"strict schema honoured (probe answered {result.city!r})"


def available_models(limit: int = 40) -> list[str]:
    """Model ids this API key can actually reach, with obvious non-chat
    families filtered out.

    Printed by `verify-credentials` so OPENAI_MODEL is chosen from reality
    rather than from a docs page or a guess.
    """
    ids = sorted(m.id for m in client().models.list())
    noise = ("whisper", "tts", "dall-e", "embedding", "moderation", "audio", "realtime")
    return [i for i in ids if not any(n in i for n in noise)][:limit]
