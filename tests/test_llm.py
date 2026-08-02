"""The provider layer.

`extract_json` gets the heaviest coverage here: in prompt mode it is the only
thing standing between a model's reply and a validated record, and it runs on
payloads that legitimately contain braces and quotes (company descriptions,
email bodies). A naive regex would corrupt those silently.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from signal_engine import llm


class TestExtractJson:
    def test_bare_object(self):
        assert llm.extract_json('{"a": 1}') == '{"a": 1}'

    def test_surrounding_whitespace(self):
        assert llm.extract_json('  \n {"a": 1} \n ') == '{"a": 1}'

    @pytest.mark.parametrize(
        "wrapped",
        [
            '```json\n{"a": 1}\n```',
            '```\n{"a": 1}\n```',
            '```JSON\n{"a": 1}\n```',
        ],
    )
    def test_markdown_fences_are_stripped(self, wrapped):
        assert json.loads(llm.extract_json(wrapped)) == {"a": 1}

    def test_leading_prose_is_skipped(self):
        text = 'Sure, here is the JSON you asked for:\n\n{"a": 1}'
        assert json.loads(llm.extract_json(text)) == {"a": 1}

    def test_trailing_prose_is_dropped(self):
        text = '{"a": 1}\n\nLet me know if you need anything else!'
        assert json.loads(llm.extract_json(text)) == {"a": 1}

    def test_prose_on_both_sides(self):
        text = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps.'
        assert json.loads(llm.extract_json(text)) == {"a": 1}

    def test_nested_objects_are_kept_whole(self):
        payload = {"a": {"b": {"c": [1, 2, {"d": 3}]}}}
        assert json.loads(llm.extract_json(json.dumps(payload))) == payload

    def test_a_brace_inside_a_string_does_not_terminate_early(self):
        # Real risk: company descriptions and drafted email bodies contain
        # braces. Counting braces without tracking strings truncates the object.
        payload = {"desc": "we use {curly} templates", "n": 2}
        assert json.loads(llm.extract_json(json.dumps(payload))) == payload

    def test_an_escaped_quote_inside_a_string_is_handled(self):
        payload = {"quote": 'she said "hi" and left', "n": 1}
        assert json.loads(llm.extract_json(json.dumps(payload))) == payload

    def test_a_backslash_before_a_quote_is_handled(self):
        payload = {"path": "C:\\\\dir\\\\file", "ok": True}
        assert json.loads(llm.extract_json(json.dumps(payload))) == payload

    def test_newlines_inside_strings_survive(self):
        payload = {"body": "line one\nline two\nline three"}
        assert json.loads(llm.extract_json(json.dumps(payload))) == payload

    def test_first_complete_object_wins(self):
        assert json.loads(llm.extract_json('{"a": 1} {"b": 2}')) == {"a": 1}

    @pytest.mark.parametrize("bad", ["", "no json at all", "just prose here."])
    def test_returns_none_when_there_is_no_object(self, bad):
        assert llm.extract_json(bad) is None

    def test_returns_none_for_an_unbalanced_object(self):
        # Truncated output must fail loudly rather than yield a partial record.
        assert llm.extract_json('{"a": 1, "b": {"c": 2}') is None

    def test_realistic_funding_payload_round_trips(self):
        payload = {
            "company_name": "Acme {Labs}",
            "one_line_description": 'A "developer-first" platform for CI/CD',
            "amount_usd": 20000000,
            "investors": ["Foundry", "a16z"],
            "hq_city": None,
        }
        text = f"Here is the extraction:\n```json\n{json.dumps(payload)}\n```\nDone."
        assert json.loads(llm.extract_json(text)) == payload


class _Tiny(BaseModel):
    city: str
    big: bool


class TestSchemaInstruction:
    def test_embeds_the_json_schema(self):
        text = llm._schema_instruction(_Tiny)
        assert "city" in text and "big" in text
        assert "json" in text.lower()

    def test_forbids_prose_and_fences(self):
        text = llm._schema_instruction(_Tiny)
        assert "no markdown code fences" in text.lower()
        assert "nothing else" in text.lower()

    def test_tells_the_model_to_use_null_rather_than_omit(self):
        # Omitted keys fail validation; invented values corrupt the record.
        # Both are worse than an explicit null.
        assert "null" in llm._schema_instruction(_Tiny).lower()


class TestParsePromptMode:
    def test_parses_a_clean_reply(self):
        result = llm._parse_prompt_mode('{"city":"Paris","big":true}', _Tiny, "t")
        assert result.city == "Paris" and result.big is True

    def test_parses_a_fenced_reply(self):
        result = llm._parse_prompt_mode(
            '```json\n{"city":"Paris","big":true}\n```', _Tiny, "t"
        )
        assert result.city == "Paris"

    def test_raises_with_a_readable_message_when_no_json_present(self):
        with pytest.raises(ValueError, match="no JSON object found"):
            llm._parse_prompt_mode("I cannot help with that.", _Tiny, "extract")

    def test_raises_when_json_is_present_but_wrong_shape(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            llm._parse_prompt_mode('{"wrong":"keys"}', _Tiny, "t")


class TestEffortMapping:
    @pytest.mark.parametrize(
        "ours,theirs",
        [("low", "low"), ("medium", "medium"), ("high", "high"),
         ("xhigh", "high"), ("max", "high")],
    )
    def test_five_levels_collapse_onto_three(self, ours, theirs):
        assert llm._EFFORT_MAP[ours] == theirs

    def test_every_effort_value_is_mapped(self):
        # A missing key would KeyError mid-run rather than at startup.
        from typing import get_args

        for value in get_args(llm.Effort):
            assert value in llm._EFFORT_MAP


class TestNormalizeModelId:
    def test_strips_the_gemini_models_prefix(self):
        assert llm.normalize_model_id("models/gemini-3.6-flash") == "gemini-3.6-flash"

    def test_leaves_a_bare_id_alone(self):
        assert llm.normalize_model_id("gpt-5") == "gpt-5"

    def test_does_not_strip_other_slashes(self):
        # OpenRouter-style ids are namespaced too but are used verbatim.
        assert llm.normalize_model_id("openai/gpt-5") == "openai/gpt-5"


class TestUnsupportedEffortDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "Unsupported parameter: 'reasoning_effort'",
            "reasoning_effort is not supported with this model",
            "Unknown parameter: reasoning_effort.",
        ],
    )
    def test_recognises_rejections(self, message):
        assert llm._is_unsupported_effort_error(Exception(message))

    @pytest.mark.parametrize(
        "message",
        ["rate limit exceeded", "invalid api key", "model not found"],
    )
    def test_ignores_unrelated_errors(self, message):
        # Misclassifying a real error as an effort problem would silently retry
        # and then surface the wrong diagnosis.
        assert not llm._is_unsupported_effort_error(Exception(message))
