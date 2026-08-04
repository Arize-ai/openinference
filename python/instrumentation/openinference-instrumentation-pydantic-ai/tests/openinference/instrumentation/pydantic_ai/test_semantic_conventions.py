import json
from typing import Any, Dict

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_AGENT_NAME,
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GenAiOperationNameValues,
)

from openinference.instrumentation.pydantic_ai.semantic_conventions import get_attributes
from openinference.semconv.trace import SpanAttributes, ToolAttributes

# Legacy (instrumentation version 2) flat attribute keys used by pydantic-ai.
LEGACY_TOOL_ARGUMENTS_KEY = "tool_arguments"
LEGACY_TOOL_RESPONSE_KEY = "tool_response"
LEGACY_AGENT_NAME_KEY = "agent_name"

GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"


def test_tool_attributes_from_legacy_flat_keys_only() -> None:
    """Instrumentation version 2 emits flat tool_arguments/tool_response keys."""
    gen_ai_attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL.value,
        GEN_AI_TOOL_NAME: "get_weather",
        GEN_AI_TOOL_CALL_ID: "call_123",
        LEGACY_TOOL_ARGUMENTS_KEY: '{"city": "Paris"}',
        LEGACY_TOOL_RESPONSE_KEY: '{"temp": 20}',
    }

    attributes = dict(get_attributes(gen_ai_attrs))

    assert attributes[SpanAttributes.TOOL_PARAMETERS] == '{"city": "Paris"}'
    assert attributes[SpanAttributes.OUTPUT_VALUE] == '{"temp": 20}'


def test_tool_attributes_from_dotted_keys_only() -> None:
    """Instrumentation version >=3 (default since pydantic-ai 2.0.0) emits dotted keys."""
    gen_ai_attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL.value,
        GEN_AI_TOOL_NAME: "get_weather",
        GEN_AI_TOOL_CALL_ID: "call_123",
        GEN_AI_TOOL_CALL_ARGUMENTS: '{"city": "Paris"}',
        GEN_AI_TOOL_CALL_RESULT: '{"temp": 20}',
    }

    attributes = dict(get_attributes(gen_ai_attrs))

    assert attributes[SpanAttributes.TOOL_PARAMETERS] == '{"city": "Paris"}'
    assert attributes[SpanAttributes.OUTPUT_VALUE] == '{"temp": 20}'


def test_tool_attributes_absent_when_neither_key_present() -> None:
    """Neither flat nor dotted tool argument/result keys means no TOOL_PARAMETERS/OUTPUT_VALUE."""
    gen_ai_attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL.value,
        GEN_AI_TOOL_NAME: "get_weather",
        GEN_AI_TOOL_CALL_ID: "call_123",
    }

    attributes = dict(get_attributes(gen_ai_attrs))

    assert SpanAttributes.TOOL_PARAMETERS not in attributes
    assert SpanAttributes.OUTPUT_VALUE not in attributes


def test_tool_attributes_dotted_keys_take_precedence_over_legacy_flat_keys() -> None:
    """When both flat and dotted keys are present, the dotted (newer) value wins."""
    gen_ai_attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL.value,
        GEN_AI_TOOL_NAME: "get_weather",
        GEN_AI_TOOL_CALL_ID: "call_123",
        LEGACY_TOOL_ARGUMENTS_KEY: '{"city": "legacy"}',
        LEGACY_TOOL_RESPONSE_KEY: '{"temp": "legacy"}',
        GEN_AI_TOOL_CALL_ARGUMENTS: '{"city": "dotted"}',
        GEN_AI_TOOL_CALL_RESULT: '{"temp": "dotted"}',
    }

    attributes = dict(get_attributes(gen_ai_attrs))

    assert attributes[SpanAttributes.TOOL_PARAMETERS] == '{"city": "dotted"}'
    assert attributes[SpanAttributes.OUTPUT_VALUE] == '{"temp": "dotted"}'


def test_ignore_token_counts_triggered_by_legacy_flat_agent_name_only() -> None:
    """A span carrying a flat agent_name key (e.g. an AGENT span) suppresses token counts,
    even though it also carries usage attributes (which pydantic-ai adds to agent spans)."""
    gen_ai_attrs: Dict[str, Any] = {
        LEGACY_AGENT_NAME_KEY: "my_agent",
        GEN_AI_USAGE_INPUT_TOKENS: 10,
        GEN_AI_USAGE_OUTPUT_TOKENS: 20,
    }

    attributes = dict(get_attributes(gen_ai_attrs))

    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in attributes
    # The span kind fallback should still resolve to AGENT via the flat key.
    assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "AGENT"


def test_ignore_token_counts_triggered_by_v3_invoke_agent_operation() -> None:
    """In pydantic-ai instrumentation v3+, AGENT spans carry gen_ai.operation.name='invoke_agent'.
    Token counts must be suppressed via the operation name, NOT gen_ai.agent.name alone —
    newer pydantic-ai emits gen_ai.agent.name on LLM spans too, so it cannot be used as the
    sole discriminator without incorrectly suppressing LLM-span token counts."""
    gen_ai_attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        GEN_AI_AGENT_NAME: "my_agent",
        GEN_AI_USAGE_INPUT_TOKENS: 10,
        GEN_AI_USAGE_OUTPUT_TOKENS: 20,
    }

    attributes = dict(get_attributes(gen_ai_attrs))

    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in attributes
    assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "AGENT"


def test_token_counts_present_when_no_agent_name_key() -> None:
    """Sanity check: without any agent_name key, token counts are not suppressed
    (e.g. a plain LLM span)."""
    gen_ai_attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        GEN_AI_USAGE_INPUT_TOKENS: 10,
        GEN_AI_USAGE_OUTPUT_TOKENS: 20,
    }

    attributes = dict(get_attributes(gen_ai_attrs))

    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 30


def test_instrumentation_version_5_default_dotted_keys() -> None:
    """Reflects pydantic-ai's actual default instrumentation version (5, the default since
    pydantic-ai 2.0.0; confirmed via `DEFAULT_INSTRUMENTATION_VERSION = 5` in pydantic-ai's
    `_instrumentation.py`), where only dotted gen_ai.* keys are ever emitted and the legacy flat
    keys (tool_arguments, tool_response, agent_name) never appear.

    This complements the other tests above, which exercise precedence/fallback behavior using
    synthetic or hypothetical combinations of flat and dotted keys (including older
    instrumentation versions). Here we instead simulate the two real-world span shapes pydantic-ai
    version 5 actually produces -- a TOOL span and an AGENT span -- each using dotted keys only.
    """
    # A TOOL span as emitted by instrumentation version 5: dotted keys only, no legacy flat keys.
    tool_gen_ai_attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.EXECUTE_TOOL.value,
        GEN_AI_TOOL_NAME: "get_weather",
        GEN_AI_TOOL_CALL_ID: "call_123",
        GEN_AI_TOOL_CALL_ARGUMENTS: '{"city": "Paris"}',
        GEN_AI_TOOL_CALL_RESULT: '{"temp": 20}',
    }
    assert LEGACY_TOOL_ARGUMENTS_KEY not in tool_gen_ai_attrs
    assert LEGACY_TOOL_RESPONSE_KEY not in tool_gen_ai_attrs

    tool_attributes = dict(get_attributes(tool_gen_ai_attrs))

    assert tool_attributes[SpanAttributes.TOOL_PARAMETERS] == '{"city": "Paris"}'
    assert tool_attributes[SpanAttributes.OUTPUT_VALUE] == '{"temp": 20}'

    # An AGENT span as emitted by instrumentation version 5: gen_ai.operation.name='invoke_agent'
    # is the reliable discriminator (gen_ai.agent.name alone is not sufficient because newer
    # pydantic-ai emits it on LLM spans too).
    agent_gen_ai_attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        GEN_AI_AGENT_NAME: "my_agent",
        GEN_AI_USAGE_INPUT_TOKENS: 10,
        GEN_AI_USAGE_OUTPUT_TOKENS: 20,
    }
    assert LEGACY_AGENT_NAME_KEY not in agent_gen_ai_attrs

    agent_attributes = dict(get_attributes(agent_gen_ai_attrs))

    assert agent_attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "AGENT"
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT not in agent_attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION not in agent_attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in agent_attributes


def _make_model_request_params(tools: list[Dict[str, Any]]) -> Dict[str, Any]:
    return {"model_request_parameters": json.dumps({"function_tools": tools})}


def test_tool_description_none_does_not_raise() -> None:
    """A tool with description=None must not emit a None span attribute.

    pydantic-ai may emit model_request_parameters where a tool's description
    field is present but null (tools with no docstring). Yielding None as an
    OTel attribute value causes the OTLP exporter to raise
    'Invalid type <class NoneType>'.
    """
    gen_ai_attrs = _make_model_request_params(
        [{"name": "get_weather", "description": None, "properties": {"city": {"type": "string"}}}]
    )

    attrs = dict(get_attributes(gen_ai_attrs))

    assert attrs.get(f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_NAME}") == "get_weather"
    assert f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_DESCRIPTION}" not in attrs


def test_tool_description_present_is_emitted() -> None:
    """A tool with a non-None description should still be emitted normally."""
    gen_ai_attrs = _make_model_request_params(
        [{"name": "get_weather", "description": "Returns the weather.", "properties": {}}]
    )

    attrs = dict(get_attributes(gen_ai_attrs))

    assert (
        attrs.get(f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_DESCRIPTION}")
        == "Returns the weather."
    )


def test_tool_json_schema_from_parameters_json_schema_key() -> None:
    """pydantic-ai serializes a ToolDefinition's JSON schema under the
    ``parameters_json_schema`` key (confirmed for both instrumentation v2 / pydantic-ai
    1.51.0 and v5 / pydantic-ai 2.0.0). The extractor must read that key so the LLM span
    carries ``llm.tools.0.tool.json_schema`` -- reading ``properties`` alone silently drops it.
    """
    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    gen_ai_attrs = _make_model_request_params(
        [{"name": "get_weather", "description": "d", "parameters_json_schema": schema}]
    )

    attrs = dict(get_attributes(gen_ai_attrs))

    json_schema_attr = attrs.get(f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}")
    assert json_schema_attr is not None
    assert json.loads(json_schema_attr) == schema


def test_tool_json_schema_falls_back_to_properties_key() -> None:
    """If only the legacy ``properties`` key is present, it is still used as the schema."""
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    gen_ai_attrs = _make_model_request_params([{"name": "get_weather", "properties": schema}])

    attrs = dict(get_attributes(gen_ai_attrs))

    json_schema_attr = attrs.get(f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}")
    assert json_schema_attr is not None
    assert json.loads(json_schema_attr) == schema


def test_tool_json_schema_parameters_json_schema_takes_precedence() -> None:
    """When both the canonical ``parameters_json_schema`` and the legacy ``properties`` key
    are present, the canonical key wins (mirrors the tool argument/result precedence)."""
    canonical = {"type": "object", "properties": {"city": {"type": "string"}}}
    legacy = {"type": "object", "properties": {"stale": {"type": "string"}}}
    gen_ai_attrs = _make_model_request_params(
        [{"name": "get_weather", "parameters_json_schema": canonical, "properties": legacy}]
    )

    attrs = dict(get_attributes(gen_ai_attrs))

    json_schema_attr = attrs.get(f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}")
    assert json_schema_attr is not None
    assert json.loads(json_schema_attr) == canonical


def test_tool_json_schema_absent_when_neither_key_present() -> None:
    """A tool with no schema key emits no ``tool.json_schema`` attribute (but still its name)."""
    gen_ai_attrs = _make_model_request_params([{"name": "get_weather"}])

    attrs = dict(get_attributes(gen_ai_attrs))

    assert attrs.get(f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_NAME}") == "get_weather"
    assert f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}" not in attrs
