"""
Tests for the pure OpenInference -> GenAI attribute mappers.

Each mapper is a pure function: given a mapping of OI attributes, it returns
a dict of GenAI attributes. Mappers are independent so they can be audited
and tested in isolation, and composable so they can be combined in
``convert_oi_to_genai``.
"""

import json
from typing import Any, Dict, cast

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.genai import (
    convert_oi_to_genai,
    map_agent,
    map_conversation,
    map_invocation_parameters,
    map_messages,
    map_model_name,
    map_provider,
    map_retrieval,
    map_span_kind,
    map_token_counts,
    map_tool_call,
    map_tools,
)
from openinference.instrumentation.genai.attributes import GenAIAttributes as GA
from openinference.instrumentation.genai.values import (
    GenAIOperationNameValues,
    GenAIProviderNameValues,
)
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


def _loads(value: AttributeValue) -> Any:
    """Helper: GenAI JSON attributes are stored as strings; cast for mypy."""
    return json.loads(cast(str, value))


class TestMapSpanKind:
    def test_llm_maps_to_chat(self) -> None:
        result = map_span_kind({SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM"})
        assert result == {GA.GEN_AI_OPERATION_NAME: GenAIOperationNameValues.CHAT.value}

    def test_embedding_maps_to_embeddings(self) -> None:
        result = map_span_kind({SpanAttributes.OPENINFERENCE_SPAN_KIND: "EMBEDDING"})
        assert result == {GA.GEN_AI_OPERATION_NAME: GenAIOperationNameValues.EMBEDDINGS.value}

    def test_retriever_maps_to_retrieval(self) -> None:
        result = map_span_kind({SpanAttributes.OPENINFERENCE_SPAN_KIND: "RETRIEVER"})
        assert result == {GA.GEN_AI_OPERATION_NAME: GenAIOperationNameValues.RETRIEVAL.value}

    def test_tool_maps_to_execute_tool(self) -> None:
        result = map_span_kind({SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL"})
        assert result == {GA.GEN_AI_OPERATION_NAME: GenAIOperationNameValues.EXECUTE_TOOL.value}

    def test_agent_maps_to_invoke_agent(self) -> None:
        result = map_span_kind({SpanAttributes.OPENINFERENCE_SPAN_KIND: "AGENT"})
        assert result == {GA.GEN_AI_OPERATION_NAME: GenAIOperationNameValues.INVOKE_AGENT.value}

    def test_llm_with_prompts_maps_to_text_completion(self) -> None:
        result = map_span_kind(
            {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
                "llm.prompts.0.prompt.text": "Say hi",
            }
        )
        assert result == {GA.GEN_AI_OPERATION_NAME: GenAIOperationNameValues.TEXT_COMPLETION.value}

    def test_unknown_kind_is_empty(self) -> None:
        assert map_span_kind({SpanAttributes.OPENINFERENCE_SPAN_KIND: "CHAIN"}) == {}
        assert map_span_kind({SpanAttributes.OPENINFERENCE_SPAN_KIND: "GUARDRAIL"}) == {}
        assert map_span_kind({}) == {}


class TestMapModelName:
    def test_llm_model_name(self) -> None:
        assert map_model_name({SpanAttributes.LLM_MODEL_NAME: "gpt-4o"}) == {
            GA.GEN_AI_REQUEST_MODEL: "gpt-4o"
        }

    def test_embedding_model_name_fallback(self) -> None:
        assert map_model_name({SpanAttributes.EMBEDDING_MODEL_NAME: "text-embedding-3-small"}) == {
            GA.GEN_AI_REQUEST_MODEL: "text-embedding-3-small"
        }

    def test_llm_model_takes_precedence(self) -> None:
        assert map_model_name(
            {
                SpanAttributes.LLM_MODEL_NAME: "gpt-4o",
                SpanAttributes.EMBEDDING_MODEL_NAME: "text-embedding-3-small",
            }
        ) == {GA.GEN_AI_REQUEST_MODEL: "gpt-4o"}

    def test_empty(self) -> None:
        assert map_model_name({}) == {}


class TestMapProvider:
    def test_openai_system_only(self) -> None:
        assert map_provider({SpanAttributes.LLM_SYSTEM: "openai"}) == {
            GA.GEN_AI_PROVIDER_NAME: GenAIProviderNameValues.OPENAI.value
        }

    def test_azure_openai_composite(self) -> None:
        assert map_provider(
            {
                SpanAttributes.LLM_SYSTEM: "openai",
                SpanAttributes.LLM_PROVIDER: "azure",
            }
        ) == {GA.GEN_AI_PROVIDER_NAME: GenAIProviderNameValues.AZURE_AI_OPENAI.value}

    def test_anthropic_on_bedrock(self) -> None:
        assert map_provider(
            {
                SpanAttributes.LLM_SYSTEM: "anthropic",
                SpanAttributes.LLM_PROVIDER: "aws",
            }
        ) == {GA.GEN_AI_PROVIDER_NAME: GenAIProviderNameValues.AWS_BEDROCK.value}

    def test_vertex_ai(self) -> None:
        assert map_provider(
            {
                SpanAttributes.LLM_SYSTEM: "vertexai",
                SpanAttributes.LLM_PROVIDER: "google",
            }
        ) == {GA.GEN_AI_PROVIDER_NAME: GenAIProviderNameValues.GCP_VERTEX_AI.value}

    def test_mistralai_spelling(self) -> None:
        assert map_provider({SpanAttributes.LLM_SYSTEM: "mistralai"}) == {
            GA.GEN_AI_PROVIDER_NAME: GenAIProviderNameValues.MISTRAL_AI.value
        }

    def test_xai_spelling(self) -> None:
        assert map_provider({SpanAttributes.LLM_PROVIDER: "xai"}) == {
            GA.GEN_AI_PROVIDER_NAME: GenAIProviderNameValues.X_AI.value
        }

    def test_provider_only_groq(self) -> None:
        assert map_provider({SpanAttributes.LLM_PROVIDER: "groq"}) == {
            GA.GEN_AI_PROVIDER_NAME: GenAIProviderNameValues.GROQ.value
        }

    def test_unknown_provider_returns_empty(self) -> None:
        assert map_provider({SpanAttributes.LLM_PROVIDER: "some-weird-thing"}) == {}
        assert map_provider({}) == {}


class TestMapTokenCounts:
    def test_basic_counts(self) -> None:
        result = map_token_counts(
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 30,
            }
        )
        # total is not in GenAI; it's computed by consumers
        assert result == {
            GA.GEN_AI_USAGE_INPUT_TOKENS: 10,
            GA.GEN_AI_USAGE_OUTPUT_TOKENS: 20,
        }

    def test_cache_details(self) -> None:
        result = map_token_counts(
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 100,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 40,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 10,
            }
        )
        assert result == {
            GA.GEN_AI_USAGE_INPUT_TOKENS: 100,
            GA.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: 40,
            GA.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS: 10,
        }

    def test_empty(self) -> None:
        assert map_token_counts({}) == {}


class TestMapInvocationParameters:
    def test_all_known_params(self) -> None:
        params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 512,
            "frequency_penalty": 0.1,
            "presence_penalty": -0.2,
            "stop": ["\n", "<END>"],
            "seed": 42,
            "n": 2,
        }
        result = map_invocation_parameters(
            {SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(params)}
        )
        assert result == {
            GA.GEN_AI_REQUEST_TEMPERATURE: 0.7,
            GA.GEN_AI_REQUEST_TOP_P: 0.9,
            GA.GEN_AI_REQUEST_TOP_K: 50,
            GA.GEN_AI_REQUEST_MAX_TOKENS: 512,
            GA.GEN_AI_REQUEST_FREQUENCY_PENALTY: 0.1,
            GA.GEN_AI_REQUEST_PRESENCE_PENALTY: -0.2,
            GA.GEN_AI_REQUEST_STOP_SEQUENCES: ("\n", "<END>"),
            GA.GEN_AI_REQUEST_SEED: 42,
            GA.GEN_AI_REQUEST_CHOICE_COUNT: 2,
        }

    def test_scalar_stop(self) -> None:
        params = {"stop": "###"}
        result = map_invocation_parameters(
            {SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(params)}
        )
        assert result == {GA.GEN_AI_REQUEST_STOP_SEQUENCES: ("###",)}

    def test_openai_alias_max_completion_tokens(self) -> None:
        params = {"max_completion_tokens": 256}
        result = map_invocation_parameters(
            {SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(params)}
        )
        assert result == {GA.GEN_AI_REQUEST_MAX_TOKENS: 256}

    def test_unknown_params_are_skipped(self) -> None:
        params = {"temperature": 1.0, "custom_field": "bar"}
        result = map_invocation_parameters(
            {SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(params)}
        )
        assert result == {GA.GEN_AI_REQUEST_TEMPERATURE: 1.0}

    def test_invalid_json_is_empty(self) -> None:
        assert (
            map_invocation_parameters({SpanAttributes.LLM_INVOCATION_PARAMETERS: "not-json"}) == {}
        )

    def test_empty(self) -> None:
        assert map_invocation_parameters({}) == {}


class TestMapMessages:
    def test_simple_user_message(self) -> None:
        oi = _input_message(0, role="user", content="Hello")
        result = map_messages(oi)
        assert GA.GEN_AI_INPUT_MESSAGES in result
        messages = _loads(result[GA.GEN_AI_INPUT_MESSAGES])
        assert messages == [{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}]

    def test_system_message_extracted(self) -> None:
        oi: Dict[str, Any] = {}
        oi.update(_input_message(0, role="system", content="You are helpful"))
        oi.update(_input_message(1, role="user", content="Hi"))
        result = map_messages(oi)
        system = _loads(result[GA.GEN_AI_SYSTEM_INSTRUCTIONS])
        assert system == [{"type": "text", "content": "You are helpful"}]
        messages = _loads(result[GA.GEN_AI_INPUT_MESSAGES])
        # system message not in input.messages
        assert [m["role"] for m in messages] == ["user"]

    def test_multimodal_contents(self) -> None:
        oi = {
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}": "text",
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}": "Look at this image",
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}": "image",
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.1.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.image.url": "https://example.com/a.png",
        }
        result = map_messages(oi)
        messages = _loads(result[GA.GEN_AI_INPUT_MESSAGES])
        assert messages == [
            {
                "role": "user",
                "parts": [
                    {"type": "text", "content": "Look at this image"},
                    {"type": "uri", "modality": "image", "uri": "https://example.com/a.png"},
                ],
            }
        ]

    def test_assistant_tool_calls(self) -> None:
        base = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
        oi = {
            f"{base}.{MessageAttributes.MESSAGE_ROLE}": "assistant",
            f"{base}.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_ID}": "call_1",
            f"{base}.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}": "get_weather",
            f"{base}.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}": '{"city":"Paris"}',
        }
        result = map_messages(oi)
        messages = _loads(result[GA.GEN_AI_INPUT_MESSAGES])
        assert messages == [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": "call_1",
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    }
                ],
            }
        ]

    def test_tool_role_response(self) -> None:
        base = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
        oi = {
            f"{base}.{MessageAttributes.MESSAGE_ROLE}": "tool",
            f"{base}.{MessageAttributes.MESSAGE_TOOL_CALL_ID}": "call_1",
            f"{base}.{MessageAttributes.MESSAGE_CONTENT}": "Rainy, 57F",
        }
        result = map_messages(oi)
        messages = _loads(result[GA.GEN_AI_INPUT_MESSAGES])
        assert messages == [
            {
                "role": "tool",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "id": "call_1",
                        "response": "Rainy, 57F",
                    }
                ],
            }
        ]

    def test_output_messages(self) -> None:
        base = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
        oi = {
            f"{base}.{MessageAttributes.MESSAGE_ROLE}": "assistant",
            f"{base}.{MessageAttributes.MESSAGE_CONTENT}": "Hello back",
        }
        result = map_messages(oi)
        messages = _loads(result[GA.GEN_AI_OUTPUT_MESSAGES])
        assert messages == [
            {"role": "assistant", "parts": [{"type": "text", "content": "Hello back"}]}
        ]

    def test_empty(self) -> None:
        assert map_messages({}) == {}


class TestMapTools:
    def test_tools_as_json_strings(self) -> None:
        schema_a = {"type": "function", "function": {"name": "get_weather"}}
        schema_b = {"type": "function", "function": {"name": "get_time"}}
        oi = {
            f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}": json.dumps(schema_a),
            f"{SpanAttributes.LLM_TOOLS}.1.{ToolAttributes.TOOL_JSON_SCHEMA}": json.dumps(schema_b),
        }
        result = map_tools(oi)
        assert GA.GEN_AI_TOOL_DEFINITIONS in result
        defs = _loads(result[GA.GEN_AI_TOOL_DEFINITIONS])
        assert defs == [schema_a, schema_b]

    def test_empty(self) -> None:
        assert map_tools({}) == {}


class TestMapToolCall:
    def test_execute_tool_attributes(self) -> None:
        oi = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
            SpanAttributes.TOOL_NAME: "get_weather",
            SpanAttributes.TOOL_DESCRIPTION: "Fetch current weather",
            SpanAttributes.TOOL_ID: "call_1",
            SpanAttributes.INPUT_VALUE: '{"city": "Paris"}',
            SpanAttributes.OUTPUT_VALUE: "Rainy, 57F",
        }
        result = map_tool_call(oi)
        assert result == {
            GA.GEN_AI_TOOL_NAME: "get_weather",
            GA.GEN_AI_TOOL_DESCRIPTION: "Fetch current weather",
            GA.GEN_AI_TOOL_CALL_ID: "call_1",
            GA.GEN_AI_TOOL_CALL_ARGUMENTS: '{"city": "Paris"}',
            GA.GEN_AI_TOOL_CALL_RESULT: "Rainy, 57F",
        }

    def test_non_tool_span_is_empty(self) -> None:
        oi = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
            SpanAttributes.TOOL_NAME: "should_not_emit",
        }
        assert map_tool_call(oi) == {}


class TestMapConversation:
    def test_session_id(self) -> None:
        assert map_conversation({SpanAttributes.SESSION_ID: "sess_1"}) == {
            GA.GEN_AI_CONVERSATION_ID: "sess_1"
        }

    def test_empty(self) -> None:
        assert map_conversation({}) == {}


class TestMapAgent:
    def test_agent_name(self) -> None:
        assert map_agent({SpanAttributes.AGENT_NAME: "Planner"}) == {
            GA.GEN_AI_AGENT_NAME: "Planner"
        }


class TestMapRetrieval:
    def test_documents(self) -> None:
        oi: Dict[str, AttributeValue] = {
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.document.id": "a",
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.document.content": "alpha",
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.document.score": 0.9,
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.1.document.id": "b",
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.1.document.content": "beta",
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.1.document.score": 0.5,
        }
        result = map_retrieval(oi)
        docs = _loads(result[GA.GEN_AI_RETRIEVAL_DOCUMENTS])
        assert docs == [
            {"id": "a", "content": "alpha", "score": 0.9},
            {"id": "b", "content": "beta", "score": 0.5},
        ]

    def test_empty(self) -> None:
        assert map_retrieval({}) == {}


class TestConvertOiToGenai:
    def test_full_chat_span(self) -> None:
        oi: Dict[str, AttributeValue] = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
            SpanAttributes.LLM_MODEL_NAME: "gpt-4o",
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_PROVIDER: "openai",
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 7,
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 13,
            SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(
                {"temperature": 0.5, "max_tokens": 128}
            ),
            SpanAttributes.SESSION_ID: "sess_42",
        }
        oi.update(_input_message(0, role="user", content="Hi"))
        oi.update(_output_message(0, role="assistant", content="Hello!"))
        result = convert_oi_to_genai(oi)
        assert result[GA.GEN_AI_OPERATION_NAME] == "chat"
        assert result[GA.GEN_AI_REQUEST_MODEL] == "gpt-4o"
        assert result[GA.GEN_AI_PROVIDER_NAME] == "openai"
        assert result[GA.GEN_AI_USAGE_INPUT_TOKENS] == 7
        assert result[GA.GEN_AI_USAGE_OUTPUT_TOKENS] == 13
        assert result[GA.GEN_AI_REQUEST_TEMPERATURE] == 0.5
        assert result[GA.GEN_AI_REQUEST_MAX_TOKENS] == 128
        assert result[GA.GEN_AI_CONVERSATION_ID] == "sess_42"
        assert GA.GEN_AI_INPUT_MESSAGES in result
        assert GA.GEN_AI_OUTPUT_MESSAGES in result

    def test_purity(self) -> None:
        oi: Dict[str, Any] = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
            SpanAttributes.LLM_MODEL_NAME: "gpt-4o",
        }
        before = dict(oi)
        convert_oi_to_genai(oi)
        assert oi == before, "mapper must not mutate its input"

    def test_empty_returns_empty(self) -> None:
        assert convert_oi_to_genai({}) == {}


def _input_message(index: int, *, role: str, content: str) -> Dict[str, Any]:
    base = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}"
    return {
        f"{base}.{MessageAttributes.MESSAGE_ROLE}": role,
        f"{base}.{MessageAttributes.MESSAGE_CONTENT}": content,
    }


def _output_message(index: int, *, role: str, content: str) -> Dict[str, Any]:
    base = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}"
    return {
        f"{base}.{MessageAttributes.MESSAGE_ROLE}": role,
        f"{base}.{MessageAttributes.MESSAGE_CONTENT}": content,
    }
