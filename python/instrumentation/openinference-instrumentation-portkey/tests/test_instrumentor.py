import json
from importlib import import_module

import pytest
import respx
from httpx import Response
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import MessageAttributes, SpanAttributes


@pytest.mark.vcr(
    before_record_request=lambda request: setattr(  # type: ignore[func-returns-value]
        request,
        "headers",
        {k: v for k, v in request.headers.items() if not k.lower().startswith("x-portkey")},
    )
    or request,
    before_record_response=lambda response: {
        **response,
        "headers": {
            k: v for k, v in response["headers"].items() if not k.lower().startswith("x-portkey")
        },
    },
)
def test_chat_completion(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    portkey = import_module("portkey_ai")
    client = portkey.Portkey(
        api_key="REDACTED",
        virtual_key="REDACTED",
    )
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the weather like?"}], model="gpt-4o-mini"
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    expected_attributes = {
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0."
        f"{MessageAttributes.MESSAGE_CONTENT}": "What's the weather like?",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gpt-4o-mini-2024-07-18",
        SpanAttributes.LLM_FINISH_REASON: resp.choices[0].finish_reason,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: resp.usage.total_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: resp.usage.prompt_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: resp.usage.completion_tokens,
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": resp.choices[
            0
        ].message.role,
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": resp.choices[
            0
        ].message.content,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value


@pytest.mark.vcr(
    before_record_request=lambda request: setattr(  # type: ignore[func-returns-value]
        request,
        "headers",
        {k: v for k, v in request.headers.items() if not k.lower().startswith("x-portkey")},
    )
    or request,
    before_record_response=lambda response: {
        **response,
        "headers": {
            k: v for k, v in response["headers"].items() if not k.lower().startswith("x-portkey")
        },
    },
)
def test_prompt_template(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    prompt_id = "pp-weather-pr-b74c4f"
    portkey = import_module("portkey_ai")
    variables = {"location": "New York City"}
    client = portkey.Portkey(
        api_key="REDACTED",
        virtual_key="REDACTED",
    )
    resp = client.prompts.completions.create(
        prompt_id=prompt_id,
        variables=variables,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    expected_attributes = {
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gpt-4.1-2025-04-14",
        SpanAttributes.LLM_FINISH_REASON: resp.choices[0].finish_reason,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: resp.usage.total_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: resp.usage.prompt_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: resp.usage.completion_tokens,
        SpanAttributes.PROMPT_ID: prompt_id,
        SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES: json.dumps(variables),
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "assistant",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": resp.choices[
            0
        ].message.content,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value


@pytest.mark.asyncio
@pytest.mark.vcr(
    before_record_request=lambda request: setattr(  # type: ignore[func-returns-value]
        request,
        "headers",
        {k: v for k, v in request.headers.items() if not k.lower().startswith("x-portkey")},
    )
    or request,
    before_record_response=lambda response: {
        **response,
        "headers": {
            k: v for k, v in response["headers"].items() if not k.lower().startswith("x-portkey")
        },
    },
)
async def test_async_chat_completion(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    portkey = import_module("portkey_ai")
    client = portkey.AsyncPortkey(
        api_key="REDACTED",
        virtual_key="REDACTED",
    )
    resp = await client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the weather like?"}], model="gpt-4o-mini"
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "AsyncCompletions"
    attributes = dict(span.attributes or {})

    expected_attributes = {
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0."
        f"{MessageAttributes.MESSAGE_CONTENT}": "What's the weather like?",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gpt-4o-mini-2024-07-18",
        SpanAttributes.LLM_FINISH_REASON: resp.choices[0].finish_reason,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: resp.usage.total_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: resp.usage.prompt_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: resp.usage.completion_tokens,
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": resp.choices[
            0
        ].message.role,
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": resp.choices[
            0
        ].message.content,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value


@pytest.mark.asyncio
@pytest.mark.vcr(
    before_record_request=lambda request: setattr(  # type: ignore[func-returns-value]
        request,
        "headers",
        {k: v for k, v in request.headers.items() if not k.lower().startswith("x-portkey")},
    )
    or request,
    before_record_response=lambda response: {
        **response,
        "headers": {
            k: v for k, v in response["headers"].items() if not k.lower().startswith("x-portkey")
        },
    },
)
async def test_async_prompt_template(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    prompt_id = "pp-weather-pr-b74c4f"
    portkey = import_module("portkey_ai")
    variables = {"location": "New York City"}
    client = portkey.AsyncPortkey(
        api_key="REDACTED",
        virtual_key="REDACTED",
    )
    resp = await client.prompts.completions.create(
        prompt_id=prompt_id,
        variables=variables,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "AsyncCompletions"
    attributes = dict(span.attributes or {})

    expected_attributes = {
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gpt-4.1-2025-04-14",
        SpanAttributes.LLM_FINISH_REASON: resp.choices[0].finish_reason,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: resp.usage.total_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: resp.usage.prompt_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: resp.usage.completion_tokens,
        SpanAttributes.PROMPT_ID: prompt_id,
        SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES: json.dumps(variables),
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "assistant",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": resp.choices[
            0
        ].message.content,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value


@pytest.mark.parametrize(
    "finish_reason",
    [
        "stop",
        "length",
        "tool_calls",
        "content_filter",
    ],
)
def test_finish_reason_values(
    finish_reason: str,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    with respx.mock(
        base_url="https://api.portkey.ai",
        assert_all_called=True,
    ) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=Response(
                status_code=200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1750000000,
                    "model": "gpt-4o-mini-2024-07-18",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello!",
                            },
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 10,
                        "total_tokens": 15,
                    },
                },
            )
        )

        portkey = import_module("portkey_ai")
        client = portkey.Portkey(
            api_key="REDACTED",
            virtual_key="REDACTED",
        )
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    assert attributes.get(SpanAttributes.LLM_FINISH_REASON) == finish_reason


def test_uninstrument_restores_all_wrapped_methods(
    tracer_provider: trace_api.TracerProvider,
) -> None:
    from portkey_ai.api_resources.apis import chat_complete, generation

    from openinference.instrumentation.portkey import PortkeyInstrumentor

    instrumentor = PortkeyInstrumentor()
    original_chat_create = chat_complete.Completions.create
    original_async_chat_create = chat_complete.AsyncCompletions.create
    original_prompt_create = generation.Completions.create
    original_async_prompt_create = generation.AsyncCompletions.create

    try:
        instrumentor.instrument(tracer_provider=tracer_provider)

        assert chat_complete.Completions.create is not original_chat_create
        assert chat_complete.AsyncCompletions.create is not original_async_chat_create
        assert generation.Completions.create is not original_prompt_create
        assert generation.AsyncCompletions.create is not original_async_prompt_create
    finally:
        instrumentor.uninstrument()

    assert chat_complete.Completions.create is original_chat_create
    assert chat_complete.AsyncCompletions.create is original_async_chat_create
    assert generation.Completions.create is original_prompt_create
    assert generation.AsyncCompletions.create is original_async_prompt_create
