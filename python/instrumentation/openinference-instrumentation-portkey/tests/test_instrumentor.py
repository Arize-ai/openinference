import json
from importlib import import_module

import pytest
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
