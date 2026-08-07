import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import cohere
import pytest
from cohere.types import (
    AssistantMessageResponse,
    TextAssistantMessageResponseContentItem,
    ToolCallV2,
    ToolCallV2Function,
    ToolV2,
    ToolV2Function,
    Usage,
    UsageTokens,
)
from cohere.v2.raw_client import AsyncRawV2Client, RawV2Client
from cohere.v2.types import V2ChatResponse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import (
    REDACTED_VALUE,
    OITracer,
    TraceConfig,
    using_attributes,
)
from openinference.instrumentation.cohere import CohereInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


def _text_response() -> V2ChatResponse:
    return V2ChatResponse(
        id="c-0",
        finish_reason="COMPLETE",
        message=AssistantMessageResponse(
            role="assistant",
            content=[
                TextAssistantMessageResponseContentItem(
                    type="text", text="The sky is blue because of Rayleigh scattering."
                )
            ],
        ),
        usage=Usage(tokens=UsageTokens(input_tokens=26, output_tokens=12)),
    )


def _tool_response() -> V2ChatResponse:
    return V2ChatResponse(
        id="c-1",
        finish_reason="TOOL_CALL",
        message=AssistantMessageResponse(
            role="assistant",
            content=[],
            tool_calls=[
                ToolCallV2(
                    id="call-1",
                    type="function",
                    function=ToolCallV2Function(
                        name="get_current_weather",
                        arguments=json.dumps({"city": "Paris"}),
                    ),
                )
            ],
        ),
        usage=Usage(tokens=UsageTokens(input_tokens=40, output_tokens=8)),
    )


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(tracer_provider: TracerProvider) -> Iterator[None]:
    CohereInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    CohereInstrumentor().uninstrument()


def _client() -> "cohere.ClientV2":
    return cohere.ClientV2(api_key="fake-key")


def _user_message(content: str) -> Any:
    # A plain dict, typed as Any so mypy accepts it where the SDK expects
    # typed message objects; the extractor must handle both forms.
    return {"role": "user", "content": content}


def test_oitracer() -> None:
    assert isinstance(CohereInstrumentor()._tracer, OITracer)


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="cohere")
    assert isinstance(entrypoint.load()(), CohereInstrumentor)


def test_chat(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )

    response = _client().chat(
        model="command-a-03-2025",
        messages=[_user_message("Why is the sky blue?")],
        temperature=0.1,
    )
    assert response.message.content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert spans[0].name == "ClientV2.chat"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == "cohere"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "cohere"
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "command-a-03-2025"
    # Only parameters the caller actually set appear in the invocation parameters;
    # cohere's OMIT sentinel defaults must not leak in.
    assert json.loads(str(attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS])) == {"temperature": 0.1}
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    )
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "Why is the sky blue?"
    )
    assert "Rayleigh" in str(
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
    )
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 26
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 12
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 38
    assert SpanAttributes.INPUT_VALUE in attrs
    assert SpanAttributes.OUTPUT_VALUE in attrs


def test_chat_with_tool_call(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_tool_response())
    )

    tools: Any = [{"type": "function", "function": {"name": "get_current_weather"}}]
    _client().chat(
        model="command-a-03-2025",
        messages=[_user_message("What is the weather in Paris?")],
        tools=tools,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_ID}"] == "call-1"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_current_weather"
    raw_args = str(attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"])
    assert json.loads(raw_args) == {"city": "Paris"}
    assert attrs["llm.tools.0.tool.json_schema"]


def test_chat_records_tool_result_message(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool result message links back to the call that produced it."""
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )

    messages: Any = [
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "tool", "tool_call_id": "call-1", "content": '{"temperature_c": 18}'},
    ]
    _client().chat(model="command-a-03-2025", messages=messages)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.1"
    assert attrs[f"{prefix}.{MessageAttributes.MESSAGE_ROLE}"] == "tool"
    assert attrs[f"{prefix}.{MessageAttributes.MESSAGE_TOOL_CALL_ID}"] == "call-1"


def test_chat_error_span_keeps_model_name(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed calls still carry the request attributes, including the model."""

    def _raise(self: Any, **k: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(RawV2Client, "chat", _raise)

    with pytest.raises(RuntimeError):
        _client().chat(
            model="command-a-03-2025",
            messages=[_user_message("Why is the sky blue?")],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "command-a-03-2025"
    assert attrs[SpanAttributes.LLM_PROVIDER] == "cohere"


async def test_async_chat(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _mock_chat(self: Any, **k: Any) -> Any:
        return SimpleNamespace(data=_text_response())

    monkeypatch.setattr(AsyncRawV2Client, "chat", _mock_chat)

    await cohere.AsyncClientV2(api_key="fake-key").chat(
        model="command-a-03-2025",
        messages=[_user_message("Why is the sky blue?")],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "AsyncClientV2.chat"
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "command-a-03-2025"
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 38


async def test_async_chat_cancellation_ends_span(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _mock_chat(self: Any, **k: Any) -> Any:
        raise asyncio.CancelledError()

    monkeypatch.setattr(AsyncRawV2Client, "chat", _mock_chat)

    with pytest.raises(asyncio.CancelledError):
        await cohere.AsyncClientV2(api_key="fake-key").chat(
            model="command-a-03-2025",
            messages=[_user_message("Why is the sky blue?")],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


def test_suppress_tracing(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openinference.instrumentation import suppress_tracing

    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )
    with suppress_tracing():
        _client().chat(
            model="command-a-03-2025",
            messages=[_user_message("Why is the sky blue?")],
        )
    assert len(in_memory_span_exporter.get_finished_spans()) == 0


def test_context_attributes_propagation(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )
    with using_attributes(
        session_id="my-session",
        user_id="my-user",
        metadata={"env": "test"},
        tags=["tag-1", "tag-2"],
    ):
        _client().chat(
            model="command-a-03-2025",
            messages=[_user_message("Why is the sky blue?")],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.SESSION_ID] == "my-session"
    assert attrs[SpanAttributes.USER_ID] == "my-user"
    assert json.loads(str(attrs[SpanAttributes.METADATA])) == {"env": "test"}
    assert list(attrs[SpanAttributes.TAG_TAGS]) == ["tag-1", "tag-2"]  # type: ignore[arg-type]


def test_trace_config_masking(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )
    CohereInstrumentor().uninstrument()
    CohereInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
    )

    _client().chat(
        model="command-a-03-2025",
        messages=[_user_message("This input is sensitive.")],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    assert not any("sensitive" in str(value) for value in attrs.values())


def test_request_options_are_not_recorded(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`request_options` carries transport config, including credentials."""
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )

    _client().chat(
        model="command-a-03-2025",
        messages=[_user_message("Why is the sky blue?")],
        request_options={"additional_headers": {"Authorization": "Bearer super-secret-token"}},
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert "request_options" not in str(attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS])
    assert not any("super-secret-token" in str(value) for value in attrs.values())


def test_generator_messages_are_not_consumed(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extraction must not drain a one-shot iterator before the SDK reads it."""
    seen: List[Any] = []

    def _capture(self: Any, **kwargs: Any) -> Any:
        seen.extend(kwargs["messages"])
        return SimpleNamespace(data=_text_response())

    monkeypatch.setattr(RawV2Client, "chat", _capture)

    # Typed as Any because the SDK annotates `messages` as a list; the point of
    # the test is that a caller passing a generator is not broken by tracing.
    message_generator: Any = (m for m in [_user_message("Why is the sky blue?")])
    _client().chat(model="command-a-03-2025", messages=message_generator)

    # The SDK still received the message; the span simply omits input messages.
    assert len(seen) == 1
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1


def test_unserializable_argument_still_produces_a_span(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single unencodable argument must not drop the whole span."""
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )
    circular: Dict[str, Any] = {}
    circular["self"] = circular
    documents: Any = [circular]

    _client().chat(
        model="command-a-03-2025",
        messages=[_user_message("Why is the sky blue?")],
        documents=documents,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "command-a-03-2025"


def test_typed_tool_is_recorded_as_json(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pydantic tool objects must serialize as JSON, not as their repr."""
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )

    _client().chat(
        model="command-a-03-2025",
        messages=[_user_message("What is the weather?")],
        tools=[
            ToolV2(
                type="function",
                function=ToolV2Function(
                    name="get_weather",
                    description="Get the weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                ),
            )
        ],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    schema = json.loads(
        str(attrs[f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"])
    )
    assert schema["function"]["name"] == "get_weather"


def test_non_text_content_blocks_are_recorded(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tool-result blocks without a `text` field must not vanish."""
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )

    tool_message: Any = {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": [{"type": "document", "document": {"data": {"temperature_c": 18}}}],
    }
    _client().chat(model="command-a-03-2025", messages=[tool_message])

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    content = str(
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
    )
    assert "temperature_c" in content
