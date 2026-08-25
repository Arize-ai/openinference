import asyncio
import base64
import json
import struct
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Sequence

import cohere
import pytest
from cohere.core.request_options import RequestOptions
from cohere.types import (
    ApiMeta,
    ApiMetaTokens,
    AssistantMessageResponse,
    EmbedByTypeResponse,
    EmbedByTypeResponseEmbeddings,
    EmbedInput,
    TextAssistantMessageResponseContentItem,
    TextEmbedContent,
    ToolCallV2,
    ToolCallV2Function,
    ToolV2,
    ToolV2Function,
    Usage,
    UsageTokens,
)
from cohere.v2.raw_client import AsyncRawV2Client, RawV2Client
from cohere.v2.types import (
    V2ChatResponse,
    V2RerankResponse,
    V2RerankResponseResultsItem,
)
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
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceSpanKindValues,
    RerankerAttributes,
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


def _vector(value: Any) -> List[float]:
    assert isinstance(value, Sequence)
    return [float(component) for component in value]


def _rerank_response() -> V2RerankResponse:
    return V2RerankResponse(
        id="rerank-1",
        results=[
            V2RerankResponseResultsItem(index=1, relevance_score=0.95),
            V2RerankResponseResultsItem(index=0, relevance_score=0.4),
        ],
    )


def _raw_stream(events: "List[Any]") -> Any:
    """Build a stand-in for ``RawV2Client.chat_stream``.

    The raw client returns a context manager whose ``data`` is the event iterator;
    ``V2Client.chat_stream`` consumes it as ``with ... as r: yield from r.data``.
    """

    @contextmanager
    def _chat_stream(self: Any, **kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(data=iter(events))

    return _chat_stream


def _raw_async_stream(events: "List[Any]") -> Any:
    """Async counterpart of :func:`_raw_stream`."""

    @asynccontextmanager
    async def _chat_stream(self: Any, **kwargs: Any) -> Any:
        async def _data() -> Any:
            for event in events:
                yield event

        yield SimpleNamespace(data=_data())

    return _chat_stream


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


def test_rerank_trace_config_masking(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RawV2Client,
        "rerank",
        lambda self, **kwargs: SimpleNamespace(data=_rerank_response()),
    )
    CohereInstrumentor().uninstrument()
    CohereInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
    )

    _client().rerank(
        model="rerank-v3.5",
        query="sensitive query",
        documents=["sensitive first document", "sensitive second document"],
        top_n=2,
    )

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    assert attrs[RerankerAttributes.RERANKER_QUERY] == REDACTED_VALUE
    assert attrs[RerankerAttributes.RERANKER_MODEL_NAME] == "rerank-v3.5"
    assert attrs[RerankerAttributes.RERANKER_TOP_K] == 2
    assert not any(
        key.startswith(RerankerAttributes.RERANKER_INPUT_DOCUMENTS)
        or key.startswith(RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS)
        for key in attrs
    )
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


def _stream_events() -> "list[Any]":
    """The event sequence cohere emits for a streamed text response."""
    return [
        SimpleNamespace(type="message-start", delta=None, index=None),
        SimpleNamespace(type="content-start", delta=None, index=0),
        SimpleNamespace(
            type="content-delta",
            index=0,
            delta=SimpleNamespace(
                message=SimpleNamespace(content=SimpleNamespace(text="The sky is blue "))
            ),
        ),
        SimpleNamespace(
            type="content-delta",
            index=0,
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    content=SimpleNamespace(text="because of Rayleigh scattering.")
                )
            ),
        ),
        SimpleNamespace(type="content-end", delta=None, index=0),
        SimpleNamespace(
            type="message-end",
            delta=SimpleNamespace(
                finish_reason="COMPLETE",
                usage=Usage(tokens=UsageTokens(input_tokens=26, output_tokens=12)),
            ),
        ),
    ]


def test_chat_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RawV2Client, "chat_stream", _raw_stream(_stream_events()))

    events = list(
        _client().chat_stream(
            model="command-a-03-2025",
            messages=[_user_message("Why is the sky blue?")],
        )
    )
    assert len(events) == 6

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert spans[0].name == "ClientV2.chat_stream"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "command-a-03-2025"
    assert attrs[SpanAttributes.OUTPUT_VALUE] == ("The sky is blue because of Rayleigh scattering.")
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "The sky is blue because of Rayleigh scattering."
    )
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 26
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 12
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 38


def test_chat_stream_span_is_open_until_consumed(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The span must not finish while the response is still streaming."""
    monkeypatch.setattr(RawV2Client, "chat_stream", _raw_stream(_stream_events()))

    stream = _client().chat_stream(
        model="command-a-03-2025",
        messages=[_user_message("Why is the sky blue?")],
    )
    next(iter(stream))
    assert in_memory_span_exporter.get_finished_spans() == ()

    list(stream)
    assert len(in_memory_span_exporter.get_finished_spans()) == 1


def test_chat_stream_with_tool_calls(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        SimpleNamespace(
            type="tool-call-start",
            index=0,
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=SimpleNamespace(
                        id="call-1",
                        function=SimpleNamespace(name="get_current_weather", arguments=""),
                    )
                )
            ),
        ),
        SimpleNamespace(
            type="tool-call-delta",
            index=0,
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=SimpleNamespace(
                        id=None, function=SimpleNamespace(name=None, arguments='{"city":')
                    )
                )
            ),
        ),
        SimpleNamespace(
            type="tool-call-delta",
            index=0,
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=SimpleNamespace(
                        id=None, function=SimpleNamespace(name=None, arguments=' "Paris"}')
                    )
                )
            ),
        ),
        SimpleNamespace(type="tool-call-end", index=0, delta=None),
        SimpleNamespace(
            type="message-end",
            delta=SimpleNamespace(
                finish_reason="TOOL_CALL",
                usage=Usage(tokens=UsageTokens(input_tokens=40, output_tokens=8)),
            ),
        ),
    ]
    monkeypatch.setattr(RawV2Client, "chat_stream", _raw_stream(events))

    list(
        _client().chat_stream(
            model="command-a-03-2025",
            messages=[_user_message("What is the weather in Paris?")],
        )
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_ID}"] == "call-1"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_current_weather"
    raw_args = str(attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"])
    assert json.loads(raw_args) == {"city": "Paris"}


def test_chat_stream_error_ends_span(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error mid-stream must still close the span with an error status."""

    @contextmanager
    def _failing_stream(self: Any, **kwargs: Any) -> Iterator[Any]:
        def _data() -> Iterator[Any]:
            yield _stream_events()[2]
            raise RuntimeError("stream broke")

        yield SimpleNamespace(data=_data())

    monkeypatch.setattr(RawV2Client, "chat_stream", _failing_stream)

    stream = _client().chat_stream(
        model="command-a-03-2025",
        messages=[_user_message("Why is the sky blue?")],
    )
    with pytest.raises(RuntimeError):
        list(stream)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code is StatusCode.ERROR
    assert dict(spans[0].attributes or {})[SpanAttributes.LLM_MODEL_NAME] == "command-a-03-2025"


async def test_async_chat_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(AsyncRawV2Client, "chat_stream", _raw_async_stream(_stream_events()))

    client = cohere.AsyncClientV2(api_key="fake-key")
    # `chat_stream` is an async generator function, so it is not awaited.
    stream = client.chat_stream(
        model="command-a-03-2025",
        messages=[_user_message("Why is the sky blue?")],
    )
    collected = [event async for event in stream]
    assert len(collected) == 6

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert spans[0].name == "AsyncClientV2.chat_stream"
    assert attrs[SpanAttributes.OUTPUT_VALUE] == ("The sky is blue because of Rayleigh scattering.")
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 38


def test_chat_stream_suppressed(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openinference.instrumentation import suppress_tracing

    monkeypatch.setattr(RawV2Client, "chat_stream", _raw_stream(_stream_events()))

    with suppress_tracing():
        list(
            _client().chat_stream(
                model="command-a-03-2025",
                messages=[_user_message("Why is the sky blue?")],
            )
        )

    assert in_memory_span_exporter.get_finished_spans() == ()


def test_embed(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = EmbedByTypeResponse(
        id="emb-0",
        embeddings=EmbedByTypeResponseEmbeddings(float_=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        texts=["Hello world", "Goodbye world"],
        meta=ApiMeta(tokens=ApiMetaTokens(input_tokens=5)),
    )

    monkeypatch.setattr(RawV2Client, "embed", lambda self, **k: SimpleNamespace(data=fake_response))

    _client().embed(
        model="embed-v4.0",
        texts=["Hello world", "Goodbye world"],
        input_type="search_document",
        embedding_types=["float"],
        output_dimension=256,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "CreateEmbeddings"
    assert span.status.status_code is StatusCode.OK
    attrs = dict(span.attributes or {})
    assert (
        attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.EMBEDDING.value
    )
    assert attrs.pop(SpanAttributes.EMBEDDING_MODEL_NAME) == "embed-v4.0"
    assert json.loads(str(attrs.pop(SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS))) == {
        "model": "embed-v4.0",
        "input_type": "search_document",
        "output_dimension": 256,
        "embedding_types": ["float"],
    }
    input_value = json.loads(str(attrs.pop(SpanAttributes.INPUT_VALUE)))
    assert input_value["texts"] == ["Hello world", "Goodbye world"]
    assert attrs.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    output_value = json.loads(str(attrs.pop(SpanAttributes.OUTPUT_VALUE)))
    assert output_value["embeddings"]["float"][0] == [0.1, 0.2, 0.3]
    assert attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert (
        attrs.pop(f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}")
        == "Hello world"
    )
    assert (
        attrs.pop(f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.1.{EmbeddingAttributes.EMBEDDING_TEXT}")
        == "Goodbye world"
    )
    assert _vector(
        attrs.pop(f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}")
    ) == [0.1, 0.2, 0.3]
    assert _vector(
        attrs.pop(f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.1.{EmbeddingAttributes.EMBEDDING_VECTOR}")
    ) == [0.4, 0.5, 0.6]
    assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 5
    assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 5
    assert not attrs


def test_embed_extracts_text_from_structured_inputs(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = EmbedByTypeResponse(
        id="emb-structured",
        embeddings=EmbedByTypeResponseEmbeddings(float_=[[0.1, 0.2]]),
    )
    monkeypatch.setattr(RawV2Client, "embed", lambda self, **k: SimpleNamespace(data=fake_response))

    _client().embed(
        model="embed-v4.0",
        inputs=[
            EmbedInput(
                content=[
                    TextEmbedContent(type="text", text="Hello "),
                    TextEmbedContent(type="text", text="world"),
                ]
            )
        ],
        input_type="search_document",
        embedding_types=["float"],
    )

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert (
        attrs[f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"]
        == "Hello world"
    )
    invocation_parameters = json.loads(str(attrs[SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS]))
    assert "inputs" not in invocation_parameters


def test_embed_decodes_base64_vectors(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_vector = base64.b64encode(struct.pack("<2f", 1.5, 2.0)).decode()
    fake_response = EmbedByTypeResponse(
        id="emb-base64",
        embeddings=EmbedByTypeResponseEmbeddings(base64=[encoded_vector]),
    )
    monkeypatch.setattr(RawV2Client, "embed", lambda self, **k: SimpleNamespace(data=fake_response))

    _client().embed(
        model="embed-v4.0",
        texts=["Hello world"],
        input_type="search_document",
        embedding_types=["base64"],
    )

    (span,) = in_memory_span_exporter.get_finished_spans()
    vector = dict(span.attributes or {})[
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
    ]
    assert _vector(vector) == [1.5, 2.0]


async def test_async_embed(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = EmbedByTypeResponse(
        id="emb-async",
        embeddings=EmbedByTypeResponseEmbeddings(float_=[[0.1, 0.2, 0.3]]),
    )

    async def _mock_embed(self: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(data=fake_response)

    monkeypatch.setattr(AsyncRawV2Client, "embed", _mock_embed)

    await cohere.AsyncClientV2(api_key="fake-key").embed(
        model="embed-v4.0",
        texts=["Hello world"],
        input_type="search_document",
        embedding_types=["float"],
    )

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.name == "CreateEmbeddings"
    attrs = dict(span.attributes or {})
    assert attrs[SpanAttributes.EMBEDDING_MODEL_NAME] == "embed-v4.0"
    assert _vector(
        attrs[f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"]
    ) == [0.1, 0.2, 0.3]


def test_embed_trace_config_masks_text_and_vectors(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = EmbedByTypeResponse(
        id="emb-masked",
        embeddings=EmbedByTypeResponseEmbeddings(float_=[[0.1, 0.2, 0.3]]),
    )
    monkeypatch.setattr(RawV2Client, "embed", lambda self, **k: SimpleNamespace(data=fake_response))
    CohereInstrumentor().uninstrument()
    CohereInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_embeddings_vectors=True, hide_embeddings_text=True),
    )

    _client().embed(
        model="embed-v4.0",
        texts=["Sensitive text"],
        input_type="search_document",
        embedding_types=["float"],
    )

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert (
        attrs[f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"]
        == REDACTED_VALUE
    )
    assert (
        attrs[f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"]
        == REDACTED_VALUE
    )


def test_embed_error_span_keeps_model_name(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(self: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(RawV2Client, "embed", _raise)

    with pytest.raises(RuntimeError):
        _client().embed(
            model="embed-v4.0",
            texts=["Hello world"],
            input_type="search_document",
            embedding_types=["float"],
        )

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert attrs[SpanAttributes.EMBEDDING_MODEL_NAME] == "embed-v4.0"
    assert SpanAttributes.LLM_PROVIDER not in attrs
    assert SpanAttributes.LLM_SYSTEM not in attrs


def test_embed_suppressed(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openinference.instrumentation import suppress_tracing

    fake_response = EmbedByTypeResponse(
        id="emb-suppressed",
        embeddings=EmbedByTypeResponseEmbeddings(float_=[[0.1, 0.2, 0.3]]),
    )
    monkeypatch.setattr(RawV2Client, "embed", lambda self, **k: SimpleNamespace(data=fake_response))

    with suppress_tracing():
        _client().embed(
            model="embed-v4.0",
            texts=["Hello world"],
            input_type="search_document",
            embedding_types=["float"],
        )

    assert in_memory_span_exporter.get_finished_spans() == ()


def test_rerank(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: Dict[str, Any] = {}

    def _mock_rerank(self: Any, **kwargs: Any) -> Any:
        received.update(kwargs)
        return SimpleNamespace(data=_rerank_response())

    monkeypatch.setattr(RawV2Client, "rerank", _mock_rerank)
    documents: Any = [
        "The first document",
        {"id": "doc-2", "title": "Second", "text": "The second document"},
    ]
    request_options: RequestOptions = {"additional_headers": {"x-test-secret": "not-recorded"}}

    with using_attributes(
        session_id="rerank-session",
        user_id="rerank-user",
        metadata={"env": "test"},
        tags=["rerank"],
    ):
        response = _client().rerank(
            model="rerank-v3.5",
            query="Which document is second?",
            documents=documents,
            top_n=2,
            max_tokens_per_doc=512,
            priority=1,
            request_options=request_options,
        )

    assert response.results[0].index == 1
    assert received["documents"] is documents
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.name == "ClientV2.rerank"
    assert span.status.status_code is StatusCode.OK
    attrs = dict(span.attributes or {})
    assert "not-recorded" not in json.dumps(attrs, default=str)
    assert (
        attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.RERANKER.value
    )
    assert attrs.pop(RerankerAttributes.RERANKER_QUERY) == "Which document is second?"
    assert attrs.pop(RerankerAttributes.RERANKER_MODEL_NAME) == "rerank-v3.5"
    assert attrs.pop(RerankerAttributes.RERANKER_TOP_K) == 2

    input_prefix = RerankerAttributes.RERANKER_INPUT_DOCUMENTS
    output_prefix = RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS
    assert (
        attrs.pop(f"{input_prefix}.0.{DocumentAttributes.DOCUMENT_CONTENT}") == "The first document"
    )
    structured_input = json.loads(
        str(attrs.pop(f"{input_prefix}.1.{DocumentAttributes.DOCUMENT_CONTENT}"))
    )
    assert structured_input == {
        "id": "doc-2",
        "title": "Second",
        "text": "The second document",
    }
    assert attrs.pop(f"{input_prefix}.1.{DocumentAttributes.DOCUMENT_ID}") == "doc-2"
    structured_output = json.loads(
        str(attrs.pop(f"{output_prefix}.0.{DocumentAttributes.DOCUMENT_CONTENT}"))
    )
    assert structured_output == structured_input
    assert attrs.pop(f"{output_prefix}.0.{DocumentAttributes.DOCUMENT_ID}") == "doc-2"
    assert attrs.pop(f"{output_prefix}.0.{DocumentAttributes.DOCUMENT_SCORE}") == 0.95
    assert (
        attrs.pop(f"{output_prefix}.1.{DocumentAttributes.DOCUMENT_CONTENT}")
        == "The first document"
    )
    assert attrs.pop(f"{output_prefix}.1.{DocumentAttributes.DOCUMENT_SCORE}") == 0.4

    input_value = json.loads(str(attrs.pop(SpanAttributes.INPUT_VALUE)))
    assert input_value["documents"] == documents
    assert input_value["max_tokens_per_doc"] == 512
    assert input_value["priority"] == 1
    assert "request_options" not in input_value
    assert attrs.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    output_value = json.loads(str(attrs.pop(SpanAttributes.OUTPUT_VALUE)))
    assert output_value["results"][0] == {"index": 1, "relevance_score": 0.95}
    assert attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attrs.pop(SpanAttributes.SESSION_ID) == "rerank-session"
    assert attrs.pop(SpanAttributes.USER_ID) == "rerank-user"
    assert json.loads(str(attrs.pop(SpanAttributes.METADATA))) == {"env": "test"}
    assert list(attrs.pop(SpanAttributes.TAG_TAGS)) == ["rerank"]  # type: ignore[arg-type]
    assert not attrs


async def test_async_rerank(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _mock_rerank(self: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(data=_rerank_response())

    monkeypatch.setattr(AsyncRawV2Client, "rerank", _mock_rerank)

    await cohere.AsyncClientV2(api_key="fake-key").rerank(
        model="rerank-v3.5",
        query="Which document is second?",
        documents=["The first document", "The second document"],
        top_n=2,
    )

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.name == "AsyncClientV2.rerank"
    assert span.status.status_code is StatusCode.OK
    attrs = dict(span.attributes or {})
    assert (
        attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.RERANKER.value
    )
    assert (
        attrs[
            f"{RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS}.0."
            f"{DocumentAttributes.DOCUMENT_CONTENT}"
        ]
        == "The second document"
    )


def test_rerank_does_not_drain_generator_documents(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_documents: List[str] = []

    def _mock_rerank(self: Any, **kwargs: Any) -> Any:
        received_documents.extend(kwargs["documents"])
        return SimpleNamespace(data=_rerank_response())

    monkeypatch.setattr(RawV2Client, "rerank", _mock_rerank)
    documents: Any = (document for document in ["The first document", "The second document"])

    _client().rerank(
        model="rerank-v3.5",
        query="Which document is second?",
        documents=documents,
        top_n=2,
    )

    assert received_documents == ["The first document", "The second document"]
    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert not any(
        key.startswith(RerankerAttributes.RERANKER_INPUT_DOCUMENTS)
        or key.startswith(RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS)
        for key in attrs
    )


def test_rerank_ignores_out_of_range_result_indices(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = V2RerankResponse(
        id="rerank-invalid-index",
        results=[V2RerankResponseResultsItem(index=3, relevance_score=0.8)],
    )
    monkeypatch.setattr(
        RawV2Client,
        "rerank",
        lambda self, **kwargs: SimpleNamespace(data=response),
    )

    result = _client().rerank(
        model="rerank-v3.5",
        query="Which document is second?",
        documents=["The only document"],
        top_n=1,
    )

    assert result is response
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.OK
    attrs = dict(span.attributes or {})
    assert not any(key.startswith(RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS) for key in attrs)


def test_rerank_keeps_core_attributes_when_documents_exceed_span_limit(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [f"Document {index}" for index in range(150)]
    response = V2RerankResponse(
        id="rerank-many-documents",
        results=[
            V2RerankResponseResultsItem(index=index, relevance_score=index / len(documents))
            for index in reversed(range(len(documents)))
        ],
    )
    monkeypatch.setattr(
        RawV2Client,
        "rerank",
        lambda self, **kwargs: SimpleNamespace(data=response),
    )

    with using_attributes(session_id="rerank-many-documents"):
        _client().rerank(
            model="rerank-v3.5",
            query="Which document ranks highest?",
            documents=documents,
            top_n=len(documents),
        )

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert (
        attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.RERANKER.value
    )
    assert attrs[RerankerAttributes.RERANKER_QUERY] == "Which document ranks highest?"
    assert attrs[RerankerAttributes.RERANKER_MODEL_NAME] == "rerank-v3.5"
    assert attrs[RerankerAttributes.RERANKER_TOP_K] == len(documents)
    assert attrs[SpanAttributes.SESSION_ID] == "rerank-many-documents"


def test_rerank_error_span_keeps_request_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(self: Any, **kwargs: Any) -> Any:
        raise RuntimeError("rerank failed")

    monkeypatch.setattr(RawV2Client, "rerank", _raise)

    with pytest.raises(RuntimeError, match="rerank failed"):
        _client().rerank(
            model="rerank-v3.5",
            query="Which document is second?",
            documents=["The first document", "The second document"],
            top_n=2,
        )

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    attrs = dict(span.attributes or {})
    assert (
        attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.RERANKER.value
    )
    assert attrs[RerankerAttributes.RERANKER_QUERY] == "Which document is second?"
    assert attrs[RerankerAttributes.RERANKER_MODEL_NAME] == "rerank-v3.5"
    assert attrs[RerankerAttributes.RERANKER_TOP_K] == 2


def test_rerank_suppressed(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openinference.instrumentation import suppress_tracing

    monkeypatch.setattr(
        RawV2Client,
        "rerank",
        lambda self, **kwargs: SimpleNamespace(data=_rerank_response()),
    )

    with suppress_tracing():
        _client().rerank(
            model="rerank-v3.5",
            query="Which document is second?",
            documents=["The first document", "The second document"],
            top_n=2,
        )

    assert in_memory_span_exporter.get_finished_spans() == ()
