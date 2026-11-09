import json
from typing import Any, Dict, List

import pytest
from mem0 import AsyncMemory, AsyncMemoryClient, Memory, MemoryClient
from mem0.client.types import SearchMemoryOptions
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    suppress_tracing,
    using_attributes,
)
from openinference.instrumentation.config import REDACTED_VALUE
from openinference.instrumentation.mem0 import Mem0Instrumentor
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

# Must match the canned fact returned by the conftest FakeLLM and platform transport.
EXTRACTED_FACT = "Likes to play tennis on weekends"

CHAIN = OpenInferenceSpanKindValues.CHAIN.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value


def _attrs(span: ReadableSpan) -> Dict[str, Any]:
    return dict(span.attributes or {})


def _spans_by_name(exporter: InMemorySpanExporter) -> Dict[str, ReadableSpan]:
    spans = exporter.get_finished_spans()
    by_name = {span.name: span for span in spans}
    assert len(by_name) == len(spans), [span.name for span in spans]
    return by_name


def _document(attrs: Dict[str, Any], index: int, field: str) -> Any:
    return attrs.get(f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}.{field}")


def _seed(memory: Memory, exporter: InMemorySpanExporter) -> str:
    """Adds one inferred memory and clears the exported spans; returns the memory id."""
    result = memory.add("I love playing tennis every weekend", user_id="alice")
    exporter.clear()
    memory_id: str = result["results"][0]["id"]
    return memory_id


def test_oitracer() -> None:
    assert isinstance(Mem0Instrumentor()._tracer, OITracer)


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="mem0")
    assert isinstance(entrypoint.load()(), Mem0Instrumentor)


def test_add_emits_chain_span(
    memory: Memory, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    result = memory.add(
        [{"role": "user", "content": "I love playing tennis every weekend"}],
        user_id="alice",
        agent_id="coach",
        run_id="run-1",
        metadata={"source": "chat"},
    )
    assert memory.llm.calls == 1  # the stub fact-extraction LLM ran once

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = _attrs(span)
    assert span.name == "Memory.add"
    assert span.status.status_code is StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == CHAIN
    assert attrs[SpanAttributes.INPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(attrs[SpanAttributes.INPUT_VALUE]) == {
        "messages": [{"role": "user", "content": "I love playing tennis every weekend"}],
        "user_id": "alice",
        "agent_id": "coach",
        "run_id": "run-1",
        "metadata": {"source": "chat"},
    }
    assert json.loads(attrs[SpanAttributes.METADATA]) == {
        "user_id": "alice",
        "agent_id": "coach",
        "run_id": "run-1",
    }
    assert attrs[SpanAttributes.OUTPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(attrs[SpanAttributes.OUTPUT_VALUE]) == result
    assert result["results"][0]["memory"] == EXTRACTED_FACT


def test_search_emits_retriever_span_with_documents(
    memory: Memory, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    memory_id = _seed(memory, in_memory_span_exporter)
    result = memory.search("tennis", filters={"user_id": "alice"}, top_k=5)

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = _attrs(span)
    assert span.name == "Memory.search"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == RETRIEVER
    assert attrs[SpanAttributes.INPUT_VALUE] == "tennis"
    assert attrs[SpanAttributes.INPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.TEXT.value
    assert json.loads(attrs[SpanAttributes.METADATA]) == {"user_id": "alice"}
    # Retrieved memories are recorded as documents rather than repeated in output.value.
    assert SpanAttributes.OUTPUT_VALUE not in attrs
    (hit,) = result["results"]
    assert _document(attrs, 0, DocumentAttributes.DOCUMENT_ID) == memory_id
    assert _document(attrs, 0, DocumentAttributes.DOCUMENT_CONTENT) == EXTRACTED_FACT
    assert _document(attrs, 0, DocumentAttributes.DOCUMENT_SCORE) == pytest.approx(hit["score"])
    metadata = json.loads(_document(attrs, 0, DocumentAttributes.DOCUMENT_METADATA))
    assert metadata["user_id"] == "alice"
    assert "memory" not in metadata and "score" not in metadata


def test_get_all_emits_retriever_span(
    memory: Memory, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    _seed(memory, in_memory_span_exporter)
    memory.add("raw note", user_id="alice", infer=False)
    in_memory_span_exporter.clear()

    memory.get_all(filters={"user_id": "alice"})

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = _attrs(span)
    assert span.name == "Memory.get_all"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == RETRIEVER
    assert json.loads(attrs[SpanAttributes.INPUT_VALUE]) == {"filters": {"user_id": "alice"}}
    contents = {_document(attrs, index, DocumentAttributes.DOCUMENT_CONTENT) for index in range(2)}
    assert contents == {EXTRACTED_FACT, "raw note"}


def test_get_update_history_delete_emit_chain_spans(
    memory: Memory, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    memory_id = _seed(memory, in_memory_span_exporter)

    got = memory.get(memory_id)
    memory.update(memory_id, text="Plays tennis on Sundays")
    history = memory.history(memory_id)
    memory.delete(memory_id)
    memory.delete_all(user_id="alice")

    spans = _spans_by_name(in_memory_span_exporter)
    assert set(spans) == {
        "Memory.get",
        "Memory.update",
        "Memory.history",
        "Memory.delete",
        "Memory.delete_all",
    }
    for span in spans.values():
        assert _attrs(span)[SpanAttributes.OPENINFERENCE_SPAN_KIND] == CHAIN
        assert span.status.status_code is StatusCode.OK

    get_attrs = _attrs(spans["Memory.get"])
    assert json.loads(get_attrs[SpanAttributes.INPUT_VALUE]) == {"memory_id": memory_id}
    assert json.loads(get_attrs[SpanAttributes.OUTPUT_VALUE]) == got

    update_attrs = _attrs(spans["Memory.update"])
    assert json.loads(update_attrs[SpanAttributes.INPUT_VALUE]) == {
        "memory_id": memory_id,
        "text": "Plays tennis on Sundays",
    }
    assert json.loads(update_attrs[SpanAttributes.OUTPUT_VALUE]) == {
        "message": "Memory updated successfully!"
    }

    history_attrs = _attrs(spans["Memory.history"])
    assert json.loads(history_attrs[SpanAttributes.OUTPUT_VALUE]) == json.loads(
        json.dumps(history, default=str)
    )

    delete_all_attrs = _attrs(spans["Memory.delete_all"])
    assert json.loads(delete_all_attrs[SpanAttributes.METADATA]) == {"user_id": "alice"}


def test_get_missing_memory_has_no_output(
    memory: Memory, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    assert memory.get("00000000-0000-0000-0000-000000000000") is None
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.OK
    assert SpanAttributes.OUTPUT_VALUE not in _attrs(span)


def test_error_is_recorded(memory: Memory, in_memory_span_exporter: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError, match="filters must contain"):
        memory.get_all(filters={})

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.name == "Memory.get_all"
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description is not None
    assert span.status.description.startswith("ValueError: filters must contain")
    (event,) = span.events
    assert event.name == "exception"


def test_llm_failure_is_recorded(
    memory: Memory, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    class FailingLLM:
        def generate_response(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
            raise RuntimeError("provider unavailable")

    memory.llm = FailingLLM()
    with pytest.raises(Exception, match="provider unavailable"):
        memory.add("hello", user_id="alice")

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.name == "Memory.add"
    assert span.status.status_code is StatusCode.ERROR


async def test_async_memory(
    async_memory: AsyncMemory, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    added = await async_memory.add("I love playing tennis every weekend", user_id="bob")
    memory_id = added["results"][0]["id"]
    await async_memory.search("tennis", filters={"user_id": "bob"})
    await async_memory.get(memory_id)
    await async_memory.get_all(filters={"user_id": "bob"})
    await async_memory.update(memory_id, text="Plays tennis on Sundays")
    await async_memory.history(memory_id)
    await async_memory.delete(memory_id)
    await async_memory.delete_all(user_id="bob")

    spans = _spans_by_name(in_memory_span_exporter)
    assert set(spans) == {
        f"AsyncMemory.{name}"
        for name in ("add", "search", "get", "get_all", "update", "history", "delete", "delete_all")
    }
    for name, span in spans.items():
        attrs = _attrs(span)
        expected_kind = RETRIEVER if name.endswith(("search", "get_all")) else CHAIN
        assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == expected_kind
        assert span.status.status_code is StatusCode.OK
    search_attrs = _attrs(spans["AsyncMemory.search"])
    assert _document(search_attrs, 0, DocumentAttributes.DOCUMENT_CONTENT) == EXTRACTED_FACT
    assert json.loads(_attrs(spans["AsyncMemory.add"])[SpanAttributes.METADATA]) == {
        "user_id": "bob"
    }


def test_memory_client(
    memory_client: MemoryClient, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    memory_client.add("I love tennis", filters={"user_id": "alice"})
    memory_client.search("tennis", SearchMemoryOptions(filters={"user_id": "alice"}, top_k=3))

    spans = _spans_by_name(in_memory_span_exporter)
    assert set(spans) == {"MemoryClient.add", "MemoryClient.search"}

    add_attrs = _attrs(spans["MemoryClient.add"])
    assert add_attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == CHAIN
    assert json.loads(add_attrs[SpanAttributes.INPUT_VALUE]) == {
        "messages": "I love tennis",
        "filters": {"user_id": "alice"},
    }
    assert json.loads(add_attrs[SpanAttributes.METADATA]) == {"user_id": "alice"}

    search_attrs = _attrs(spans["MemoryClient.search"])
    assert search_attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == RETRIEVER
    assert search_attrs[SpanAttributes.INPUT_VALUE] == "tennis"
    # The typed `options` model is read like keyword arguments.
    assert json.loads(search_attrs[SpanAttributes.METADATA]) == {"user_id": "alice"}
    assert _document(search_attrs, 0, DocumentAttributes.DOCUMENT_ID) == "mem-1"
    assert _document(search_attrs, 0, DocumentAttributes.DOCUMENT_SCORE) == 0.9


def test_memory_client_error(
    memory_client: MemoryClient, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with pytest.raises(Exception):
        memory_client.get("missing")
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.name == "MemoryClient.get"
    assert span.status.status_code is StatusCode.ERROR


async def test_async_memory_client(
    async_memory_client: AsyncMemoryClient, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    await async_memory_client.search("tennis", filters={"user_id": "alice"})
    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = _attrs(span)
    assert span.name == "AsyncMemoryClient.search"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == RETRIEVER
    assert _document(attrs, 0, DocumentAttributes.DOCUMENT_CONTENT) == EXTRACTED_FACT


def test_suppress_tracing(memory: Memory, in_memory_span_exporter: InMemorySpanExporter) -> None:
    with suppress_tracing():
        memory.add("raw note", user_id="alice", infer=False)
        memory.get_all(filters={"user_id": "alice"})
    assert not in_memory_span_exporter.get_finished_spans()


def test_context_attributes(memory: Memory, in_memory_span_exporter: InMemorySpanExporter) -> None:
    with using_attributes(
        session_id="session-1",
        user_id="app-user",
        metadata={"tier": "pro", "user_id": "from-context"},
        tags=["memory"],
    ):
        memory.add("raw note", user_id="alice", agent_id="coach", infer=False)

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = _attrs(span)
    assert attrs[SpanAttributes.SESSION_ID] == "session-1"
    assert attrs[SpanAttributes.USER_ID] == "app-user"
    assert list(attrs[SpanAttributes.TAG_TAGS]) == ["memory"]
    # mem0 scope ids are merged into context metadata; context values take precedence.
    assert json.loads(attrs[SpanAttributes.METADATA]) == {
        "tier": "pro",
        "user_id": "from-context",
        "agent_id": "coach",
    }


def test_context_metadata_kept_without_scope(
    memory: Memory, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with using_attributes(metadata={"tier": "pro"}):
        memory.history("00000000-0000-0000-0000-000000000000")
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert json.loads(_attrs(span)[SpanAttributes.METADATA]) == {"tier": "pro"}


@pytest.mark.parametrize("hide_inputs", [False, True])
@pytest.mark.parametrize("hide_outputs", [False, True])
def test_trace_config_masking(
    hide_inputs: bool,
    hide_outputs: bool,
    memory: Memory,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    Mem0Instrumentor().uninstrument()
    Mem0Instrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=hide_inputs, hide_outputs=hide_outputs),
    )
    memory.add("I love playing tennis every weekend", user_id="alice")
    memory.search("tennis", filters={"user_id": "alice"})

    spans = _spans_by_name(in_memory_span_exporter)
    add_attrs = _attrs(spans["Memory.add"])
    search_attrs = _attrs(spans["Memory.search"])
    if hide_inputs:
        assert add_attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
        assert search_attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
        assert SpanAttributes.INPUT_MIME_TYPE not in add_attrs
    else:
        assert "tennis" in add_attrs[SpanAttributes.INPUT_VALUE]
        assert search_attrs[SpanAttributes.INPUT_VALUE] == "tennis"
    if hide_outputs:
        assert add_attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
        assert _document(search_attrs, 0, DocumentAttributes.DOCUMENT_CONTENT) == REDACTED_VALUE
    else:
        assert EXTRACTED_FACT in add_attrs[SpanAttributes.OUTPUT_VALUE]
        assert _document(search_attrs, 0, DocumentAttributes.DOCUMENT_CONTENT) == EXTRACTED_FACT


def test_uninstrument_restores_methods(
    memory: Memory, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    Mem0Instrumentor().uninstrument()
    assert not hasattr(Memory.add, "__wrapped__")
    assert not hasattr(AsyncMemory.search, "__wrapped__")
    memory.add("raw note", user_id="alice", infer=False)
    assert not in_memory_span_exporter.get_finished_spans()
    Mem0Instrumentor().instrument()
