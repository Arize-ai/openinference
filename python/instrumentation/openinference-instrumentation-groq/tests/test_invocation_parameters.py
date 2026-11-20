import asyncio
import json
from typing import Any, Dict, Mapping, Optional, Type, Union, cast

import pytest
from groq import AsyncGroq, Groq
from groq._base_client import _StreamT
from groq._types import Body, RequestFiles, RequestOptions, ResponseT
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import AttributeValue

MODEL = "llama-3.1-8b-instant"

MOCK_COMPLETION: Dict[str, Any] = {
    "id": "chat_comp_0",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": None,
            "message": {
                "content": "idk, sorry!",
                "role": "assistant",
                "function_call": None,
                "tool_calls": None,
            },
        }
    ],
    "created": 1722531851,
    "model": MODEL,
    "object": "chat.completion",
    "system_fingerprint": "fp0",
    "usage": {
        "completion_tokens": 5,
        "prompt_tokens": 5,
        "total_tokens": 10,
    },
}


def _mock_post(
    self: Any,
    path: str = "fake/url",
    *,
    cast_to: Type[ResponseT],
    body: Optional[Body] = None,
    options: RequestOptions = {},
    files: Optional[RequestFiles] = None,
    stream: bool = False,
    stream_cls: Optional[Type[_StreamT]] = None,
) -> Union[ResponseT, _StreamT]:
    return cast(ResponseT, MOCK_COMPLETION)


async def _async_mock_post(
    self: Any,
    path: str = "fake/url",
    *,
    cast_to: Type[ResponseT],
    body: Optional[Body] = None,
    options: RequestOptions = {},
    files: Optional[RequestFiles] = None,
    stream: bool = False,
    stream_cls: Optional[Type[_StreamT]] = None,
) -> Union[ResponseT, _StreamT]:
    return cast(ResponseT, MOCK_COMPLETION)


def _llm_span_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
) -> Dict[str, AttributeValue]:
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    return dict(cast(Mapping[str, AttributeValue], spans[0].attributes))


def test_parameters_the_caller_omitted_are_not_recorded(
    setup_groq_instrumentation: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """
    Only what the caller passed belongs in `llm.invocation_parameters`: the SDK's own
    "not given" markers for the ~34 optional parameters are not user input.
    """
    client = Groq(api_key="fake")
    client.chat.completions._post = _mock_post  # type: ignore[assignment]
    client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}],
        model=MODEL,
    )
    attributes = _llm_span_attributes(in_memory_span_exporter)
    assert json.loads(cast(str, attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS])) == {
        "model": MODEL
    }


def test_parameters_the_caller_omitted_are_not_recorded_in_input_value(
    setup_groq_instrumentation: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = Groq(api_key="fake")
    client.chat.completions._post = _mock_post  # type: ignore[assignment]
    client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}],
        model=MODEL,
    )
    attributes = _llm_span_attributes(in_memory_span_exporter)
    recorded = json.loads(cast(str, attributes[SpanAttributes.INPUT_VALUE]))
    assert sorted(recorded) == ["messages", "model"]


def test_no_unset_sentinel_repr_leaks_into_recorded_values(
    setup_groq_instrumentation: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """
    The markers serialise to a memory address, so a leaked one makes the same call
    record a different value in every process.
    """
    client = Groq(api_key="fake")
    client.chat.completions._post = _mock_post  # type: ignore[assignment]
    client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}],
        model=MODEL,
    )
    attributes = _llm_span_attributes(in_memory_span_exporter)
    for key in (SpanAttributes.LLM_INVOCATION_PARAMETERS, SpanAttributes.INPUT_VALUE):
        assert "object at 0x" not in str(attributes[key])


def test_parameters_the_caller_passed_are_still_recorded(
    setup_groq_instrumentation: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = Groq(api_key="fake")
    client.chat.completions._post = _mock_post  # type: ignore[assignment]
    client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}],
        model=MODEL,
        temperature=0.5,
        max_tokens=10,
        stop="END",
    )
    attributes = _llm_span_attributes(in_memory_span_exporter)
    recorded = json.loads(cast(str, attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS]))
    assert recorded == {"model": MODEL, "temperature": 0.5, "max_tokens": 10, "stop": "END"}


def test_async_path_records_only_the_parameters_the_caller_passed(
    setup_groq_instrumentation: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = AsyncGroq(api_key="fake")
    client.chat.completions._post = _async_mock_post  # type: ignore[assignment]

    async def exec_completion() -> None:
        await client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            model=MODEL,
        )

    asyncio.run(exec_completion())
    attributes = _llm_span_attributes(in_memory_span_exporter)
    assert json.loads(cast(str, attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS])) == {
        "model": MODEL
    }


def _assert_interrupted_span(span: ReadableSpan, exception_type: Type[BaseException]) -> None:
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description is not None
    assert span.status.description.startswith(f"{exception_type.__name__}:")
    assert len(span.events) == 1
    exception_event = span.events[0]
    assert exception_event.name == "exception"
    assert exception_event.attributes is not None
    expected_exception_types = (
        {"CancelledError", "asyncio.exceptions.CancelledError"}
        if exception_type is asyncio.CancelledError
        else {exception_type.__name__}
    )
    assert exception_event.attributes["exception.type"] in expected_exception_types

    attributes = dict(span.attributes or {})
    assert (
        attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == OpenInferenceLLMProviderValues.GROQ.value
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(cast(str, attributes.pop(SpanAttributes.INPUT_VALUE))) == {
        "messages": [{"role": "user", "content": "hi"}],
        "model": MODEL,
    }
    assert json.loads(cast(str, attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS))) == {
        "model": MODEL
    }
    message = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    assert attributes.pop(f"{message}.{MessageAttributes.MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{message}.{MessageAttributes.MESSAGE_CONTENT}") == "hi"
    assert not attributes


def test_sync_call_interrupted_by_base_exception_ends_span(
    setup_groq_instrumentation: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def _interrupted_post(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    client = Groq(api_key="fake")
    client.chat.completions._post = _interrupted_post  # type: ignore[assignment]
    with pytest.raises(KeyboardInterrupt):
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            model=MODEL,
        )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_interrupted_span(spans[0], KeyboardInterrupt)


def test_async_call_cancelled_ends_span(
    setup_groq_instrumentation: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async def _cancelled_post(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError

    client = AsyncGroq(api_key="fake")
    client.chat.completions._post = _cancelled_post  # type: ignore[assignment]

    async def call() -> None:
        await client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            model=MODEL,
        )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(call())
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_interrupted_span(spans[0], asyncio.CancelledError)
