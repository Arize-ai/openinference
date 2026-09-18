import asyncio
import json
from typing import Any, Dict, Mapping, Optional, Type, Union, cast

from groq import AsyncGroq, Groq
from groq._base_client import _StreamT
from groq._types import Body, RequestFiles, RequestOptions, ResponseT
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from openinference.semconv.trace import SpanAttributes

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
    assert json.loads(attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS]) == {"model": MODEL}


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
    recorded = json.loads(attributes[SpanAttributes.INPUT_VALUE])
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
    recorded = json.loads(attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS])
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
    assert json.loads(attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS]) == {"model": MODEL}
