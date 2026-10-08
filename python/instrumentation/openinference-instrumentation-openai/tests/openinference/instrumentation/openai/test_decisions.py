"""Tests for the OpenAI Decisions API (`client.decisions.create`).

Decision calls are recorded as DECISION spans: the model is identified under `decision.*`
rather than `llm.*`, the request and response bodies are recorded as `input.value` and
`output.value`, and no `llm.*` attribute is emitted. See spec/decision_spans.md.

The Decisions API was added in openai 3.26.0; on older SDKs this module is skipped.
"""

import asyncio
import json
from contextlib import suppress
from importlib import import_module
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Mapping, cast
from urllib.parse import urljoin

import pytest
from httpx import Response
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respx import MockRouter

from openinference.instrumentation import (
    REDACTED_VALUE,
    TraceConfig,
    suppress_tracing,
    using_attributes,
)
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.openai._types import AttributeValue
from openinference.instrumentation.openai._utils import _get_decision_type
from openinference.semconv.trace import (
    OpenInferenceDecisionProviderValues,
    OpenInferenceDecisionSystemValues,
    OpenInferenceLLMProviderValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

openai = import_module("openai")

pytestmark = pytest.mark.skipif(
    _get_decision_type(openai) is None,
    reason="The Decisions API requires openai>=3.26.0",
)

_OPENAI_BASE_URL = "https://api.openai.com/v1/"
_AZURE_BASE_URL = "https://aoairesource.openai.azure.com"
_OPENINFERENCE_SCOPE = "openinference.instrumentation.openai"

_REQUEST_MODEL = "gpt-6-luna"
_RESPONSE_MODEL = "gpt-6-luna-2026-10-01"
_SPAN_NAME = "Decision"

# A 1x1 PNG, the inline data URL form the Decisions API requires for images.
_IMAGE_DATA_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

_USAGE: Dict[str, Any] = {
    "input_tokens": 412,
    "output_tokens": 3,
    "total_tokens": 415,
    "input_tokens_details": {"cached_tokens": 128, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}

# Request questions and the answers the API returns for them, keyed by question type.
_QUESTIONS: Dict[str, List[Dict[str, Any]]] = {
    "predicate": [
        {"type": "predicate", "name": "is_refund", "instructions": "Is this a refund request?"},
    ],
    "choice": [
        {
            "type": "choice",
            "name": "department",
            "instructions": "Which department should handle this?",
            "choices": [
                {"value": "billing", "description": "Charges and refunds"},
                {"value": "shipping"},
                {"value": False, "description": "No department applies"},
            ],
        },
    ],
    "score": [
        {
            "type": "score",
            "name": "urgency",
            "instructions": "How urgent is this message?",
            "levels": [
                {"label": "low", "description": "Can wait a week"},
                {"label": "medium"},
                {"label": "high", "description": "Needs a reply today"},
            ],
        },
    ],
}
_QUESTIONS["mixed"] = [*_QUESTIONS["predicate"], *_QUESTIONS["choice"], *_QUESTIONS["score"]]
# The host may decline a question; the answer is then a refusal carrying only the name.
_QUESTIONS["refusal"] = [
    {"type": "predicate", "name": "is_refund", "instructions": "Is this a refund request?"},
    {"type": "predicate", "instructions": "Is the sender a minor?"},
]

_ANSWERS: Dict[str, List[Dict[str, Any]]] = {
    "predicate": [{"type": "predicate", "name": "is_refund", "probability": 0.93}],
    "choice": [
        {
            "type": "choice",
            "name": "department",
            "choice": "billing",
            "confidence": 0.81,
            "probabilities": [
                {"value": "billing", "probability": 0.81},
                {"value": "shipping", "probability": 0.14},
                {"value": False, "probability": 0.05},
            ],
        },
    ],
    "score": [
        {
            "type": "score",
            "name": "urgency",
            "score": 1.7,
            "confidence": 0.62,
            "probabilities": [
                {"value": 0, "label": "low", "probability": 0.1},
                {"value": 1, "label": "medium", "probability": 0.2},
                {"value": 2, "label": "high", "probability": 0.7},
            ],
        },
    ],
}
_ANSWERS["mixed"] = [*_ANSWERS["predicate"], *_ANSWERS["choice"], *_ANSWERS["score"]]
_ANSWERS["refusal"] = [
    {"type": "predicate", "name": "is_refund", "probability": 0.93},
    {"type": "refusal", "name": None},
]

_INPUT_TEXT = "Hi, I was charged twice for order #4242 and would like my money back."


def _decision_json(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"model": _RESPONSE_MODEL, "answers": answers, "usage": _USAGE}


def _error_json() -> Dict[str, Any]:
    return {
        "error": {
            "message": "Unsupported question type.",
            "type": "invalid_request_error",
            "param": "questions",
            "code": None,
        }
    }


def _client(is_async: bool, base_url: str = _OPENAI_BASE_URL) -> Any:
    if is_async:
        return openai.AsyncOpenAI(api_key="sk-", base_url=base_url)
    return openai.OpenAI(api_key="sk-", base_url=base_url)


def _create(
    is_async: bool,
    create_kwargs: Mapping[str, Any],
    base_url: str = _OPENAI_BASE_URL,
) -> Any:
    """Call `decisions.create` with the sync or async client and return the response."""
    client = _client(is_async, base_url)
    if is_async:

        async def task() -> Any:
            return await client.decisions.create(**create_kwargs)

        return asyncio.run(task())
    return client.decisions.create(**create_kwargs)


def _openinference_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = tuple(
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope is not None
        and span.instrumentation_scope.name == _OPENINFERENCE_SCOPE
    )
    assert len(spans) == 1
    span = spans[0]
    assert span.name == _SPAN_NAME
    return span


def _attributes(span: ReadableSpan) -> Dict[str, AttributeValue]:
    return dict(cast(Mapping[str, AttributeValue], span.attributes))


def _pop_common_attributes(
    attributes: Dict[str, AttributeValue],
    provider: str = OpenInferenceDecisionProviderValues.OPENAI.value,
) -> None:
    """Pop the attributes every decision span carries regardless of outcome or config."""
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND, None) == (
        OpenInferenceSpanKindValues.DECISION.value
    )
    assert attributes.pop(SpanAttributes.DECISION_SYSTEM, None) == (
        OpenInferenceDecisionSystemValues.OPENAI.value
    )
    assert attributes.pop(SpanAttributes.DECISION_PROVIDER, None) == provider
    assert attributes.pop(SpanAttributes.DECISION_REQUEST_MODEL_NAME, None) == _REQUEST_MODEL


def _pop_input(attributes: Dict[str, AttributeValue], expected: Mapping[str, Any]) -> None:
    input_value = attributes.pop(SpanAttributes.INPUT_VALUE, None)
    assert isinstance(input_value, str)
    assert json.loads(input_value) == expected
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(SpanAttributes.INPUT_MIME_TYPE, None))
        == OpenInferenceMimeTypeValues.JSON
    )


def _pop_output(attributes: Dict[str, AttributeValue], expected: Mapping[str, Any]) -> None:
    output_value = attributes.pop(SpanAttributes.OUTPUT_VALUE, None)
    assert isinstance(output_value, str)
    assert json.loads(output_value) == expected
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE, None))
        == OpenInferenceMimeTypeValues.JSON
    )


def _pop_response_attributes(attributes: Dict[str, AttributeValue]) -> None:
    assert attributes.pop(SpanAttributes.DECISION_RESPONSE_MODEL_NAME, None) == _RESPONSE_MODEL
    assert attributes.pop(SpanAttributes.DECISION_MODEL_NAME, None) == _RESPONSE_MODEL
    input_tokens, output_tokens = _USAGE["input_tokens"], _USAGE["output_tokens"]
    assert attributes.pop(SpanAttributes.DECISION_TOKEN_COUNT_INPUT, None) == input_tokens
    assert attributes.pop(SpanAttributes.DECISION_TOKEN_COUNT_OUTPUT, None) == output_tokens


def _assert_no_llm_attributes(attributes: Mapping[str, AttributeValue]) -> None:
    # Decision spans must not be counted as LLM usage: no llm.* attribute at all, in
    # particular not llm.provider, llm.system, llm.model_name, llm.invocation_parameters
    # or llm.token_count.*.
    assert not [key for key in attributes if key.startswith("llm.")]


@pytest.fixture
def custom_instrumentation(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[Callable[[TraceConfig], None]]:
    """Re-instrument with a custom TraceConfig for the duration of one test."""
    OpenAIInstrumentor().uninstrument()

    def _instrument(config: TraceConfig) -> None:
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)

    yield _instrument
    OpenAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("kind", ["predicate", "choice", "score", "mixed", "refusal"])
def test_decisions(
    is_async: bool,
    kind: str,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    response_json = _decision_json(_ANSWERS[kind])
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "decisions")).mock(
        return_value=Response(status_code=200, json=response_json)
    )
    create_kwargs: Dict[str, Any] = {
        "model": _REQUEST_MODEL,
        "input": _INPUT_TEXT,
        "questions": _QUESTIONS[kind],
    }
    decision = _create(is_async, create_kwargs)
    assert decision.model == _RESPONSE_MODEL
    assert [answer.type for answer in decision.answers] == [
        answer["type"] for answer in _ANSWERS[kind]
    ]

    span = _openinference_span(in_memory_span_exporter)
    assert span.status.is_ok
    assert not span.status.description
    attributes = _attributes(span)
    _assert_no_llm_attributes(attributes)
    _pop_common_attributes(attributes)
    _pop_input(attributes, create_kwargs)
    _pop_output(attributes, response_json)
    _pop_response_attributes(attributes)
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.parametrize("is_async", [False, True])
def test_decisions_with_image_input(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    response_json = _decision_json(_ANSWERS["choice"])
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "decisions")).mock(
        return_value=Response(status_code=200, json=response_json)
    )
    create_kwargs: Dict[str, Any] = {
        "model": _REQUEST_MODEL,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Which department does this receipt go to?"},
                    {"type": "input_image", "image_url": _IMAGE_DATA_URL, "detail": "low"},
                ],
            }
        ],
        "questions": _QUESTIONS["choice"],
        "safety_identifier": "user-1234",
    }
    _create(is_async, create_kwargs)

    span = _openinference_span(in_memory_span_exporter)
    assert span.status.is_ok
    attributes = _attributes(span)
    _assert_no_llm_attributes(attributes)
    _pop_common_attributes(attributes)
    # The inline image is part of the request body and is recorded verbatim by default.
    assert _IMAGE_DATA_URL in cast(str, attributes[SpanAttributes.INPUT_VALUE])
    _pop_input(attributes, create_kwargs)
    _pop_output(attributes, response_json)
    _pop_response_attributes(attributes)
    assert attributes == {}


@pytest.mark.parametrize("is_async", [False, True])
def test_decisions_error(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "decisions")).mock(
        return_value=Response(status_code=400, json=_error_json())
    )
    create_kwargs: Dict[str, Any] = {
        "model": _REQUEST_MODEL,
        "input": _INPUT_TEXT,
        "questions": _QUESTIONS["predicate"],
    }
    with pytest.raises(openai.BadRequestError):
        _create(is_async, create_kwargs)

    span = _openinference_span(in_memory_span_exporter)
    assert not span.status.is_ok and not span.status.is_unset
    assert span.status.description and span.status.description.startswith(
        openai.BadRequestError.__name__
    )
    assert len(span.events) == 1
    assert span.events[0].name == "exception"
    attributes = _attributes(span)
    _assert_no_llm_attributes(attributes)
    _pop_common_attributes(attributes)
    _pop_input(attributes, create_kwargs)
    # Without a response, decision.model_name falls back to the requested model.
    assert attributes.pop(SpanAttributes.DECISION_MODEL_NAME, None) == _REQUEST_MODEL
    assert attributes == {}


def test_decisions_with_raw_response(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    response_json = _decision_json(_ANSWERS["score"])
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "decisions")).mock(
        return_value=Response(status_code=200, json=response_json)
    )
    create_kwargs: Dict[str, Any] = {
        "model": _REQUEST_MODEL,
        "input": _INPUT_TEXT,
        "questions": _QUESTIONS["score"],
    }
    raw = _client(is_async=False).decisions.with_raw_response.create(**create_kwargs)
    assert raw.parse().model == _RESPONSE_MODEL

    span = _openinference_span(in_memory_span_exporter)
    assert span.status.is_ok
    attributes = _attributes(span)
    _assert_no_llm_attributes(attributes)
    _pop_common_attributes(attributes)
    _pop_input(attributes, create_kwargs)
    _pop_output(attributes, response_json)
    _pop_response_attributes(attributes)
    assert attributes == {}


def test_decisions_provider_is_inferred_from_host(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """decision.provider reuses the host inference that llm.provider uses on LLM spans."""
    respx_mock.post(urljoin(_AZURE_BASE_URL, "decisions")).mock(
        return_value=Response(status_code=200, json=_decision_json(_ANSWERS["predicate"]))
    )
    create_kwargs: Dict[str, Any] = {
        "model": _REQUEST_MODEL,
        "input": _INPUT_TEXT,
        "questions": _QUESTIONS["predicate"],
    }
    _create(is_async=False, create_kwargs=create_kwargs, base_url=_AZURE_BASE_URL)

    attributes = _attributes(_openinference_span(in_memory_span_exporter))
    _assert_no_llm_attributes(attributes)
    assert attributes.pop(SpanAttributes.DECISION_PROVIDER, None) == (
        OpenInferenceLLMProviderValues.AZURE.value
    )
    assert attributes.pop(SpanAttributes.DECISION_SYSTEM, None) == (
        OpenInferenceDecisionSystemValues.OPENAI.value
    )


@pytest.mark.parametrize("is_async", [False, True])
def test_decisions_context_attributes(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "decisions")).mock(
        return_value=Response(status_code=200, json=_decision_json(_ANSWERS["mixed"]))
    )
    create_kwargs: Dict[str, Any] = {
        "model": _REQUEST_MODEL,
        "input": _INPUT_TEXT,
        "questions": _QUESTIONS["mixed"],
    }
    session_id = "my-test-session-id"
    user_id = "my-test-user-id"
    metadata = {"test-int": 1, "test-str": "string", "test-list": [1, 2, 3]}
    tags = ["tag-1", "tag-2"]
    with using_attributes(session_id=session_id, user_id=user_id, metadata=metadata, tags=tags):
        _create(is_async, create_kwargs)

    attributes = _attributes(_openinference_span(in_memory_span_exporter))
    _assert_no_llm_attributes(attributes)
    assert attributes.pop(SpanAttributes.SESSION_ID, None) == session_id
    assert attributes.pop(SpanAttributes.USER_ID, None) == user_id
    attr_metadata = attributes.pop(SpanAttributes.METADATA, None)
    assert isinstance(attr_metadata, str)
    assert json.loads(attr_metadata) == metadata
    assert list(cast(List[str], attributes.pop(SpanAttributes.TAG_TAGS, None))) == tags
    _pop_common_attributes(attributes)
    _pop_input(attributes, create_kwargs)
    _pop_output(attributes, _decision_json(_ANSWERS["mixed"]))
    _pop_response_attributes(attributes)
    assert attributes == {}


@pytest.mark.parametrize("is_async", [False, True])
def test_decisions_suppress_tracing(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "decisions")).mock(
        return_value=Response(status_code=200, json=_decision_json(_ANSWERS["predicate"]))
    )
    create_kwargs: Dict[str, Any] = {
        "model": _REQUEST_MODEL,
        "input": _INPUT_TEXT,
        "questions": _QUESTIONS["predicate"],
    }
    with suppress_tracing():
        decision = _create(is_async, create_kwargs)
    assert decision.model == _RESPONSE_MODEL
    assert not [
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if span.instrumentation_scope is not None
        and span.instrumentation_scope.name == _OPENINFERENCE_SCOPE
    ]


@pytest.mark.parametrize("hide_inputs", [False, True])
@pytest.mark.parametrize("hide_outputs", [False, True])
def test_decisions_with_config_hiding_inputs_and_outputs(
    hide_inputs: bool,
    hide_outputs: bool,
    custom_instrumentation: Callable[[TraceConfig], None],
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """input.value and output.value are the only places the request and the answers live,
    so hide_inputs / hide_outputs alone redact them completely."""
    custom_instrumentation(TraceConfig(hide_inputs=hide_inputs, hide_outputs=hide_outputs))
    response_json = _decision_json(_ANSWERS["mixed"])
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "decisions")).mock(
        return_value=Response(status_code=200, json=response_json)
    )
    create_kwargs: Dict[str, Any] = {
        "model": _REQUEST_MODEL,
        "input": _INPUT_TEXT,
        "questions": _QUESTIONS["mixed"],
    }
    _create(is_async=False, create_kwargs=create_kwargs)

    span = _openinference_span(in_memory_span_exporter)
    assert span.status.is_ok
    attributes = _attributes(span)
    _assert_no_llm_attributes(attributes)
    _pop_common_attributes(attributes)
    if hide_inputs:
        assert attributes.pop(SpanAttributes.INPUT_VALUE, None) == REDACTED_VALUE
        assert SpanAttributes.INPUT_MIME_TYPE not in attributes
    else:
        _pop_input(attributes, create_kwargs)
    if hide_outputs:
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE, None) == REDACTED_VALUE
        assert SpanAttributes.OUTPUT_MIME_TYPE not in attributes
    else:
        _pop_output(attributes, response_json)
    # Model identification and usage are not sensitive and survive both flags.
    _pop_response_attributes(attributes)
    assert attributes == {}


def test_decisions_with_config_hiding_input_images(
    custom_instrumentation: Callable[[TraceConfig], None],
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    custom_instrumentation(TraceConfig(hide_input_images=True))
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "decisions")).mock(
        return_value=Response(status_code=200, json=_decision_json(_ANSWERS["choice"]))
    )
    text_part = {"type": "input_text", "text": "Which department does this receipt go to?"}
    create_kwargs: Dict[str, Any] = {
        "model": _REQUEST_MODEL,
        "input": [
            {
                "role": "user",
                "content": [text_part, {"type": "input_image", "image_url": _IMAGE_DATA_URL}],
            }
        ],
        "questions": _QUESTIONS["choice"],
    }
    _create(is_async=False, create_kwargs=create_kwargs)

    attributes = _attributes(_openinference_span(in_memory_span_exporter))
    input_value = attributes.pop(SpanAttributes.INPUT_VALUE, None)
    assert isinstance(input_value, str)
    assert _IMAGE_DATA_URL not in input_value
    assert json.loads(input_value)["input"] == [
        {
            "role": "user",
            "content": [text_part, {"type": "input_image", "image_url": REDACTED_VALUE}],
        }
    ]


def test_decision_type_is_absent_on_older_sdks() -> None:
    """On openai < 3.26.0 there is no Decision type, and lookup must degrade to None rather
    than raise, so the instrumentor keeps working for every other endpoint."""
    assert _get_decision_type(openai) is openai.types.Decision

    def fake_sdk(**attributes: Any) -> ModuleType:
        return cast(ModuleType, SimpleNamespace(**attributes))

    assert _get_decision_type(fake_sdk(types=SimpleNamespace())) is None
    assert _get_decision_type(fake_sdk()) is None
    # A same-named attribute that is not a class is never treated as the Decision type.
    assert _get_decision_type(fake_sdk(types=SimpleNamespace(Decision="x"))) is None


def test_unrelated_endpoints_are_unaffected(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Adding Decision dispatch must not change how other response types are classified."""
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "chat/completions")).mock(
        return_value=Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "billing"},
                        "finish_reason": "stop",
                    }
                ],
                "model": "gpt-4o-mini",
            },
        )
    )
    with suppress(openai.BadRequestError):
        _client(is_async=False).chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": _INPUT_TEXT}]
        )
    spans = [
        span
        for span in in_memory_span_exporter.get_finished_spans()
        if span.instrumentation_scope is not None
        and span.instrumentation_scope.name == _OPENINFERENCE_SCOPE
    ]
    assert len(spans) == 1
    attributes = _attributes(spans[0])
    assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == (
        OpenInferenceSpanKindValues.LLM.value
    )
    assert SpanAttributes.LLM_PROVIDER in attributes
    assert SpanAttributes.LLM_SYSTEM in attributes
    assert not [key for key in attributes if key.startswith("decision.")]
