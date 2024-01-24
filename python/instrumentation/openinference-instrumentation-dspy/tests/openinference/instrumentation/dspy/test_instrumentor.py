from typing import Generator

import dspy
import pytest
import requests_mock
from openinference.instrumentation.dspy import DSPyInstrumentor
from openinference.semconv.trace import (
    SpanAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    DSPyInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    DSPyInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


def test_openai_lm(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    class BasicQA(dspy.Signature):  # type: ignore
        """Answer questions with short factoid answers."""

        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    turbo = dspy.OpenAI(api_key="jk-fake-key", model_type="chat")
    dspy.settings.configure(lm=turbo)

    # Mock out the OpenAI API.
    url = "https://api.openai.com/v1/chat/completions"
    response = {
        "id": "chatcmpl-8kKarJQUyeuFeRsj18o6TWrxoP2zs",
        "object": "chat.completion",
        "created": 1706052941,
        "model": "gpt-3.5-turbo-0613",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Washington DC",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 39, "completion_tokens": 396, "total_tokens": 435},
        "system_fingerprint": None,
    }

    with requests_mock.Mocker() as m:
        m.post(url, json=response)
        # Define the predictor.
        generate_answer = dspy.Predict(BasicQA)

        # Call the predictor on a particular input.
        question = "What's the capital of the United States?"  # noqa: E501
        pred = generate_answer(question=question)

    assert pred.answer == "Washington DC"
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2  # 1 for the wrapping Signature, 1 for the OpenAI call
    lm_span = spans[0]
    chain_span = spans[1]
    assert chain_span.name == "BasicQA.forward"
    assert lm_span.name == "GPT3.request"
    assert question in lm_span.attributes[SpanAttributes.INPUT_VALUE]  # type: ignore
