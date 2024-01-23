from typing import Any, Dict, Generator

import dspy
import pytest
import requests_mock
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.semconv.trace import (
    SpanAttributes,
)


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
    class BasicQA(dspy.Signature):
        """Answer questions with short factoid answers."""

        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    turbo = dspy.OpenAI(model='gpt-3.5-turbo')
    dspy.settings.configure(lm=turbo)

    # Mock out the OpenAI API.
    # NB: DSPy has a built-in cache an right now that is being hit.
    # TODO(mikeldking): find documentation on how to disable it
    url = "https://api.openai.com/v1/chat/completions"
    response = {
        "json": {
                "choices": [
                    {"index": 0, "message": "American", "finish_reason": "stop"}
                ],
                "model": "gpt-35-turbo",
                "usage": {
                    "completion": 50,
                    "prompt": 25,
                    "total": 75,
                },
            }
        }


    with requests_mock.Mocker() as m:
        m.get(url, json=response)
        # Define the predictor.
        generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    question = "What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?" # noqa: E501
    pred = generate_answer(question=question) 
    assert pred.answer == "American"
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2 # 1 for the wrapping Signature, 1 for the OpenAI call
    lm_span = spans[0]
    chain_span = spans[1]
    assert chain_span.name == "BasicQA.forward"
    assert lm_span.name == "GPT3.request"
    assert question in lm_span.attributes[SpanAttributes.INPUT_VALUE]
