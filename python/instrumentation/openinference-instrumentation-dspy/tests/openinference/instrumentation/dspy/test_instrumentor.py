import json
from typing import Generator

import dspy
import pytest
import responses
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from openinference.instrumentation.dspy import DSPyInstrumentor
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
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


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """
    DSPy caches responses from retrieval and language models to disk. This
    fixture clears the caches before each test case to ensure that our mocked
    responses are used.
    """
    CacheMemory.clear()
    NotebookCacheMemory.clear()


@responses.activate
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
    responses.add(
        method=responses.POST,
        url="https://api.openai.com/v1/chat/completions",
        json={
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
        },
        status=200,
    )

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
    assert chain_span.name == "Predict.BasicQA.forward"
    assert lm_span.name == "GPT3.request"
    assert question in lm_span.attributes[SpanAttributes.INPUT_VALUE]  # type: ignore


@responses.activate
def test_rag_module(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    class BasicQA(dspy.Signature):  # type: ignore
        """Answer questions with short factoid answers."""

        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    class RAG(dspy.Module):
        """
        Performs RAG on a corpus of data.
        """

        def __init__(self, num_passages=3):
            super().__init__()
            self.retrieve = dspy.Retrieve(k=num_passages)
            self.generate_answer = dspy.ChainOfThought(BasicQA)

        def forward(self, question):
            context = self.retrieve(question).passages
            prediction = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=prediction.answer)

    turbo = dspy.OpenAI(api_key="jk-fake-key", model_type="chat")
    colbertv2_url = "https://www.examplecolbertv2service.com/wiki17_abstracts"
    colbertv2 = dspy.ColBERTv2(url=colbertv2_url)
    dspy.settings.configure(lm=turbo, rm=colbertv2)

    # Mock the request to the remote ColBERTv2 service.
    responses.add(
        method=responses.GET,
        url=colbertv2_url,
        json={
            "topk": [
                {
                    "text": "first retrieved document text",
                    "pid": 1918771,
                    "rank": 1,
                    "score": 26.81817626953125,
                    "prob": 0.7290767171685155,
                    "long_text": "first retrieved document long text",
                },
                {
                    "text": "second retrieved document text",
                    "pid": 3377468,
                    "rank": 2,
                    "score": 25.304840087890625,
                    "prob": 0.16052389034616518,
                    "long_text": "second retrieved document long text",
                },
                {
                    "text": "third retrieved document text",
                    "pid": 953799,
                    "rank": 3,
                    "score": 24.93050193786621,
                    "prob": 0.11039939248531924,
                    "long_text": "third retrieved document long text",
                },
            ],
            "latency": 84.43140983581543,
        },
        status=200,
    )

    # Mock out the OpenAI API.
    responses.add(
        method=responses.POST,
        url="https://api.openai.com/v1/chat/completions",
        json={
            "id": "chatcmpl-8kKarJQUyeuFeRsj18o6TWrxoP2zs",
            "object": "chat.completion",
            "created": 1706052941,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Washington, D.C.",
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 39, "completion_tokens": 396, "total_tokens": 435},
            "system_fingerprint": None,
        },
        status=200,
    )

    rag = RAG()
    question = "What's the capital of the United States?"
    prediction = rag(question=question)

    assert prediction.answer == "Washington, D.C."
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 5

    span = spans[0]
    assert span.name == "Retrieve.forward"
    assert (
        span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.RETRIEVER.value
    )
    assert json.loads(span.attributes[SpanAttributes.INPUT_VALUE]) == {
        "query_or_queries": "What's the capital of the United States?"
    }
    assert span.attributes[SpanAttributes.INPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(span.attributes[SpanAttributes.OUTPUT_VALUE]) == {
        "passages": [
            "first retrieved document text",
            "second retrieved document text",
            "third retrieved document text",
        ]
    }
    assert (
        span.attributes[SpanAttributes.OUTPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
    )
    assert isinstance(
        span.attributes[f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_ID}"],
        str,
    )
    assert isinstance(
        span.attributes[f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.1.{DocumentAttributes.DOCUMENT_ID}"],
        str,
    )
    assert isinstance(
        span.attributes[f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.2.{DocumentAttributes.DOCUMENT_ID}"],
        str,
    )
    assert (
        span.attributes[
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_CONTENT}"
        ]
        == "first retrieved document text"
    )
    assert (
        span.attributes[
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.1.{DocumentAttributes.DOCUMENT_CONTENT}"
        ]
        == "second retrieved document text"
    )
    assert (
        span.attributes[
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.2.{DocumentAttributes.DOCUMENT_CONTENT}"
        ]
        == "third retrieved document text"
    )

    span = spans[1]
    assert span.name == "GPT3.request"
    assert (
        span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.LLM.value
    )

    span = spans[2]
    assert span.name == "GPT3.request"
    assert (
        span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.LLM.value
    )

    span = spans[3]
    assert span.name == "ChainOfThought.BasicQA.forward"
    assert (
        span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.CHAIN.value
    )
    input_value = json.loads(span.attributes[SpanAttributes.INPUT_VALUE])
    assert set(input_value.keys()) == {"signature", "context", "question"}
    signature = input_value["signature"]
    assert set(signature.keys()) == {"fields", "instructions"}
    fields = signature["fields"]
    assert all(
        lambda field: {"name", "type", "description"}.issubset(set(field.keys()))
        for field in fields
    )
    assert question == input_value["question"]
    output_value = json.loads(span.attributes[SpanAttributes.OUTPUT_VALUE])
    assert set(output_value.keys()) == {"answer", "rationale"}
    assert output_value["answer"] == "Washington, D.C."
    assert isinstance(output_value["rationale"], str)

    span = spans[4]
    assert span.name == "RAG.forward"
    assert (
        span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.CHAIN.value
    )
    assert json.loads(span.attributes[SpanAttributes.INPUT_VALUE]) == {
        "question": question,
    }
    assert span.attributes[SpanAttributes.INPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
    assert "Washington, D.C." in span.attributes[SpanAttributes.OUTPUT_VALUE]
    assert (
        span.attributes[SpanAttributes.OUTPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
    )
