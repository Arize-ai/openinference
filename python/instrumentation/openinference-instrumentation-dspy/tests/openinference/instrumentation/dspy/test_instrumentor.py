from typing import Generator

import dspy
import pytest
import responses
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
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
    assert chain_span.name == "BasicQA.forward"
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
            # self.generate_answer = BasicQA
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
                    "text": "United States capital (disambiguation) | The capital of the United States is Washington, D.C.",  # noqa: E501
                    "pid": 1918771,
                    "rank": 1,
                    "score": 26.81817626953125,
                    "prob": 0.7290767171685155,
                    "long_text": "United States capital (disambiguation) | The capital of the United States is Washington, D.C.",  # noqa: E501
                },
                {
                    "text": "List of capitals in the United States | Washington, D.C. is the current federal capital city of the United States, as it has been since 1800. Each U.S. state has its own capital city, as do many of its Insular areas. Historically, most states have not changed their capital city since becoming a state, but the capital cities of their respective preceding colonies, territories, kingdoms, and republics typically changed multiple times. There have also been other governments within the current borders of the United States with their own capitals, such as the Republic of Texas, Native American nations, and other unrecognized governments.Siva",  # noqa: E501
                    "pid": 3377468,
                    "rank": 2,
                    "score": 25.304840087890625,
                    "prob": 0.16052389034616518,
                    "long_text": "List of capitals in the United States | Washington, D.C. is the current federal capital city of the United States, as it has been since 1800. Each U.S. state has its own capital city, as do many of its Insular areas. Historically, most states have not changed their capital city since becoming a state, but the capital cities of their respective preceding colonies, territories, kingdoms, and republics typically changed multiple times. There have also been other governments within the current borders of the United States with their own capitals, such as the Republic of Texas, Native American nations, and other unrecognized governments.Siva",  # noqa: E501
                },
                {
                    "text": 'Washington, D.C. | Washington, D.C., formally the District of Columbia and commonly referred to as "Washington", "the District", or simply "D.C.", is the capital of the United States.',  # noqa: E501
                    "pid": 953799,
                    "rank": 3,
                    "score": 24.93050193786621,
                    "prob": 0.11039939248531924,
                    "long_text": 'Washington, D.C. | Washington, D.C., formally the District of Columbia and commonly referred to as "Washington", "the District", or simply "D.C.", is the capital of the United States.',  # noqa: E501
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
    assert len(spans) == 4

    span = spans[0]
    assert span.name == "GPT3.request"

    span = spans[1]
    assert span.name == "GPT3.request"

    span = spans[2]
    assert span.name == "Template.forward"

    span = spans[3]
    assert span.name == "RAG.forward"
