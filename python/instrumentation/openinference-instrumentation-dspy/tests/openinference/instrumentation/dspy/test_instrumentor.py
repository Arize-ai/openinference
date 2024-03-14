import json
from typing import Any, Generator
from unittest.mock import Mock, patch

import dspy
import pytest
import responses
import respx
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dspy.primitives.assertions import (
    assert_transform_module,
    backtrack_handler,
)
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from google.generativeai import GenerativeModel  # type: ignore
from google.generativeai.types import GenerateContentResponse  # type: ignore
from httpx import Response
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


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
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
    fixture clears the cache before each test case to ensure that our mocked
    responses are used.
    """
    CacheMemory.clear()
    NotebookCacheMemory.clear()


def test_openai_lm(
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
) -> None:
    class BasicQA(dspy.Signature):  # type: ignore
        """Answer questions with short factoid answers."""

        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    turbo = dspy.OpenAI(api_key="jk-fake-key", model_type="chat")
    dspy.settings.configure(lm=turbo)

    # Mock out the OpenAI API.
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
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
        )
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
    assert chain_span.name == "Predict(BasicQA).forward"
    assert lm_span.name == "GPT3.request"
    assert question in lm_span.attributes[INPUT_VALUE]  # type: ignore


def test_google_lm(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    model = dspy.Google(api_key="jk-fake-key")
    mock_response_object = Mock(spec=GenerateContentResponse)
    mock_response_object.parts = [Mock(text="Washington, D.C.")]
    mock_response_object.text = "Washington, D.C."
    with patch.object(GenerativeModel, "generate_content", return_value=mock_response_object):
        response = model("What is the capital of the United States?")
    assert response == ["Washington, D.C."]
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "Google.request"
    assert (attributes := span.attributes) is not None
    assert attributes[OPENINFERENCE_SPAN_KIND] == LLM.value
    assert isinstance((output_value := attributes[OUTPUT_VALUE]), str)
    assert json.loads(output_value) == {"text": "Washington, D.C."}
    assert attributes[OUTPUT_MIME_TYPE] == JSON.value


@responses.activate
def test_rag_module(
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
) -> None:
    class BasicQA(dspy.Signature):  # type: ignore
        """Answer questions with short factoid answers."""

        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    class RAG(dspy.Module):  # type: ignore
        """
        Performs RAG on a corpus of data.
        """

        def __init__(self, num_passages: int = 3) -> None:
            super().__init__()
            self.retrieve = dspy.Retrieve(k=num_passages)
            self.generate_answer = dspy.ChainOfThought(BasicQA)

        def forward(self, question: str) -> dspy.Prediction:
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
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
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
        )
    )

    rag = RAG()
    question = "What's the capital of the United States?"
    prediction = rag(question=question)

    assert prediction.answer == "Washington, D.C."
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 6

    span = spans[0]
    assert (attributes := span.attributes) is not None
    assert span.name == "ColBERTv2.__call__"
    assert attributes[OPENINFERENCE_SPAN_KIND] == RETRIEVER.value
    assert isinstance(input_value := attributes[INPUT_VALUE], str)
    assert json.loads(input_value) == {
        "k": 3,
        "query": "What's the capital of the United States?",
    }
    assert attributes[INPUT_MIME_TYPE] == JSON.value
    assert isinstance(
        attributes[f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_ID}"],
        int,
    )
    assert isinstance(
        attributes[f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_ID}"],
        int,
    )
    assert isinstance(
        attributes[f"{RETRIEVAL_DOCUMENTS}.2.{DOCUMENT_ID}"],
        int,
    )
    assert (
        attributes[f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_CONTENT}"] == "first retrieved document text"
    )
    assert (
        attributes[f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_CONTENT}"]
        == "second retrieved document text"
    )
    assert (
        attributes[f"{RETRIEVAL_DOCUMENTS}.2.{DOCUMENT_CONTENT}"] == "third retrieved document text"
    )
    assert isinstance(
        attributes[f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_SCORE}"],
        float,
    )
    assert isinstance(
        attributes[f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_SCORE}"],
        float,
    )
    assert isinstance(
        attributes[f"{RETRIEVAL_DOCUMENTS}.2.{DOCUMENT_SCORE}"],
        float,
    )

    span = spans[1]
    assert (attributes := span.attributes) is not None
    assert span.name == "Retrieve.forward"
    assert attributes[OPENINFERENCE_SPAN_KIND] == RETRIEVER.value
    assert isinstance(input_value := attributes[INPUT_VALUE], str) and json.loads(input_value) == {
        "query_or_queries": "What's the capital of the United States?"
    }
    assert attributes[INPUT_MIME_TYPE] == JSON.value
    assert (
        attributes[f"{RETRIEVAL_DOCUMENTS}.0.{DOCUMENT_CONTENT}"] == "first retrieved document text"
    )
    assert (
        attributes[f"{RETRIEVAL_DOCUMENTS}.1.{DOCUMENT_CONTENT}"]
        == "second retrieved document text"
    )
    assert (
        attributes[f"{RETRIEVAL_DOCUMENTS}.2.{DOCUMENT_CONTENT}"] == "third retrieved document text"
    )

    span = spans[2]
    assert (attributes := span.attributes) is not None
    assert span.name == "GPT3.request"
    assert attributes[OPENINFERENCE_SPAN_KIND] == LLM.value

    span = spans[3]
    assert (attributes := span.attributes) is not None
    assert span.name == "GPT3.request"
    assert attributes[OPENINFERENCE_SPAN_KIND] == LLM.value

    span = spans[4]
    assert (attributes := span.attributes) is not None
    assert span.name == "ChainOfThought(BasicQA).forward"
    assert attributes[OPENINFERENCE_SPAN_KIND] == CHAIN.value
    assert isinstance(input_value := attributes[INPUT_VALUE], str)
    input_value_data = json.loads(input_value)
    assert set(input_value_data.keys()) == {"context", "question"}
    assert question == input_value_data["question"]
    assert isinstance(output_value := attributes[OUTPUT_VALUE], str)
    output_value_data = json.loads(output_value)
    assert set(output_value_data.keys()) == {"answer"}
    assert output_value_data["answer"] == "Washington, D.C."

    span = spans[5]
    assert (attributes := span.attributes) is not None
    assert span.name == "RAG.forward"
    assert attributes[OPENINFERENCE_SPAN_KIND] == CHAIN.value
    assert isinstance(input_value := attributes[INPUT_VALUE], str)
    assert json.loads(input_value) == {
        "question": question,
    }
    assert attributes[INPUT_MIME_TYPE] == JSON.value
    assert isinstance(output_value := attributes[OUTPUT_VALUE], str)
    assert "Washington, D.C." in output_value
    assert attributes[OUTPUT_MIME_TYPE] == JSON.value


def test_compilation(
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
) -> None:
    class AssertModule(dspy.Module):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            self.query = dspy.Predict("question -> answer")

        def forward(self, question: str) -> dspy.Prediction:
            response = self.query(question=question)
            dspy.Assert(
                response.answer != "I don't know",
                "I don't know is not a valid answer",
            )
            return response

    student = AssertModule()
    teacher = assert_transform_module(AssertModule(), backtrack_handler)

    def exact_match(example: dspy.Example, pred: dspy.Example, trace: Any = None) -> bool:
        return bool(example.answer.lower() == pred.answer.lower())

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "chatcmpl-92UvclZCQxpucXceE70xwd5i6pX7E",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": "2",
                            "role": "assistant",
                            "function_call": None,
                            "tool_calls": None,
                        },
                    }
                ],
                "created": 1710382572,
                "model": "gpt-4-0613",
                "object": "chat.completion",
                "system_fingerprint": None,
                "usage": {"completion_tokens": 1, "prompt_tokens": 64, "total_tokens": 65},
            },
        )
    )

    with dspy.context(lm=dspy.OpenAI(model="gpt-4")):
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=exact_match,
            max_bootstrapped_demos=1,
            max_labeled_demos=1,
            num_candidate_programs=1,
            num_threads=1,
        )
        teleprompter.compile(
            student=student,
            teacher=teacher,
            trainset=[
                dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
                dspy.Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
            ],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert spans, "no spans were recorded"
    for span in spans:
        assert not span.events, "spans should not contain exception events"


DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS

CHAIN = OpenInferenceSpanKindValues.CHAIN
LLM = OpenInferenceSpanKindValues.LLM
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING

JSON = OpenInferenceMimeTypeValues.JSON
