import json
from importlib.metadata import version
from typing import Any, Dict, Generator, List, Tuple, cast

import dspy
import pytest
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.dspy import (
    LLM_MODEL_NAME,
    LLM_PROVIDER,
    DSPyInstrumentor,
)
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

VERSION = cast(Tuple[int, int, int], tuple(map(int, version("dspy").split(".")[:3])))


def remove_all_vcr_request_headers(request: Any) -> Any:
    """
    Removes all request headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_request_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    request.headers.clear()
    return request


def remove_all_vcr_response_headers(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes all response headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_response_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    response["headers"] = {}
    return response


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
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


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="dspy")
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, DSPyInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self) -> None:
        assert isinstance(DSPyInstrumentor()._tracer, OITracer)


class TestLM:
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_openai_chat_completions_api_invoked_via_prompt_positional_argument(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        openai_api_key: str,
    ) -> None:
        lm = dspy.LM(
            "openai/gpt-4",
            cache=False,
            temperature=0.1,  # non-default
            top_p=0.1,
            api_key=openai_api_key,  # explicitly set api key as an argument to ensure it is masked
        )
        prompt = "Who won the World Cup in 2018?"
        responses = lm(
            prompt,
            temperature=0.2,  # overrides temperature setting in init
        )  # invoked via positional prompt argument
        assert len(responses) == 1
        response = responses[0]
        assert "france" in response.lower()
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "LM.__call__"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        input_data = json.loads(input_value)
        assert input_data == {
            "prompt": prompt,
            "messages": None,
            "kwargs": {"temperature": 0.2},
        }
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert isinstance(output_data := json.loads(output_value), list)
        assert len(output_data) == 1
        assert output_data[0] == response
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {
            "max_tokens": 4000,  # default setting in LM
            "temperature": 0.2,  # from __call__
            "top_p": 0.1,  # from __init__
        }
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == prompt
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == response
        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
        assert not attributes

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_openai_chat_completions_api_invoked_via_messages_kwarg(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        openai_api_key: str,
    ) -> None:
        lm = dspy.LM("openai/gpt-4", cache=False)
        prompt = "Who won the World Cup in 2018?"
        messages = [{"role": "user", "content": prompt}]
        responses = lm(messages=messages)  # invoked via messages kwarg
        assert len(responses) == 1
        response = responses[0]
        assert "france" in response.lower()
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "LM.__call__"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        input_data = json.loads(input_value)
        assert input_data == {
            "prompt": None,
            "messages": messages,
            "kwargs": {},
        }
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert isinstance(output_data := json.loads(output_value), list)
        assert len(output_data) == 1
        assert output_data[0] == response
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {
            "temperature": 0.0,
            "max_tokens": 4000,
        }
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == prompt
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == response
        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
        assert not attributes

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_openai_completions_api_invoked_via_prompt_positional_argument(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        openai_api_key: str,
    ) -> None:
        lm = dspy.LM(
            "text-completion-openai/gpt-3.5-turbo-instruct",
            model_type="text",
            cache=False,
        )
        prompt = "Who won the World Cup in 2018?"
        responses = lm(prompt)  # invoked via messages kwarg
        assert len(responses) == 1
        response = responses[0]
        assert "france" in response.lower()
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "LM.__call__"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        input_data = json.loads(input_value)
        assert input_data == {
            "prompt": prompt,
            "messages": None,
            "kwargs": {},
        }
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert isinstance(output_data := json.loads(output_value), list)
        assert len(output_data) == 1
        assert output_data[0] == response
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {
            "temperature": 0.0,
            "max_tokens": 4000,
        }
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == prompt
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == response
        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-3.5-turbo-instruct"
        assert not attributes

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_exception_event_recorded_on_lm_error(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        openai_api_key: str,
    ) -> None:
        lm = dspy.LM("openai/gpt-4", cache=False)
        prompt = "Who won the World Cup in 2018?"
        with pytest.raises(Exception):
            lm(prompt)
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "LM.__call__"
        assert not span.status.is_ok
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
        assert (event_attributes := event.attributes) is not None
        assert isinstance(exception_type := event_attributes["exception.type"], str)
        assert exception_type.startswith("litellm.exceptions")
        assert isinstance(exception_message := event_attributes["exception.message"], str)
        assert "Connection error" in exception_message
        assert isinstance(exception_stacktrace := event_attributes["exception.stacktrace"], str)
        assert "Incorrect API key provided" in exception_stacktrace
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        input_data = json.loads(input_value)
        assert input_data == {
            "prompt": prompt,
            "messages": None,
            "kwargs": {},
        }
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == prompt
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {
            "temperature": 0.0,
            "max_tokens": 4000,
        }
        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
        assert not attributes

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_subclass(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        openai_api_key: str,
    ) -> None:
        class MyLM(dspy.LM):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__("openai/gpt-4", cache=False)

            def __call__(
                self,
                question: str,
            ) -> List[str]:  # signature is different from superclass
                return cast(List[str], super().__call__(question))

        lm = MyLM()
        prompt = "Who won the World Cup in 2018?"
        responses = lm(prompt)
        assert len(responses) == 1
        response = responses[0]
        assert "france" in response.lower()
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "MyLM.__call__"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        input_data = json.loads(input_value)
        assert input_data == {
            "prompt": prompt,
            "messages": None,
            "kwargs": {},
        }
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert isinstance(output_data := json.loads(output_value), list)
        assert len(output_data) == 1
        assert output_data[0] == response
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {
            "temperature": 0.0,
            "max_tokens": 4000,
        }
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == prompt
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == response
        assert attributes.pop(LLM_PROVIDER) == "openai"
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
        assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
@pytest.mark.parametrize("is_async", [False, True])
async def test_rag_module(
    in_memory_span_exporter: InMemorySpanExporter,
    is_async: bool,
    openai_api_key: str,
) -> None:
    K = 3

    class BasicQA(dspy.Signature):  # type: ignore
        """Answer questions with short factoid answers."""

        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    class RAG(dspy.Module):  # type: ignore
        """
        Performs RAG on a corpus of data.
        """

        def __init__(self) -> None:
            super().__init__()
            self.retrieve = dspy.Retrieve(k=K)
            self.generate_answer = dspy.ChainOfThought(BasicQA)

        def forward(self, question: str) -> dspy.Prediction:
            context = self.retrieve(question).passages
            prediction = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=prediction.answer)

        async def aforward(self, question: str) -> dspy.Prediction:
            context = self.retrieve(question).passages
            prediction = await self.generate_answer.acall(context=context, question=question)

            return dspy.Prediction(context=context, answer=prediction.answer)

    dspy.settings.configure(
        lm=dspy.LM("openai/gpt-4", cache=False),
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
    )

    rag = RAG()
    question = "What's the capital of the United States?"

    if is_async:
        prediction = await rag.acall(question=question)
    else:
        prediction = rag(question=question)

    assert prediction.answer == "Washington, D.C."

    spans = list(in_memory_span_exporter.get_finished_spans())
    spans.sort(key=lambda span: span.start_time or 0)

    assert len(spans) == 8

    it = iter(spans)

    span = next(it)
    expected_span_name = "RAG.aforward" if is_async else "RAG.forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert json.loads(input_value) == {
        "question": question,
    }
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert "Washington, D.C." in output_value
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert not attributes

    span = next(it)
    assert span.name == "Retrieve.forward"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.RETRIEVER.value
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    assert json.loads(input_value) == {"query": "What's the capital of the United States?"}
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    for i in range(K):
        assert isinstance(attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_CONTENT}"), str)
    assert not attributes

    span = next(it)
    attributes = dict(span.attributes or {})
    assert span.name == "ColBERTv2.__call__"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.RETRIEVER.value
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    assert json.loads(input_value) == {
        "k": K,
        "query": "What's the capital of the United States?",
    }
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    for i in range(K):
        assert isinstance(attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_CONTENT}"), str)
        assert isinstance(attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_ID}"), int)
        assert isinstance(attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_SCORE}"), float)
    assert not attributes

    span = next(it)
    expected_span_name = "ChainOfThought.aforward" if is_async else "ChainOfThought.forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert json.loads(input_value)["question"] == question
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert "Washington, D.C." in output_value
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert not attributes

    span = next(it)
    expected_span_name = "Predict.aforward" if is_async else "Predict.forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    input_value_data = json.loads(input_value)
    assert set(input_value_data.keys()) == {"context", "question"}
    assert question == input_value_data["question"]
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    expected_output_value = (
        '{"reasoning": "The capital of a country is a well-established fact.'
        " The capital of the United States is a widely known piece of "
        'information.", "answer": "Washington, D.C."}'
    )

    assert output_value == expected_output_value
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert not attributes

    span = next(it)
    expected_span_name = "Predict(StringSignature).forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes

    span = next(it)
    expected_span_name = "ChatAdapter.acall" if is_async else "ChatAdapter.__call__"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes

    span = next(it)
    expected_span_name = "LM.acall" if is_async else "LM.__call__"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    input_data = json.loads(input_value)
    assert set(input_data.keys()) == {"prompt", "messages", "kwargs"}
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
    assert isinstance(output_data := json.loads(output_value), list)
    assert len(output_data) == 1
    assert isinstance(output_data[0], str)
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == {
        "temperature": 0.0,
        "max_tokens": 4000,
    }
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
    assert isinstance(attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        message_content_1 := attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}"),
        str,
    )
    assert question in message_content_1
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(
        message_content_0 := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"),
        str,
    )
    assert "Washington, D.C." in message_content_0
    assert attributes.pop(LLM_PROVIDER) == "openai"
    assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
@pytest.mark.parametrize("is_async", [False, True])
async def test_react(
    in_memory_span_exporter: InMemorySpanExporter,
    is_async: bool,
    openai_api_key: str,
) -> None:
    dspy.settings.configure(
        lm=dspy.LM("openai/gpt-4o-mini"),
    )

    def add(x: int, y: int) -> int:
        return x + y

    react = dspy.ReAct("question -> answer", tools=[add])
    question = "What is 2 + 2?"

    if is_async:
        response = await react.acall(question=question)
    else:
        response = react(question=question)

    assert response.answer == "4"

    spans = list(in_memory_span_exporter.get_finished_spans())
    spans.sort(key=lambda span: span.start_time or 0)

    assert len(spans) == 16

    it = iter(spans)

    span = next(it)
    expected_span_name = "ReAct.aforward" if is_async else "ReAct.forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert json.loads(input_value)["input_args"] == {
        "question": question,
    }
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert "4" in output_value
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert not attributes

    span = next(it)
    expected_span_name = "Predict.aforward" if is_async else "Predict.forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert json.loads(input_value) == {"question": question, "trajectory": ""}
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    expected_output = (
        '{"next_thought": "I need to perform the addition of 2 and 2 to answer '
        'the question.", "next_tool_name": "add", "next_tool_args": {"x": 2, "y": 2}}'
    )
    assert output_value == expected_output
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert not attributes

    span = next(it)
    expected_span_name = "Predict(StringSignature).forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert json.loads(input_value) == {"question": question, "trajectory": ""}
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    output_value = json.loads(output_value)
    assert isinstance(output_value, dict)
    assert output_value == {
        "next_thought": "I need to perform the addition of 2 and 2 to answer the question.",
        "next_tool_name": "add",
        "next_tool_args": {
            "x": 2,
            "y": 2,
        },
    }
    assert not attributes

    span = next(it)
    expected_span_name = "ChatAdapter.acall" if is_async else "ChatAdapter.__call__"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert (
        "Given the fields `question`, produce the fields `answer`."
        in json.loads(input_value)["signature"]
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    output_value = json.loads(output_value)
    assert isinstance(output_value, list)
    assert len(output_value) == 1
    # Handle both old format (next_tool_name/next_tool_args) and new format
    if "next_tool_name" in output_value[0]:
        # In newer DSPy versions, this might be "add" instead of "finish"
        assert (
            output_value[0].get("next_tool_name") in ["finish", "add"]
            and "next_tool_args" in output_value[0]
        )
    else:
        assert "answer" in output_value[0] or "reasoning" in output_value[0]
    assert not attributes

    span = next(it)
    expected_span_name = "LM.acall" if is_async else "LM.__call__"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert "Given the fields `question`, produce the fields `answer`." in input_value
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    output_value = json.loads(output_value)
    assert isinstance(output_value, list)
    assert len(output_value) == 1
    assert "I need to perform the addition of 2 and 2" in output_value[-1]

    span = next(it)
    expected_span_name = "add.acall" if is_async else "add.__call__"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    input_value = json.loads(input_value)
    assert isinstance(input_value, dict)
    assert input_value == {"kwargs": {"x": 2, "y": 2}}
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert output_value == "4"
    assert not attributes

    span = next(it)
    expected_span_name = "Predict.aforward" if is_async else "Predict.forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    input_value = json.loads(input_value)
    assert isinstance(input_value, dict)
    assert input_value.pop("question") == "What is 2 + 2?"
    assert input_value.pop("trajectory", "").endswith("[[ ## observation_0 ## ]]\n4")
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    expected_output = (
        '{"next_thought": "I have completed the addition and found that 2 + 2 equals 4. '
        "I can now finish the task as I have all the information needed to answer "
        'the question.", "next_tool_name": "finish", "next_tool_args": {}}'
    )
    assert output_value == expected_output
    assert not attributes

    span = next(it)
    expected_span_name = "Predict(StringSignature).forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    input_value = json.loads(input_value)
    assert isinstance(input_value, dict)
    assert input_value.pop("question") == "What is 2 + 2?"
    assert input_value.pop("trajectory", "").endswith("[[ ## observation_0 ## ]]\n4")
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    output_value = json.loads(output_value)
    assert isinstance(output_value, dict)
    assert output_value.pop("next_tool_name") == "finish"
    assert output_value.pop("next_tool_args") == {}
    assert not attributes

    span = next(it)
    expected_span_name = "ChatAdapter.acall" if is_async else "ChatAdapter.__call__"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert (
        "Given the fields `question`, produce the fields `answer`."
        in json.loads(input_value)["signature"]
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    output_value = json.loads(output_value)
    assert isinstance(output_value, list)
    assert len(output_value) == 1
    # Handle both old format (next_tool_name/next_tool_args) and new format (answer/reasoning)
    if "next_tool_name" in output_value[0]:
        assert (
            output_value[0].get("next_tool_name") == "finish"
            and "next_tool_args" in output_value[0]
        )
    else:
        assert "answer" in output_value[0] and output_value[0]["answer"] == "4"
    assert not attributes

    span = next(it)
    expected_span_name = "LM.acall" if is_async else "LM.__call__"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert "Given the fields `question`, produce the fields `answer`." in input_value
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    output_value = json.loads(output_value)
    assert isinstance(output_value, list)
    assert len(output_value) == 1
    assert (
        "next_thought" in output_value[-1]
        and "next_tool_name" in output_value[-1]
        and "finish" in output_value[-1]
        and "completed" in output_value[-1]
    )

    span = next(it)
    expected_span_name = "finish.acall" if is_async else "finish.__call__"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    input_value = json.loads(input_value)
    assert isinstance(input_value, dict)
    assert input_value == {"kwargs": {}}
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert output_value == '"Completed."'
    assert not attributes

    span = next(it)
    expected_span_name = "ChainOfThought.aforward" if is_async else "ChainOfThought.forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert json.loads(input_value)["question"] == question
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert '"answer": "4"' in output_value
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert not attributes

    span = next(it)
    expected_span_name = "Predict.aforward" if is_async else "Predict.forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert json.loads(input_value)["question"] == question
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert '"answer": "4"' in output_value
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert not attributes

    span = next(it)
    expected_span_name = "Predict(StringSignature).forward"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    input_value = json.loads(input_value)
    assert isinstance(input_value, dict)
    assert input_value.pop("question") == "What is 2 + 2?"
    assert input_value.pop("trajectory").endswith("[[ ## observation_1 ## ]]\nCompleted.")
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    output_value = json.loads(output_value)
    assert isinstance(output_value, dict)
    assert output_value.pop("answer") == "4"
    assert not attributes

    span = next(it)
    expected_span_name = "ChatAdapter.acall" if is_async else "ChatAdapter.__call__"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert (
        "Given the fields `question`, produce the fields `answer`."
        in json.loads(input_value)["signature"]
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    output_value = json.loads(output_value)
    assert isinstance(output_value, list)
    assert len(output_value) == 1
    # Handle both old format (next_tool_name/next_tool_args) and new format (answer/reasoning)
    if "next_tool_name" in output_value[0]:
        assert (
            output_value[0].get("next_tool_name") == "finish"
            and "next_tool_args" in output_value[0]
        )
    else:
        assert "answer" in output_value[0] and output_value[0]["answer"] == "4"
    assert not attributes

    span = next(it)
    expected_span_name = "LM.acall" if is_async else "LM.__call__"
    assert span.name == expected_span_name
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
    input_value = attributes.pop(INPUT_VALUE)
    assert isinstance(input_value, str)
    assert "Given the fields `question`, produce the fields `answer`." in input_value
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    output_value = json.loads(output_value)
    assert isinstance(output_value, list)
    assert len(output_value) == 1
    assert "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]" in output_value[-1]


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
@pytest.mark.skipif(VERSION >= (2, 6, 22), reason="requires dspy < 2.6.22")
def test_compilation(
    in_memory_span_exporter: InMemorySpanExporter,
    openai_api_key: str,
) -> None:
    from dspy.primitives.assertions import (
        assert_transform_module,
        backtrack_handler,
    )

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

    with dspy.context(lm=dspy.LM("openai/gpt-4", cache=False)):
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
        assert not span.events


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_context_attributes_are_instrumented(
    in_memory_span_exporter: InMemorySpanExporter,
    openai_api_key: str,
) -> None:
    session_id = "my-test-session-id"
    user_id = "my-test-user-id"
    metadata = {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }
    tags = ["tag-1", "tag-2"]
    prompt_template = (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )
    prompt_template_version = "v1.0"
    prompt_template_variables = {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }

    K = 3

    class BasicQA(dspy.Signature):  # type: ignore
        """Answer questions with short factoid answers."""

        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    class RAG(dspy.Module):  # type: ignore
        """
        Performs RAG on a corpus of data.
        """

        def __init__(self) -> None:
            super().__init__()
            self.retrieve = dspy.Retrieve(k=K)
            self.generate_answer = dspy.ChainOfThought(BasicQA)

        def forward(self, question: str) -> dspy.Prediction:
            context = self.retrieve(question).passages
            prediction = self.generate_answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=prediction.answer)

    dspy.settings.configure(
        lm=dspy.LM("openai/gpt-4", cache=False),
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
    )
    rag = RAG()
    question = "What's the capital of the United States?"
    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        prediction = rag(question=question)

    assert prediction.answer == "Washington, D.C."
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 8
    for span in spans:
        attributes = dict(span.attributes or {})
        assert attributes.get(SESSION_ID) == session_id
        assert attributes.get(USER_ID) == user_id
        assert isinstance(metadata_str := attributes.get(METADATA), str)
        assert json.loads(metadata_str) == metadata
        assert attributes.get(TAG_TAGS) == tuple(tags)
        assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE) == prompt_template
        assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) == prompt_template_version
        assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) == json.dumps(
            prompt_template_variables
        )


CHAIN = OpenInferenceSpanKindValues.CHAIN.value
LLM = OpenInferenceSpanKindValues.LLM.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
METADATA = SpanAttributes.METADATA
TAG_TAGS = SpanAttributes.TAG_TAGS
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
