import json
from typing import (
    Any,
    Dict,
    Generator,
    List,
    cast,
)

import dspy
import pytest
from dsp.modules.cache_utils import CacheMemory, NotebookCacheMemory
from dspy.primitives.assertions import (
    assert_transform_module,
    backtrack_handler,
)
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from litellm import AuthenticationError  # type: ignore[attr-defined]
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pytest import MonkeyPatch

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.dspy import DSPyInstrumentor
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)


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


@pytest.fixture
def openai_api_key(monkeypatch: MonkeyPatch) -> str:
    api_key = "sk-fake-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    return api_key


@pytest.fixture
def gemini_api_key(monkeypatch: MonkeyPatch) -> str:
    api_key = "sk-fake-key"
    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    return api_key


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """
    DSPy caches responses from retrieval and language models to disk. This
    fixture clears the cache before each test case to ensure that our mocked
    responses are used.
    """
    try:
        CacheMemory.clear()
        NotebookCacheMemory.clear()
    except Exception:
        pass


def test_oitracer() -> None:
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
            "max_tokens": 1000,  # default setting in LM
            "temperature": 0.2,  # from __call__
            "top_p": 0.1,  # from __init__
        }
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == prompt
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == response
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
            "max_tokens": 1000,
        }
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == prompt
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == response
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
            "text-completion-openai/gpt-3.5-turbo-instruct", model_type="text", cache=False
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
            "max_tokens": 1000,
        }
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == prompt
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == response
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
        with pytest.raises(AuthenticationError):
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
        assert event_attributes["exception.type"] == "litellm.exceptions.AuthenticationError"
        assert isinstance(exception_message := event_attributes["exception.message"], str)
        assert "401" in exception_message
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
            "max_tokens": 1000,
        }
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
            "max_tokens": 1000,
        }
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == prompt
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == response
        assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_rag_module(
    in_memory_span_exporter: InMemorySpanExporter,
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

    dspy.settings.configure(
        lm=dspy.LM("openai/gpt-4", cache=False),
        rm=dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts"),
    )

    rag = RAG()
    question = "What's the capital of the United States?"
    prediction = rag(question=question)
    assert prediction.answer == "Washington, D.C."
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 7
    it = iter(spans)

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
    assert span.name == "Retrieve.forward"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.RETRIEVER.value
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    assert json.loads(input_value) == {
        "query_or_queries": "What's the capital of the United States?"
    }
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    for i in range(K):
        assert isinstance(attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_CONTENT}"), str)
    assert not attributes

    span = next(it)
    assert span.name == "LM.__call__"
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
        "max_tokens": 1000,
    }
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
    assert isinstance(attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        message_content_1 := attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}"), str
    )
    assert question in message_content_1
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(
        message_content_0 := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
    )
    assert "Washington, D.C." in message_content_0
    assert not attributes

    span = next(it)
    assert span.name == "ChatAdapter.__call__"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes

    span = next(it)
    assert span.name == "Predict(StringSignature).forward"
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert not attributes

    span = next(it)
    assert span.name == "ChainOfThought.forward"
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
    assert "Prediction" in output_value
    assert "reasoning=" in output_value
    assert "answer=" in output_value
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert not attributes

    span = next(it)
    assert span.name == "RAG.forward"
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


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_compilation(
    in_memory_span_exporter: InMemorySpanExporter,
    openai_api_key: str,
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
    assert len(spans) == 7
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
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
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
