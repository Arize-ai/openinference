import json
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Mapping,
    cast,
)

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
from httpx import Response
from litellm import AuthenticationError
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
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


@pytest.fixture()
def documents() -> List[Dict[str, Any]]:
    return [
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
    ]


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
    def test_openai_chat_completions(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        lm = dspy.LM("openai/gpt-4", cache=False)
        prompt = "Who won the World Cup in 2018?"
        responses = lm(prompt)
        assert len(responses) == 1
        assert "france" in responses[0].lower()
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes)
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == prompt
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE)
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {
            "temperature": 0.0,
            "max_tokens": 1000,
        }
        assert not attributes

    def test_openai_completions(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        lm = dspy.LM("text-completion-openai/gpt-3.5-turbo-instruct", cache=False)
        prompt = "Who won the World Cup in 2018?"
        responses = lm(prompt)
        assert len(responses) == 1
        assert "france" in responses[0].lower()
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes)
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == prompt
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE)
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {
            "temperature": 0.0,
            "max_tokens": 1000,
        }
        assert not attributes

    def test_gemini(self, in_memory_span_exporter: InMemorySpanExporter) -> None:
        lm = dspy.LM("gemini/gemini-1.5-pro", cache=False)
        prompt = "Who won the World Cup in 2018?"
        responses = lm(prompt)
        assert len(responses) == 1
        assert "france" in responses[0].lower()
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes)
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == prompt
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE)
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {
            "temperature": 0.0,
            "max_tokens": 1000,
        }
        assert not attributes

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
        assert not span.status.is_ok
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
        assert event.attributes["exception.type"] == "litellm.exceptions.AuthenticationError"
        assert "401" in event.attributes["exception.message"]
        attributes = dict(span.attributes)
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == prompt
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {
            "temperature": 0.0,
            "max_tokens": 1000,
        }
        assert not attributes


@responses.activate
def test_rag_module(
    in_memory_span_exporter: InMemorySpanExporter,
    documents,
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
            "topk": documents,
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
    assert len(spans) == 5
    it = iter(spans)

    span = next(it)
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert span.name == "ColBERTv2.__call__"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.RETRIEVER.value
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    assert json.loads(input_value) == {
        "k": 3,
        "query": "What's the capital of the United States?",
    }
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    for i, doc in enumerate(documents):
        assert attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_CONTENT}", None) == doc["text"]
        assert attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_ID}", None) == doc["pid"]
        assert attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_SCORE}", None) == doc["score"]
    assert attributes == {}

    span = next(it)
    assert span.name == "Retrieve.forward"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.RETRIEVER.value
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    assert json.loads(input_value) == {
        "query_or_queries": "What's the capital of the United States?"
    }
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    for i, doc in enumerate(documents):
        assert attributes.pop(f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_CONTENT}", None) == doc["text"]
    assert attributes == {}

    span = next(it)
    assert span.name == "Predict(StringSignature).forward"

    span = next(it)
    assert span.name == "ChainOfThought.forward"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
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
    output_value_data = json.loads(output_value)
    assert (
        output_value_data
        == "Prediction(\n    rationale='Washington, D.C.',\n    answer='Washington, D.C.'\n)"
    )
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert attributes == {}

    span = next(it)
    assert span.name == "RAG.forward"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
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
    assert attributes == {}


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

    with dspy.context(lm=dspy.OpenAI(model="gpt-4", api_key="sk-fake-key")):
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
        assert not span.events, f"spans should not contain exception events {str(span.events)}"


def test_context_attributes_are_instrumented(in_memory_span_exporter: InMemorySpanExporter) -> None:
    lm = dspy.LM("openai/gpt-4", cache=False)
    prompt = "Who won the World Cup in 2018?"
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
    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        lm(prompt)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes)
    assert attributes.get(SESSION_ID) == session_id
    assert attributes.get(USER_ID) == user_id
    assert isinstance(metadata_str := attributes.get(METADATA), str)
    assert json.loads(metadata_str) == metadata
    assert list(attributes.get(TAG_TAGS, [])) == tags
    assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE) == prompt_template
    assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) == prompt_template_version
    assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) == json.dumps(
        prompt_template_variables
    )


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
