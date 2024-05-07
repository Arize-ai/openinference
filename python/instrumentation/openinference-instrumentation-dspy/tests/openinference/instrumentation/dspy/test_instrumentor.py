import json
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Mapping,
    cast,
)
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
from openinference.instrumentation import using_attributes
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
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue


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
def session_id() -> str:
    return "my-test-session-id"


@pytest.fixture()
def user_id() -> str:
    return "my-test-user-id"


@pytest.fixture()
def metadata() -> Dict[str, Any]:
    return {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }


@pytest.fixture()
def tags() -> List[str]:
    return ["tag-1", "tag-2"]


@pytest.fixture
def prompt_template() -> str:
    return (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )


@pytest.fixture
def prompt_template_version() -> str:
    return "v1.0"


@pytest.fixture
def prompt_template_variables() -> Dict[str, Any]:
    return {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }


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


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_openai_lm(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            pred = generate_answer(question=question)
    else:
        pred = generate_answer(question=question)

    assert pred.answer == "Washington DC"
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2  # 1 for the wrapping Signature, 1 for the OpenAI call
    lm_span, chain_span = spans
    # Verify lm_span
    assert lm_span.name == "GPT3.request"
    lm_attributes = dict(cast(Mapping[str, AttributeValue], lm_span.attributes))
    assert lm_attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert lm_attributes.pop(LLM_MODEL_NAME) == "gpt-3.5-turbo-instruct"
    input_value = lm_attributes.pop(INPUT_VALUE)
    assert question in input_value  # type:ignore
    assert (
        OpenInferenceMimeTypeValues(lm_attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.TEXT
    )
    assert isinstance(
        invocation_parameters_str := lm_attributes.pop(LLM_INVOCATION_PARAMETERS), str
    )
    assert json.loads(invocation_parameters_str) == {
        "temperature": 0.0,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "model": "gpt-3.5-turbo-instruct",
    }
    assert isinstance(lm_attributes.pop(OUTPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(lm_attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    if use_context_attributes:
        _check_context_attributes(
            lm_attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert lm_attributes == {}
    # Verify chain_span
    assert chain_span.name == "Predict(BasicQA).forward"
    chain_attributes = dict(cast(Mapping[str, AttributeValue], chain_span.attributes))
    assert chain_attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    input_value = chain_attributes.pop(INPUT_VALUE)
    assert question in input_value  # type:ignore
    assert (
        OpenInferenceMimeTypeValues(chain_attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert isinstance(chain_attributes.pop(OUTPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(chain_attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    if use_context_attributes:
        _check_context_attributes(
            chain_attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert chain_attributes == {}


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_google_lm(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    model = dspy.Google(api_key="jk-fake-key")
    mock_response_object = Mock(spec=GenerateContentResponse)
    mock_response_object.parts = [Mock(text="Washington, D.C.")]
    mock_response_object.text = "Washington, D.C."
    question = "What is the capital of the United States?"
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            with patch.object(
                GenerativeModel, "generate_content", return_value=mock_response_object
            ):
                response = model(question)
    else:
        with patch.object(GenerativeModel, "generate_content", return_value=mock_response_object):
            response = model(question)
    assert response == ["Washington, D.C."]
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "Google.request"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    input_value = attributes.pop(INPUT_VALUE)
    assert question in input_value  # type:ignore
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.TEXT
    )
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert json.loads(output_value) == {"text": "Washington, D.C."}
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "n": 1,
        "candidate_count": 1,
        "temperature": 0.0,
        "max_output_tokens": 2048,
        "top_p": 1,
        "top_k": 1,
    }
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}


@responses.activate
@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_rag_module(
    use_context_attributes: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
    documents: List[Dict[str, Any]],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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
    if use_context_attributes:
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
    else:
        prediction = rag(question=question)

    assert prediction.answer == "Washington, D.C."
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 6

    span = spans[0]
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
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}

    span = spans[1]
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
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}

    span = spans[2]
    assert span.name == "GPT3.request"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop(LLM_MODEL_NAME) == "gpt-3.5-turbo-instruct"
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "temperature": 0.0,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "model": "gpt-3.5-turbo-instruct",
    }
    input_value = attributes.pop(INPUT_VALUE)
    assert question in input_value  # type:ignore
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.TEXT
    )
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}

    span = spans[3]
    assert span.name == "GPT3.request"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop(LLM_MODEL_NAME) == "gpt-3.5-turbo-instruct"
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "temperature": 0.0,
        "max_tokens": 75,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "model": "gpt-3.5-turbo-instruct",
    }
    input_value = attributes.pop(INPUT_VALUE)
    assert question in input_value  # type:ignore
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.TEXT
    )
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}

    span = spans[4]
    assert span.name == "ChainOfThought(BasicQA).forward"
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
    assert set(output_value_data.keys()) == {"answer"}
    assert output_value_data["answer"] == "Washington, D.C."
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}

    span = spans[5]
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
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
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


def _check_context_attributes(
    attributes: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    assert attributes.pop(SESSION_ID, None) == session_id
    assert attributes.pop(USER_ID, None) == user_id
    attr_metadata = attributes.pop(METADATA, None)
    assert attr_metadata is not None
    assert isinstance(attr_metadata, str)  # must be json string
    metadata_dict = json.loads(attr_metadata)
    assert metadata_dict == metadata
    attr_tags = attributes.pop(TAG_TAGS, None)
    assert attr_tags is not None
    assert len(attr_tags) == len(tags)
    assert list(attr_tags) == tags
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE, None) == prompt_template
    assert (
        attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION, None) == prompt_template_version
    )
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, None) == json.dumps(
        prompt_template_variables
    )


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
