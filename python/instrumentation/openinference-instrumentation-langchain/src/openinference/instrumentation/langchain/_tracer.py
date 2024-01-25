import json
import logging
from copy import deepcopy
from datetime import datetime
from enum import Enum
from itertools import chain
from typing import Any, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Tuple
from uuid import UUID

from langchain_core.tracers.base import BaseTracer
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    RerankerAttributes,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from langchain.callbacks.tracers.schemas import Run

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _Run(NamedTuple):
    span: trace_api.Span
    token: object  # token for OTEL context API


class OpenInferenceTracer(BaseTracer):
    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer
        self._runs: Dict[UUID, _Run] = {}
        # run_inline=True so the handler is not run in a thread. E.g. see the following location.
        # https://github.com/langchain-ai/langchain/blob/5c2538b9f7fb64afed2a918b621d9d8681c7ae32/libs/core/langchain_core/callbacks/manager.py#L321  # noqa: E501
        self.run_inline = True

    def _start_trace(self, run: Run) -> None:
        span = self._tracer.start_span(run.name)
        token = context_api.attach(trace_api.set_span_in_context(span))
        self._runs[run.id] = _Run(span=span, token=token)
        super()._start_trace(run)

    def _end_trace(self, run: Run) -> None:
        if event_data := self._runs.pop(run.id, None):
            context_api.detach(event_data.token)
            span = event_data.span
            try:
                # Note that this relies on pydantic for the serialization
                # of objects like `langchain_core.documents.Document`.
                if hasattr(run, "model_dump") and callable(run.model_dump):
                    _update_span(span, run.model_dump())
                else:
                    _update_span(span, run.dict())
            except Exception:
                logger.exception("Failed to update span with run data.")
            span.end()
        super()._end_trace(run)

    def _persist_run(self, run: Run) -> None:
        pass


def _update_span(span: trace_api.Span, run: Dict[str, Any]) -> None:
    if run["error"] is None:
        span.set_status(trace_api.StatusCode.OK)
    else:
        span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, run["error"]))
    span_kind = (
        OpenInferenceSpanKindValues.AGENT
        if "agent" in run["name"].lower()
        else _langchain_run_type_to_span_kind(run["run_type"])
    )
    span.set_attribute(OPENINFERENCE_SPAN_KIND, span_kind.value)
    for io_key, io_attributes in {
        "inputs": (INPUT_VALUE, INPUT_MIME_TYPE),
        "outputs": (OUTPUT_VALUE, OUTPUT_MIME_TYPE),
    }.items():
        span.set_attributes(dict(zip(io_attributes, _convert_io(run.get(io_key)))))
    span.set_attributes(
        dict(
            _flatten(
                chain(
                    _prompts(run["inputs"]),
                    _input_messages(run["inputs"]),
                    _output_messages(run["outputs"]),
                    _prompt_template(run["serialized"]),
                    _invocation_parameters(run),
                    _model_name(run["extra"]),
                    _token_counts(run["outputs"]),
                    _function_calls(run["outputs"]),
                    _tools(run),
                    _retrieval_documents(run),
                )
            )
        )
    )


def _flatten(key_values: Iterable[Tuple[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    for key, value in key_values:
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value.items()):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, List) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping.items()):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


Message = Dict[str, Any]


def _langchain_run_type_to_span_kind(run_type: str) -> OpenInferenceSpanKindValues:
    try:
        return OpenInferenceSpanKindValues(run_type.upper())
    except ValueError:
        return OpenInferenceSpanKindValues.UNKNOWN


def _serialize_json(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _convert_io(obj: Optional[Dict[str, Any]]) -> Iterator[str]:
    if not obj:
        return
    if not isinstance(obj, dict):
        raise ValueError(f"obj should be dict, but obj={obj}")
    if len(obj) == 1 and isinstance(value := next(iter(obj.values())), str):
        yield value
    else:
        yield json.dumps(obj, default=_serialize_json)
        yield OpenInferenceMimeTypeValues.JSON.value


def _prompts(run_inputs: Dict[str, Any]) -> Iterator[Tuple[str, List[str]]]:
    """Yields prompts if present."""
    if "prompts" in run_inputs:
        yield LLM_PROMPTS, run_inputs["prompts"]


def _input_messages(run_inputs: Mapping[str, Any]) -> Iterator[Tuple[str, List[Message]]]:
    """Yields chat messages if present."""
    if not hasattr(run_inputs, "get"):
        return
    # There may be more than one set of messages. We'll use just the first set.
    if not (multiple_messages := run_inputs.get("messages")):
        return
    assert isinstance(
        multiple_messages, Iterable
    ), f"expected Iterable, found {type(multiple_messages)}"
    # This will only get the first set of messages.
    if not (first_messages := next(iter(multiple_messages), None)):
        return
    assert isinstance(first_messages, Iterable), f"expected Iterable, found {type(first_messages)}"
    parsed_messages = []
    for message_data in first_messages:
        assert hasattr(message_data, "get"), f"expected Mapping, found {type(message_data)}"
        parsed_messages.append(_parse_message_data(message_data))
    if parsed_messages:
        yield LLM_INPUT_MESSAGES, parsed_messages


def _output_messages(run_outputs: Mapping[str, Any]) -> Iterator[Tuple[str, List[Message]]]:
    """Yields chat messages if present."""
    if not hasattr(run_outputs, "get"):
        return
    # There may be more than one set of generations. We'll use just the first set.
    if not (multiple_generations := run_outputs.get("generations")):
        return
    assert isinstance(
        multiple_generations, Iterable
    ), f"expected Iterable, found {type(multiple_generations)}"
    # This will only get the first set of generations.
    if not (first_generations := next(iter(multiple_generations), None)):
        return
    assert isinstance(
        first_generations, Iterable
    ), f"expected Iterable, found {type(first_generations)}"
    parsed_messages = []
    for generation in first_generations:
        assert hasattr(generation, "get"), f"expected Mapping, found {type(generation)}"
        if message_data := generation.get("message"):
            assert hasattr(message_data, "get"), f"expected Mapping, found {type(message_data)}"
            parsed_messages.append(_parse_message_data(message_data))
    if parsed_messages:
        yield LLM_OUTPUT_MESSAGES, parsed_messages


def _parse_message_data(message_data: Mapping[str, Any]) -> Message:
    """Parses message data to grab message role, content, etc."""
    message_class_name = message_data["id"][-1]
    if message_class_name.startswith("HumanMessage"):
        role = "user"
    elif message_class_name.startswith("AIMessage"):
        role = "assistant"
    elif message_class_name.startswith("SystemMessage"):
        role = "system"
    elif message_class_name.startswith("FunctionMessage"):
        role = "function"
    elif message_class_name.startswith("ChatMessage"):
        role = message_data["kwargs"]["role"]
    else:
        raise ValueError(f"Cannot parse message of type: {message_class_name}")
    parsed_message_data: Dict[str, Any] = {MESSAGE_ROLE: role}
    if kwargs := message_data.get("kwargs"):
        assert hasattr(kwargs, "get"), f"expected Mapping, found {type(kwargs)}"
        if content := kwargs.get("content"):
            assert isinstance(content, str), f"content must be str, found {type(content)}"
            parsed_message_data[MESSAGE_CONTENT] = content
        if additional_kwargs := kwargs.get("additional_kwargs"):
            assert hasattr(
                additional_kwargs, "get"
            ), f"expected Mapping, found {type(additional_kwargs)}"
            if function_call := additional_kwargs.get("function_call"):
                assert hasattr(
                    function_call, "get"
                ), f"expected Mapping, found {type(function_call)}"
                if name := function_call.get("name"):
                    assert isinstance(name, str), f"name must be str, found {type(name)}"
                    parsed_message_data[MESSAGE_FUNCTION_CALL_NAME] = name
                if arguments := function_call.get("arguments"):
                    assert isinstance(
                        arguments, str
                    ), f"arguments must be str, found {type(arguments)}"
                    parsed_message_data[MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON] = arguments
            if tool_calls := additional_kwargs.get("tool_calls"):
                assert isinstance(
                    tool_calls, Iterable
                ), f"tool_calls must be Iterable, found {type(tool_calls)}"
                message_tool_calls = []
                for tool_call in tool_calls:
                    if message_tool_call := dict(_get_tool_call(tool_call)):
                        message_tool_calls.append(message_tool_call)
                if message_tool_calls:
                    parsed_message_data[MESSAGE_TOOL_CALLS] = message_tool_calls
    return parsed_message_data


def _get_tool_call(tool_call: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    if function := tool_call.get("function"):
        assert hasattr(function, "get"), f"expected Mapping, found {type(function)}"
        if name := function.get("name"):
            assert isinstance(name, str), f"name must be str, found {type(name)}"
            yield TOOL_CALL_FUNCTION_NAME, name
        if arguments := function.get("arguments"):
            assert isinstance(arguments, str), f"arguments must be str, found {type(arguments)}"
            yield TOOL_CALL_FUNCTION_ARGUMENTS_JSON, arguments


def _prompt_template(run_serialized: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    A best-effort attempt to locate the PromptTemplate object among the
    keyword arguments of a serialized object, e.g. an LLMChain object.
    """
    for obj in run_serialized.get("kwargs", {}).values():
        if not isinstance(obj, dict) or "id" not in obj:
            continue
        # The `id` field of the object is a list indicating the path to the
        # object's class in the LangChain package, e.g. `PromptTemplate` in
        # the `langchain.prompts.prompt` module is represented as
        # ["langchain", "prompts", "prompt", "PromptTemplate"]
        if obj["id"][-1].endswith("PromptTemplate"):
            kwargs = obj.get("kwargs", {})
            if not (template := kwargs.get("template", "")):
                continue
            yield LLM_PROMPT_TEMPLATE, template
            yield LLM_PROMPT_TEMPLATE_VARIABLES, kwargs.get("input_variables", [])
            break


def _invocation_parameters(run: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    """Yields invocation parameters if present."""
    if run["run_type"] != "llm":
        return
    run_extra = run["extra"]
    yield LLM_INVOCATION_PARAMETERS, json.dumps(run_extra.get("invocation_params", {}))


def _model_name(run_extra: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    """Yields model name if present."""
    if not (invocation_params := run_extra.get("invocation_params")):
        return
    for key in ["model_name", "model"]:
        if name := invocation_params.get(key):
            yield LLM_MODEL_NAME, name
            return


def _token_counts(run_outputs: Dict[str, Any]) -> Iterator[Tuple[str, int]]:
    """Yields token count information if present."""
    try:
        token_usage = run_outputs["llm_output"]["token_usage"]
    except Exception:
        return
    for attribute_name, key in [
        (LLM_TOKEN_COUNT_PROMPT, "prompt_tokens"),
        (LLM_TOKEN_COUNT_COMPLETION, "completion_tokens"),
        (LLM_TOKEN_COUNT_TOTAL, "total_tokens"),
    ]:
        if (token_count := token_usage.get(key)) is not None:
            yield attribute_name, token_count


def _function_calls(run_outputs: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    """Yields function call information if present."""
    try:
        function_call_data = deepcopy(
            run_outputs["generations"][0][0]["message"]["kwargs"]["additional_kwargs"][
                "function_call"
            ]
        )
        function_call_data["arguments"] = json.loads(function_call_data["arguments"])
        yield LLM_FUNCTION_CALL, json.dumps(function_call_data)
    except Exception:
        pass


def _tools(run: Dict[str, Any]) -> Iterator[Tuple[str, str]]:
    """Yields tool attributes if present."""
    if run["run_type"] != "tool":
        return
    run_serialized = run["serialized"]
    if "name" in run_serialized:
        yield TOOL_NAME, run_serialized["name"]
    if "description" in run_serialized:
        yield TOOL_DESCRIPTION, run_serialized["description"]
    # TODO: tool parameters https://github.com/Arize-ai/phoenix/issues/1330


def _retrieval_documents(
    run: Dict[str, Any],
) -> Iterator[Tuple[str, List[Any]]]:
    if run["run_type"] != "retriever":
        return
    yield (
        RETRIEVAL_DOCUMENTS,
        [
            {
                DOCUMENT_CONTENT: document.get("page_content"),
                DOCUMENT_METADATA: document.get("metadata") or {},
            }
            for document in (run.get("outputs") or {}).get("documents") or []
        ],
    )


DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_FUNCTION_CALL = SpanAttributes.LLM_FUNCTION_CALL
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RERANKER_INPUT_DOCUMENTS = RerankerAttributes.RERANKER_INPUT_DOCUMENTS
RERANKER_MODEL_NAME = RerankerAttributes.RERANKER_MODEL_NAME
RERANKER_OUTPUT_DOCUMENTS = RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS
RERANKER_QUERY = RerankerAttributes.RERANKER_QUERY
RERANKER_TOP_K = RerankerAttributes.RERANKER_TOP_K
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS
