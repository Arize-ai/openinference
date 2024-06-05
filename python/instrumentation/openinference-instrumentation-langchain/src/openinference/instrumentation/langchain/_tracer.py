import json
import logging
import math
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from itertools import chain
from threading import RLock
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)
from uuid import UUID

import wrapt  # type: ignore
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
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
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.semconv.trace import SpanAttributes as OTELSpanAttributes
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_AUDIT_TIMING = False


@wrapt.decorator  # type: ignore
def audit_timing(wrapped: Any, _: Any, args: Any, kwargs: Any) -> Any:
    if not _AUDIT_TIMING:
        return wrapped(*args, **kwargs)
    start_time = time.perf_counter()
    try:
        return wrapped(*args, **kwargs)
    finally:
        latency_ms = (time.perf_counter() - start_time) * 1000
        print(f"{wrapped.__name__}: {latency_ms:.2f}ms")


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class _DictWithLock(ObjectProxy, Generic[K, V]):  # type: ignore
    """
    A wrapped dictionary with lock
    """

    def __init__(self, wrapped: Optional[Dict[str, V]] = None) -> None:
        super().__init__(wrapped or {})
        self._self_lock = RLock()

    def get(self, key: K) -> Optional[V]:
        with self._self_lock:
            return cast(Optional[V], self.__wrapped__.get(key))

    def pop(self, key: K, *args: Any) -> Optional[V]:
        with self._self_lock:
            return cast(Optional[V], self.__wrapped__.pop(key, *args))

    def __getitem__(self, key: K) -> V:
        with self._self_lock:
            return cast(V, super().__getitem__(key))

    def __setitem__(self, key: K, value: V) -> None:
        with self._self_lock:
            super().__setitem__(key, value)

    def __delitem__(self, key: K) -> None:
        with self._self_lock:
            super().__delitem__(key)


class OpenInferenceTracer(BaseTracer):
    __slots__ = ("_tracer", "_spans_by_run")

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if TYPE_CHECKING:
            # check that `run_map` still exists in parent class
            assert self.run_map
        self.run_map = _DictWithLock[str, Run](self.run_map)
        self._tracer = tracer
        self._spans_by_run: Dict[UUID, trace_api.Span] = _DictWithLock[UUID, trace_api.Span]()
        self._lock = RLock()  # handlers may be run in a thread by langchain

    @audit_timing  # type: ignore
    def _start_trace(self, run: Run) -> None:
        self.run_map[str(run.id)] = run
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return
        with self._lock:
            parent_context = (
                trace_api.set_span_in_context(parent)
                if (parent_run_id := run.parent_run_id)
                and (parent := self._spans_by_run.get(parent_run_id))
                else None
            )
        # We can't use real time because the handler may be
        # called in a background thread.
        start_time_utc_nano = _as_utc_nano(run.start_time)
        span = self._tracer.start_span(
            name=run.name,
            context=parent_context,
            start_time=start_time_utc_nano,
        )
        # The following line of code is commented out to serve as a reminder that in a system
        # of callbacks, attaching the context can be hazardous because there is no guarantee
        # that the context will be detached. An error could happen between callbacks leaving
        # the context attached forever, and all future spans will use it as parent. What's
        # worse is that the error could have also prevented the span from being exported,
        # leaving all future spans as orphans. That is a very bad scenario.
        # token = context_api.attach(context)
        with self._lock:
            self._spans_by_run[run.id] = span

    @audit_timing  # type: ignore
    def _end_trace(self, run: Run) -> None:
        self.run_map.pop(str(run.id), None)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return
        span = self._spans_by_run.pop(run.id, None)
        if span:
            try:
                _update_span(span, run)
            except Exception:
                logger.exception("Failed to update span with run data.")
            # We can't use real time because the handler may be
            # called in a background thread.
            end_time_utc_nano = _as_utc_nano(run.end_time) if run.end_time else None
            span.end(end_time=end_time_utc_nano)

    def _persist_run(self, run: Run) -> None:
        pass

    def on_llm_error(self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any) -> Run:
        if span := self._spans_by_run.get(run_id):
            _record_exception(span, error)
        return super().on_llm_error(error, *args, run_id=run_id, **kwargs)

    def on_chain_error(self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any) -> Run:
        if span := self._spans_by_run.get(run_id):
            _record_exception(span, error)
        return super().on_chain_error(error, *args, run_id=run_id, **kwargs)

    def on_retriever_error(
        self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any
    ) -> Run:
        if span := self._spans_by_run.get(run_id):
            _record_exception(span, error)
        return super().on_retriever_error(error, *args, run_id=run_id, **kwargs)

    def on_tool_error(self, error: BaseException, *args: Any, run_id: UUID, **kwargs: Any) -> Run:
        if span := self._spans_by_run.get(run_id):
            _record_exception(span, error)
        return super().on_tool_error(error, *args, run_id=run_id, **kwargs)

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> Run:
        """
        This emulates the behavior of the LangChainTracer.
        https://github.com/langchain-ai/langchain/blob/c01467b1f4f9beae8f1edb105b17aa4f36bf6573/libs/core/langchain_core/tracers/langchain.py#L115

        Although this method exists on the parent class, i.e. `BaseTracer`,
        it requires setting `self._schema_format = "original+chat"`.
        https://github.com/langchain-ai/langchain/blob/c01467b1f4f9beae8f1edb105b17aa4f36bf6573/libs/core/langchain_core/tracers/base.py#L170

        But currently self._schema_format is marked for internal use.
        https://github.com/langchain-ai/langchain/blob/c01467b1f4f9beae8f1edb105b17aa4f36bf6573/libs/core/langchain_core/tracers/base.py#L60
        """  # noqa: E501
        return LangChainTracer.on_chat_model_start(self, *args, **kwargs)  # type: ignore


@audit_timing  # type: ignore
def _record_exception(span: trace_api.Span, error: BaseException) -> None:
    if isinstance(error, Exception):
        span.record_exception(error)
        return
    exception_type = error.__class__.__name__
    exception_message = str(error)
    if not exception_message:
        exception_message = repr(error)
    attributes: Dict[str, AttributeValue] = {
        OTELSpanAttributes.EXCEPTION_TYPE: exception_type,
        OTELSpanAttributes.EXCEPTION_MESSAGE: exception_message,
        OTELSpanAttributes.EXCEPTION_ESCAPED: False,
    }
    try:
        # See e.g. https://github.com/open-telemetry/opentelemetry-python/blob/e9c7c7529993cd13b4af661e2e3ddac3189a34d0/opentelemetry-sdk/src/opentelemetry/sdk/trace/__init__.py#L967  # noqa: E501
        attributes[OTELSpanAttributes.EXCEPTION_STACKTRACE] = traceback.format_exc()
    except Exception:
        logger.exception("Failed to record exception stacktrace.")
    span.add_event(name="exception", attributes=attributes)


@audit_timing  # type: ignore
def _update_span(span: trace_api.Span, run: Run) -> None:
    if run.error is None:
        span.set_status(trace_api.StatusCode.OK)
    else:
        span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, run.error))
    span_kind = (
        OpenInferenceSpanKindValues.AGENT
        if "agent" in run.name.lower()
        else _langchain_run_type_to_span_kind(run.run_type)
    )
    span.set_attribute(OPENINFERENCE_SPAN_KIND, span_kind.value)
    span.set_attributes(dict(get_attributes_from_context()))
    span.set_attributes(
        dict(
            _flatten(
                chain(
                    _as_input(_convert_io(run.inputs)),
                    _as_output(_convert_io(run.outputs)),
                    _prompts(run.inputs),
                    _input_messages(run.inputs),
                    _output_messages(run.outputs),
                    _prompt_template(run),
                    _invocation_parameters(run),
                    _model_name(run.extra),
                    _token_counts(run.outputs),
                    _function_calls(run.outputs),
                    _tools(run),
                    _retrieval_documents(run),
                    _metadata(run),
                )
            )
        )
    )


def _langchain_run_type_to_span_kind(run_type: str) -> OpenInferenceSpanKindValues:
    try:
        return OpenInferenceSpanKindValues(run_type.upper())
    except ValueError:
        return OpenInferenceSpanKindValues.UNKNOWN


def stop_on_exception(
    wrapped: Callable[..., Iterator[Tuple[str, Any]]],
) -> Callable[..., Iterator[Tuple[str, Any]]]:
    def wrapper(*args: Any, **kwargs: Any) -> Iterator[Tuple[str, Any]]:
        start_time = time.perf_counter()
        try:
            yield from wrapped(*args, **kwargs)
        except Exception:
            logger.exception("Failed to get attribute.")
        finally:
            if _AUDIT_TIMING:
                latency_ms = (time.perf_counter() - start_time) * 1000
                print(f"{wrapped.__name__}: {latency_ms:.3f}ms")

    return wrapper


@stop_on_exception
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


@stop_on_exception
def _as_input(values: Iterable[str]) -> Iterator[Tuple[str, str]]:
    return zip((INPUT_VALUE, INPUT_MIME_TYPE), values)


@stop_on_exception
def _as_output(values: Iterable[str]) -> Iterator[Tuple[str, str]]:
    return zip((OUTPUT_VALUE, OUTPUT_MIME_TYPE), values)


def _convert_io(obj: Optional[Mapping[str, Any]]) -> Iterator[str]:
    if not obj:
        return
    assert isinstance(obj, dict), f"expected dict, found {type(obj)}"
    if len(obj) == 1 and isinstance(value := next(iter(obj.values())), str):
        yield value
    else:
        obj = dict(_replace_nan(obj))
        yield safe_json_dumps(obj)
        yield OpenInferenceMimeTypeValues.JSON.value


def _replace_nan(obj: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    for k, v in obj.items():
        if isinstance(v, float) and not math.isfinite(v):
            yield k, None
        else:
            yield k, v


@stop_on_exception
def _prompts(inputs: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, List[str]]]:
    """Yields prompts if present."""
    if not inputs:
        return
    assert hasattr(inputs, "get"), f"expected Mapping, found {type(inputs)}"
    if prompts := inputs.get("prompts"):
        yield LLM_PROMPTS, prompts


@stop_on_exception
def _input_messages(
    inputs: Optional[Mapping[str, Any]],
) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    """Yields chat messages if present."""
    if not inputs:
        return
    assert hasattr(inputs, "get"), f"expected Mapping, found {type(inputs)}"
    # There may be more than one set of messages. We'll use just the first set.
    if not (multiple_messages := inputs.get("messages")):
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
        parsed_messages.append(dict(_parse_message_data(message_data)))
    if parsed_messages:
        yield LLM_INPUT_MESSAGES, parsed_messages


@stop_on_exception
def _output_messages(
    outputs: Optional[Mapping[str, Any]],
) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    """Yields chat messages if present."""
    if not outputs:
        return
    assert hasattr(outputs, "get"), f"expected Mapping, found {type(outputs)}"
    # There may be more than one set of generations. We'll use just the first set.
    if not (multiple_generations := outputs.get("generations")):
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
            parsed_messages.append(dict(_parse_message_data(message_data)))
    if parsed_messages:
        yield LLM_OUTPUT_MESSAGES, parsed_messages


@stop_on_exception
def _parse_message_data(message_data: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, Any]]:
    """Parses message data to grab message role, content, etc."""
    if not message_data:
        return
    assert hasattr(message_data, "get"), f"expected Mapping, found {type(message_data)}"
    id_ = message_data.get("id")
    assert isinstance(id_, List), f"expected list, found {type(id_)}"
    message_class_name = id_[-1]
    if message_class_name.startswith("HumanMessage"):
        role = "user"
    elif message_class_name.startswith("AIMessage"):
        role = "assistant"
    elif message_class_name.startswith("SystemMessage"):
        role = "system"
    elif message_class_name.startswith("FunctionMessage"):
        role = "function"
    elif message_class_name.startswith("ToolMessage"):
        role = "tool"
    elif message_class_name.startswith("ChatMessage"):
        role = message_data["kwargs"]["role"]
    else:
        raise ValueError(f"Cannot parse message of type: {message_class_name}")
    yield MESSAGE_ROLE, role
    if kwargs := message_data.get("kwargs"):
        assert hasattr(kwargs, "get"), f"expected Mapping, found {type(kwargs)}"
        if content := kwargs.get("content"):
            assert isinstance(content, str), f"expected str, found {type(content)}"
            yield MESSAGE_CONTENT, content
        if additional_kwargs := kwargs.get("additional_kwargs"):
            assert hasattr(
                additional_kwargs, "get"
            ), f"expected Mapping, found {type(additional_kwargs)}"
            if function_call := additional_kwargs.get("function_call"):
                assert hasattr(
                    function_call, "get"
                ), f"expected Mapping, found {type(function_call)}"
                if name := function_call.get("name"):
                    assert isinstance(name, str), f"expected str, found {type(name)}"
                    yield MESSAGE_FUNCTION_CALL_NAME, name
                if arguments := function_call.get("arguments"):
                    assert isinstance(arguments, str), f"expected str, found {type(arguments)}"
                    yield MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON, arguments
            if tool_calls := additional_kwargs.get("tool_calls"):
                assert isinstance(
                    tool_calls, Iterable
                ), f"expected Iterable, found {type(tool_calls)}"
                message_tool_calls = []
                for tool_call in tool_calls:
                    if message_tool_call := dict(_get_tool_call(tool_call)):
                        message_tool_calls.append(message_tool_call)
                if message_tool_calls:
                    yield MESSAGE_TOOL_CALLS, message_tool_calls


@stop_on_exception
def _get_tool_call(tool_call: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, Any]]:
    if not tool_call:
        return
    assert hasattr(tool_call, "get"), f"expected Mapping, found {type(tool_call)}"
    if function := tool_call.get("function"):
        assert hasattr(function, "get"), f"expected Mapping, found {type(function)}"
        if name := function.get("name"):
            assert isinstance(name, str), f"expected str, found {type(name)}"
            yield TOOL_CALL_FUNCTION_NAME, name
        if arguments := function.get("arguments"):
            assert isinstance(arguments, str), f"expected str, found {type(arguments)}"
            yield TOOL_CALL_FUNCTION_ARGUMENTS_JSON, arguments


@stop_on_exception
def _prompt_template(run: Run) -> Iterator[Tuple[str, Any]]:
    """
    A best-effort attempt to locate the PromptTemplate object among the
    keyword arguments of a serialized object, e.g. an LLMChain object.
    """
    serialized: Optional[Mapping[str, Any]] = run.serialized
    if not serialized:
        return
    assert hasattr(serialized, "get"), f"expected Mapping, found {type(serialized)}"
    if not (kwargs := serialized.get("kwargs")):
        return
    assert isinstance(kwargs, dict), f"expected dict, found {type(kwargs)}"
    for obj in kwargs.values():
        if not hasattr(obj, "get") or not (id_ := obj.get("id")):
            continue
        # The `id` field of the object is a list indicating the path to the
        # object's class in the LangChain package, e.g. `PromptTemplate` in
        # the `langchain.prompts.prompt` module is represented as
        # ["langchain", "prompts", "prompt", "PromptTemplate"]
        assert isinstance(id_, Sequence), f"expected list, found {type(id_)}"
        if id_[-1].endswith("PromptTemplate"):
            if not (kwargs := obj.get("kwargs")):
                continue
            assert hasattr(kwargs, "get"), f"expected Mapping, found {type(kwargs)}"
            if not (template := kwargs.get("template", "")):
                continue
            yield LLM_PROMPT_TEMPLATE, template
            if input_variables := kwargs.get("input_variables"):
                assert isinstance(
                    input_variables, list
                ), f"expected list, found {type(input_variables)}"
                template_variables = {}
                for variable in input_variables:
                    if (value := run.inputs.get(variable)) is not None:
                        template_variables[variable] = value
                if template_variables:
                    yield LLM_PROMPT_TEMPLATE_VARIABLES, safe_json_dumps(template_variables)
            break


@stop_on_exception
def _invocation_parameters(run: Run) -> Iterator[Tuple[str, str]]:
    """Yields invocation parameters if present."""
    if run.run_type.lower() != "llm":
        return
    if not (extra := run.extra):
        return
    assert hasattr(extra, "get"), f"expected Mapping, found {type(extra)}"
    if invocation_parameters := extra.get("invocation_params"):
        assert isinstance(
            invocation_parameters, Mapping
        ), f"expected Mapping, found {type(invocation_parameters)}"
        yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_parameters)


@stop_on_exception
def _model_name(extra: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, str]]:
    """Yields model name if present."""
    if not extra:
        return
    assert hasattr(extra, "get"), f"expected Mapping, found {type(extra)}"
    if not (invocation_params := extra.get("invocation_params")):
        return
    for key in ["model_name", "model"]:
        if name := invocation_params.get(key):
            yield LLM_MODEL_NAME, name
            return


@stop_on_exception
def _token_counts(outputs: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, int]]:
    """Yields token count information if present."""
    if not outputs:
        return
    assert hasattr(outputs, "get"), f"expected Mapping, found {type(outputs)}"
    if not (llm_output := outputs.get("llm_output")):
        return
    assert hasattr(llm_output, "get"), f"expected Mapping, found {type(llm_output)}"
    if not (token_usage := llm_output.get("token_usage")):
        return
    assert hasattr(token_usage, "get"), f"expected Mapping, found {type(token_usage)}"
    for attribute_name, key in [
        (LLM_TOKEN_COUNT_PROMPT, "prompt_tokens"),
        (LLM_TOKEN_COUNT_COMPLETION, "completion_tokens"),
        (LLM_TOKEN_COUNT_TOTAL, "total_tokens"),
    ]:
        if (token_count := token_usage.get(key)) is not None:
            yield attribute_name, token_count


@stop_on_exception
def _function_calls(outputs: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, str]]:
    """Yields function call information if present."""
    if not outputs:
        return
    assert hasattr(outputs, "get"), f"expected Mapping, found {type(outputs)}"
    try:
        function_call_data = deepcopy(
            outputs["generations"][0][0]["message"]["kwargs"]["additional_kwargs"]["function_call"]
        )
        function_call_data["arguments"] = json.loads(function_call_data["arguments"])
        yield LLM_FUNCTION_CALL, safe_json_dumps(function_call_data)
    except Exception:
        pass


@stop_on_exception
def _tools(run: Run) -> Iterator[Tuple[str, str]]:
    """Yields tool attributes if present."""
    if run.run_type.lower() != "tool":
        return
    if not (serialized := run.serialized):
        return
    assert hasattr(serialized, "get"), f"expected Mapping, found {type(serialized)}"
    if name := serialized.get("name"):
        yield TOOL_NAME, name
    if description := serialized.get("description"):
        yield TOOL_DESCRIPTION, description


@stop_on_exception
def _retrieval_documents(run: Run) -> Iterator[Tuple[str, List[Mapping[str, Any]]]]:
    if run.run_type.lower() != "retriever":
        return
    if not (outputs := run.outputs):
        return
    assert hasattr(outputs, "get"), f"expected Mapping, found {type(outputs)}"
    documents = outputs.get("documents")
    assert isinstance(documents, Iterable), f"expected Iterable, found {type(documents)}"
    yield RETRIEVAL_DOCUMENTS, [dict(_as_document(document)) for document in documents]


@stop_on_exception
def _metadata(run: Run) -> Iterator[Tuple[str, str]]:
    """
    Takes the LangChain chain metadata and adds it to the trace
    """
    if not run.extra or not (metadata := run.extra.get("metadata")):
        return
    assert isinstance(metadata, Mapping), f"expected Mapping, found {type(metadata)}"
    if session_id := (
        metadata.get(LANGCHAIN_SESSION_ID)
        or metadata.get(LANGCHAIN_CONVERSATION_ID)
        or metadata.get(LANGCHAIN_THREAD_ID)
    ):
        yield SESSION_ID, session_id
    yield METADATA, safe_json_dumps(metadata)


@stop_on_exception
def _as_document(document: Any) -> Iterator[Tuple[str, Any]]:
    if page_content := getattr(document, "page_content", None):
        assert isinstance(page_content, str), f"expected str, found {type(page_content)}"
        yield DOCUMENT_CONTENT, page_content
    if metadata := getattr(document, "metadata", None):
        assert isinstance(metadata, Mapping), f"expected Mapping, found {type(metadata)}"
        yield DOCUMENT_METADATA, safe_json_dumps(metadata)


def _as_utc_nano(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1_000_000_000)


LANGCHAIN_SESSION_ID = "session_id"
LANGCHAIN_CONVERSATION_ID = "conversation_id"
LANGCHAIN_THREAD_ID = "thread_id"

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
METADATA = SpanAttributes.METADATA
SESSION_ID = SpanAttributes.SESSION_ID
