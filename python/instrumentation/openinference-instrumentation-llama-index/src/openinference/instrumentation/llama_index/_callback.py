import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from time import time_ns
from types import MappingProxyType
from typing import (
    Any,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    Union,
    cast,
)
from uuid import uuid4

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
from typing_extensions import TypeAlias, TypeGuard
from wrapt import ObjectProxy

from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import (
    BASE_TRACE_EVENT,
    CBEventType,
    EventPayload,
)
from llama_index.llms.types import ChatMessage, ChatResponse
from llama_index.response.schema import Response, StreamingResponse
from llama_index.tools import ToolMetadata

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_EventId: TypeAlias = str


@dataclass
class _EventData:
    span: trace_api.Span
    token: object
    parent_id: str
    payloads: List[Dict[str, Any]]
    exceptions: List[Exception]
    event_type: CBEventType
    attributes: Dict[str, Any]
    start_time: int
    end_time: Optional[int] = None
    is_dispatched: bool = False
    """
    `is_dispatched` is True when some other process is responsible for ending the span.
    Normally the span is ended by calls to `on_trace_end`, but in the case of streaming,
    the event data is attached to the stream object, which is responsible for ending
    the span when the stream is finished. The reason for doing this is to defer the
    addition of the output value attribute, which is only available when the stream is
    finished, and that can happen a lot later than when `on_trace_end` is called.
    """


def payload_to_semantic_attributes(
    event_type: CBEventType,
    payload: Dict[str, Any],
    is_event_end: bool = False,
) -> Dict[str, Any]:
    """
    Converts a LLMapp payload to a dictionary of semantic conventions compliant attributes.
    """
    attributes: Dict[str, Any] = {}
    if event_type in (CBEventType.NODE_PARSING, CBEventType.CHUNKING):
        # TODO(maybe): handle these events
        return attributes
    if EventPayload.CHUNKS in payload and EventPayload.EMBEDDINGS in payload:
        attributes[EMBEDDING_EMBEDDINGS] = [
            {EMBEDDING_TEXT: text, EMBEDDING_VECTOR: vector}
            for text, vector in zip(payload[EventPayload.CHUNKS], payload[EventPayload.EMBEDDINGS])
        ]
    if event_type is not CBEventType.RERANKING and EventPayload.QUERY_STR in payload:
        attributes[INPUT_VALUE] = payload[EventPayload.QUERY_STR]
    if event_type is not CBEventType.RERANKING and EventPayload.NODES in payload:
        attributes[RETRIEVAL_DOCUMENTS] = [
            {
                DOCUMENT_ID: node_with_score.node.node_id,
                DOCUMENT_SCORE: node_with_score.score,
                DOCUMENT_CONTENT: node_with_score.node.text,
                DOCUMENT_METADATA: node_with_score.node.metadata,
            }
            for node_with_score in payload[EventPayload.NODES]
        ]
    if EventPayload.PROMPT in payload:
        attributes[LLM_PROMPTS] = [payload[EventPayload.PROMPT]]
    if EventPayload.MESSAGES in payload:
        messages = payload[EventPayload.MESSAGES]
        # Messages is only relevant to the LLM invocation
        if event_type is CBEventType.LLM:
            attributes[LLM_INPUT_MESSAGES] = [
                _message_payload_to_attributes(message_data) for message_data in messages
            ]
        elif event_type is CBEventType.AGENT_STEP and len(messages):
            # the agent step contains a message that is actually the input
            # akin to the query_str
            attributes[INPUT_VALUE] = _message_payload_to_str(messages[0])
    if response := (payload.get(EventPayload.RESPONSE) or payload.get(EventPayload.COMPLETION)):
        attributes.update(_get_response_output(response))
        if raw := getattr(response, "raw", None):
            assert hasattr(raw, "get"), f"raw must be Mapping, found {type(raw)}"
            attributes.update(_get_output_messages(raw))
            if usage := raw.get("usage"):
                # OpenAI token counts are available on raw.usage but can also be
                # found in additional_kwargs. Thus the duplicate handling.
                attributes.update(_get_token_counts(usage))
        # Look for token counts in additional_kwargs of the completion payload
        # This is needed for non-OpenAI models
        if (additional_kwargs := getattr(response, "additional_kwargs", None)) is not None:
            attributes.update(_get_token_counts(additional_kwargs))
    if event_type is CBEventType.RERANKING:
        if EventPayload.TOP_K in payload:
            attributes[RERANKER_TOP_K] = payload[EventPayload.TOP_K]
        if EventPayload.MODEL_NAME in payload:
            attributes[RERANKER_MODEL_NAME] = payload[EventPayload.MODEL_NAME]
        if EventPayload.QUERY_STR in payload:
            attributes[RERANKER_QUERY] = payload[EventPayload.QUERY_STR]
        if nodes := payload.get(EventPayload.NODES):
            attributes[RERANKER_OUTPUT_DOCUMENTS if is_event_end else RERANKER_INPUT_DOCUMENTS] = [
                {
                    DOCUMENT_ID: node_with_score.node.node_id,
                    DOCUMENT_SCORE: node_with_score.score,
                    DOCUMENT_CONTENT: node_with_score.node.text,
                    DOCUMENT_METADATA: node_with_score.node.metadata,
                }
                for node_with_score in nodes
            ]
    if EventPayload.TOOL in payload:
        tool_metadata = cast(ToolMetadata, payload.get(EventPayload.TOOL))
        attributes[TOOL_NAME] = tool_metadata.name
        attributes[TOOL_DESCRIPTION] = tool_metadata.description
        attributes[TOOL_PARAMETERS] = tool_metadata.to_openai_tool()["function"]["parameters"]
    if EventPayload.SERIALIZED in payload:
        serialized = payload[EventPayload.SERIALIZED]
        if event_type is CBEventType.EMBEDDING:
            if model_name := serialized.get("model_name"):
                attributes[EMBEDDING_MODEL_NAME] = model_name
        if event_type is CBEventType.LLM:
            if model_name := serialized.get("model"):
                attributes[LLM_MODEL_NAME] = model_name
                invocation_parameters = _extract_invocation_parameters(serialized)
                invocation_parameters["model"] = model_name
                attributes[LLM_INVOCATION_PARAMETERS] = json.dumps(invocation_parameters)
    return attributes


def _extract_invocation_parameters(serialized: Mapping[str, Any]) -> Dict[str, Any]:
    # FIXME: this is only based on openai. Other models have different parameters.
    if not hasattr(serialized, "get"):
        return {}
    invocation_parameters: Dict[str, Any] = {}
    additional_kwargs = serialized.get("additional_kwargs")
    if additional_kwargs and isinstance(additional_kwargs, Mapping):
        invocation_parameters.update(additional_kwargs)
    for key in ("temperature", "max_tokens"):
        if (value := serialized.get(key)) is not None:
            invocation_parameters[key] = value
    return invocation_parameters


class OpenInferenceTraceCallbackHandler(BaseCallbackHandler):
    __slots__ = (
        "_tracer",
        "_context_api_token_stack",
        "_event_data",
        "_stragglers",
    )

    def __init__(self, tracer: trace_api.Tracer) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._tracer = tracer
        self._context_api_token_stack = _ContextApiTokenStack()
        self._event_data: Dict[_EventId, _EventData] = {}
        self._stragglers: Dict[_EventId, _EventData] = _Stragglers(capacity=1000)
        """
        Stragglers are events that have not ended before their traces have ended. An example
        of a straggler is the LLM event when streaming is true, in which case its `on_event_end`
        is only called after the trace has ended. If an event is a straggler, then we are still
        waiting for its `on_event_end` to be called after its trace has ended. However, this call
        may never take place. Therefore, to prevent memory leak, this container is limited to a
        certain capacity. When it exceeds that capacity, the oldest straggler by insertion order
        is popped and are forced to finish tracing.
        """

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        event_id = event_id or str(uuid4())
        parent_id = parent_id or BASE_TRACE_EVENT

        if payload is None:
            payloads, exceptions, attributes = [], [], {}
        else:
            payloads = [payload.copy()]
            exceptions = (
                [exception]
                if isinstance((exception := payload.get(EventPayload.EXCEPTION)), Exception)
                else []
            )
            try:
                attributes = payload_to_semantic_attributes(event_type, payload)
            except Exception:
                logger.exception(
                    f"Failed to convert payload to semantic attributes. "
                    f"event_type={event_type}, payload={payload}",
                )
                attributes = {}

        start_time = time_ns()
        span: trace_api.Span = self._tracer.start_span(name=event_type.value, start_time=start_time)
        span.set_attribute(OPENINFERENCE_SPAN_KIND, _get_span_kind(event_type).value)
        token = context_api.attach(trace_api.set_span_in_context(span))
        self._context_api_token_stack.append(token)
        self._event_data[event_id] = _EventData(
            span=span,
            token=token,
            parent_id=parent_id,
            start_time=start_time,
            event_type=event_type,
            payloads=payloads,
            exceptions=exceptions,
            attributes=attributes,
        )
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_data := self._event_data.get(event_id):
            is_straggler = False
        elif event_data := self._stragglers.get(event_id):
            # Stragglers are events that have not ended before their traces have ended.
            # When `on_trace_end` is called, any event without an end time is flagged
            # as a straggler. An example straggler is the LLM event when streaming is
            # true, in which case `on_event_end` for the LLM event is only called after
            # the stream has been consumed, but `on_trace_end` will be called before
            # the stream can be iterated on.
            is_straggler = True
        else:
            return

        event_data.end_time = time_ns()
        self._context_api_token_stack.pop(event_data.token)

        if payload is not None:
            event_data.payloads.append(payload.copy())
            if isinstance((exception := payload.get(EventPayload.EXCEPTION)), Exception):
                event_data.exceptions.append(exception)
            try:
                event_data.attributes.update(
                    payload_to_semantic_attributes(event_type, payload, is_event_end=True),
                )
            except Exception:
                logger.exception(
                    f"Failed to convert payload to semantic attributes. "
                    f"event_type={event_type}, payload={payload}",
                )
            if (
                not is_straggler
                and _is_streaming_response(response := payload.get(EventPayload.RESPONSE))
                and response.response_gen is not None
            ):
                response.response_gen = _ResponseGen(response.response_gen, event_data)
                event_data.is_dispatched = True

        if is_straggler:
            # This can happen if `on_event_end` is called after `on_trace_end`, e.g. in the case of
            # the LLM span when streaming is true.
            _finish_tracing(event_data)
            self._stragglers.pop(event_id)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self._event_data.clear()

    def end_trace(self, trace_id: Optional[str] = None, *args: Any, **kwargs: Any) -> None:
        roots, adjacency_list = _build_graph(self._event_data)
        self._event_data.clear()
        dfs_stack = roots.copy()
        while dfs_stack:
            event_id, event_data = dfs_stack.pop()
            event_type = event_data.event_type
            attributes = event_data.attributes

            if event_type is CBEventType.LLM:
                # Need to update the template variables attribute by extracting their values
                # from the template events, which are sibling events preceding the LLM event.
                # Note that the stack should be sorted in chronological order, and we are
                # going backwards in time on the stack.
                while dfs_stack:
                    _, preceding_event_data = dfs_stack[-1]
                    if (
                        preceding_event_data.parent_id != event_data.parent_id
                        or preceding_event_data.event_type is not CBEventType.TEMPLATING
                    ):
                        break
                    dfs_stack.pop()  # remove template event from stack
                    for payload in preceding_event_data.payloads:
                        # Add template attributes to the LLM span to which they belong.
                        try:
                            attributes.update(_template_attributes(payload))
                        except Exception:
                            logger.exception(
                                f"Failed to convert payload to semantic attributes. "
                                f"event_type={preceding_event_data.event_type}, payload={payload}",
                            )

            if event_data.end_time is None:
                self._stragglers[event_id] = event_data
            elif not event_data.is_dispatched:
                _finish_tracing(event_data)
            dfs_stack.extend(adjacency_list[event_id])


class _ContextApiTokenStack:
    """
    Allows popping a token from the middle of the stack, whereby all tokens above
    it are also popped. Popping a non-existent token is allowed and has no effect.
    """

    __slots__ = ("_stack", "_positions")

    def __init__(self) -> None:
        self._stack: List[object] = []
        self._positions: Dict[int, int] = {}

    def append(self, token: object) -> None:
        if id(token) in self._positions:
            return
        position = len(self._stack)
        self._positions[id(token)] = position
        self._stack.append(token)

    def pop(self, token: object) -> None:
        if (position := self._positions.get(id(token))) is None:
            return
        for token in reversed(self._stack[position:]):
            self._positions.pop(id(token))
            context_api.detach(token)
        self._stack = self._stack[:position]


class _Stragglers(OrderedDict[_EventId, _EventData]):
    """
    Stragglers are events that have not ended before their traces have ended. If an event is in
    this container, then we are still waiting for its `on_event_end` to be called after its trace
    has ended. However, this call may never take place. One example is when the LLM raises an
    exception in the following code location, in which case the LLM event never ends.
    https://github.com/run-llama/llama_index/blob/dcef41ee67925cccf1ee7bb2dd386bcf0564ba29/llama_index/llms/base.py#L62
    Therefore, to prevent memory leak, this container is limited to a certain capacity, and when it
    exceeds that capacity, the oldest straggler by insertion order is popped and are forced to
    finish tracing.
    """  # noqa: E501

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self._capacity = capacity

    def __setitem__(self, key: _EventId, value: _EventData) -> None:
        if key not in self and len(self) >= self._capacity > 0:
            # pop the oldest straggler by insertion order
            _, event_data = self.popitem(last=False)
            # FIXME: It's unclear whether we should call `_finish_tracing` here or simply drop
            # the event data, and if we call finish tracing, what status we should set it to.
            # Note that calling `_finish_tracing` here would set status_code to OK even though
            # en error likely had prevented `on_event_end` from being called. The `end_time`
            # would also be set to to now because stragglers have no `end_time` (because
            # `on_event_end` has not been called).
            _finish_tracing(event_data)
        super().__setitem__(key, value)


class _ResponseGen(ObjectProxy):  # type: ignore
    __slots__ = (
        "_self_tokens",
        "_self_is_finished",
        "_self_event_data",
    )

    def __init__(self, token_gen: Any, event_data: _EventData) -> None:
        super().__init__(token_gen)
        self._self_tokens: List[str] = []
        self._self_is_finished = False
        self._self_event_data = event_data

    def __iter__(self) -> "_ResponseGen":
        return self

    def __next__(self) -> str:
        # pass through mistaken calls
        if not hasattr(self.__wrapped__, "__next__"):
            self.__wrapped__.__next__()
        try:
            value: str = self.__wrapped__.__next__()
        except Exception as exception:
            # Note that the user can still try to iterate on the stream even
            # after it's consumed (or has errored out), but we don't want to
            # end the span more than once.
            if not self._self_is_finished:
                event_data = self._self_event_data
                span = event_data.span
                if isinstance(exception, StopIteration):
                    status_code = trace_api.StatusCode.OK
                else:
                    status_code = trace_api.StatusCode.ERROR
                    span.record_exception(exception)
                if output_value := "".join(self._self_tokens):
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_value)
                attributes = event_data.attributes
                try:
                    flattened_attributes = dict(_flatten(attributes))
                except Exception:
                    logger.exception(
                        f"Failed to flatten attributes. event_type={event_data.event_type}, "
                        f"attributes={attributes}",
                    )
                else:
                    span.set_attributes(flattened_attributes)
                span.set_status(status=status_code)
                end_time = event_data.end_time
                span.end(end_time=end_time)
                self._self_is_finished = True
            raise
        else:
            self._self_tokens.append(value)
            return value


def _build_graph(
    event_data: Mapping[_EventId, _EventData],
) -> Tuple[
    List[Tuple[_EventId, _EventData]],
    DefaultDict[_EventId, List[Tuple[_EventId, _EventData]]],
]:
    """
    Builds a graph from the event data and returns a tuple of root nodes and the adjacency list,
    where child nodes are sorted by ascending start time.
    """
    adjacency_list: DefaultDict[_EventId, List[Tuple[_EventId, _EventData]]] = defaultdict(list)
    roots: List[Tuple[_EventId, _EventData]] = []
    for id_, data in event_data.items():
        events = roots if data.parent_id == BASE_TRACE_EVENT else adjacency_list[data.parent_id]
        events.append((id_, data))
    for id_, children in adjacency_list.items():
        adjacency_list[id_] = sorted(children, key=lambda x: x[1].start_time)
    return roots, adjacency_list


def _finish_tracing(event_data: _EventData) -> None:
    if not (span := event_data.span):
        return
    attributes = event_data.attributes
    if event_data.exceptions:
        span.set_status(trace_api.StatusCode.ERROR)
        for exception in event_data.exceptions:
            span.record_exception(exception)
    else:
        span.set_status(trace_api.StatusCode.OK)
    try:
        span.set_attributes(dict(_flatten(attributes)))
    except Exception:
        logger.exception(
            f"Failed to set attributes on span. event_type={event_data.event_type}, "
            f"attributes={attributes}",
        )
    span.end(end_time=event_data.end_time)


def _get_span_kind(event_type: Optional[CBEventType]) -> OpenInferenceSpanKindValues:
    """Maps a CBEventType to a SpanKind.

    Args:
        event_type (CBEventType): LlamaIndex callback event type.

    Returns:
        SpanKind: The corresponding span kind.
    """
    if event_type is None:
        return OpenInferenceSpanKindValues.UNKNOWN
    return _SPAN_KINDS.get(event_type, OpenInferenceSpanKindValues.CHAIN)


def _message_payload_to_attributes(message: Any) -> Dict[str, Optional[str]]:
    if isinstance(message, ChatMessage):
        message_attributes = {
            MESSAGE_ROLE: message.role.value,
            MESSAGE_CONTENT: message.content,
        }
        # Parse the kwargs to extract the function name and parameters for function calling
        # NB: these additional kwargs exist both for 'agent' and 'function' roles
        if "name" in message.additional_kwargs:
            message_attributes[MESSAGE_NAME] = message.additional_kwargs["name"]
        if tool_calls := message.additional_kwargs.get("tool_calls"):
            assert isinstance(
                tool_calls, Iterable
            ), f"tool_calls must be Iterable, found {type(tool_calls)}"
            message_tool_calls = []
            for tool_call in tool_calls:
                if message_tool_call := dict(_get_tool_call(tool_call)):
                    message_tool_calls.append(message_tool_call)
            if message_tool_calls:
                message_attributes[MESSAGE_TOOL_CALLS] = message_tool_calls
        return message_attributes

    return {
        MESSAGE_ROLE: "user",  # assume user if not ChatMessage
        MESSAGE_CONTENT: str(message),
    }


def _message_payload_to_str(message: Any) -> Optional[str]:
    """Converts a message payload to a string, if possible"""
    if isinstance(message, ChatMessage):
        return message.content

    return str(message)


class _CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj: object) -> Any:
        try:
            return super().default(obj)
        except TypeError:
            if callable(as_dict := getattr(obj, "dict", None)):
                return as_dict()
            raise


def _get_response_output(response: Any) -> Iterator[Tuple[str, Any]]:
    """
    Gets output from response objects. This is needed since the string representation of some
    response objects includes extra information in addition to the content itself. In the
    case of an agent's ChatResponse the output may be a `function_call` object specifying
    the name of the function to call and the arguments to call it with.
    """
    if isinstance(response, ChatResponse):
        message = response.message
        if content := message.content:
            yield OUTPUT_VALUE, content
        else:
            yield OUTPUT_VALUE, json.dumps(message.additional_kwargs, cls=_CustomJSONEncoder)
            yield OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
    elif isinstance(response, Response):
        if response.response:
            yield OUTPUT_VALUE, response.response
    elif isinstance(response, StreamingResponse):
        return
    else:  # if the response has unknown type, make a best-effort attempt to get the output
        yield OUTPUT_VALUE, str(response)


def _get_message(message: object) -> Iterator[Tuple[str, Any]]:
    if role := getattr(message, "role", None):
        assert isinstance(role, str), f"content must be str, found {type(role)}"
        yield MESSAGE_ROLE, role
    if content := getattr(message, "content", None):
        assert isinstance(content, str), f"content must be str, found {type(content)}"
        yield MESSAGE_CONTENT, content
    if tool_calls := getattr(message, "tool_calls", None):
        assert isinstance(
            tool_calls, Iterable
        ), f"tool_calls must be Iterable, found {type(tool_calls)}"
        message_tool_calls = []
        for tool_call in tool_calls:
            if message_tool_call := dict(_get_tool_call(tool_call)):
                message_tool_calls.append(message_tool_call)
        if message_tool_calls:
            yield MESSAGE_TOOL_CALLS, message_tool_calls


def _get_output_messages(raw: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    assert hasattr(raw, "get"), f"raw must be Mapping, found {type(raw)}"
    if not (choices := raw.get("choices")):
        return
    assert isinstance(choices, Iterable), f"choices must be Iterable, found {type(choices)}"
    if messages := [
        dict(_get_message(message))
        for choice in choices
        if (message := getattr(choice, "message", None)) is not None
    ]:
        yield LLM_OUTPUT_MESSAGES, messages


def _get_token_counts(usage: Union[object, Mapping[str, Any]]) -> Iterator[Tuple[str, Any]]:
    """
    Yields token count attributes from a object or mapping
    """
    # Call the appropriate function based on the type of usage
    if isinstance(usage, Mapping):
        yield from _get_token_counts_from_mapping(usage)
    elif isinstance(usage, object):
        yield from _get_token_counts_from_object(usage)


def _get_token_counts_from_object(usage: object) -> Iterator[Tuple[str, Any]]:
    """
    Yields token count attributes from response.raw.usage
    """
    if (prompt_tokens := getattr(usage, "prompt_tokens", None)) is not None:
        yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if (completion_tokens := getattr(usage, "completion_tokens", None)) is not None:
        yield LLM_TOKEN_COUNT_COMPLETION, completion_tokens
    if (total_tokens := getattr(usage, "total_tokens", None)) is not None:
        yield LLM_TOKEN_COUNT_TOTAL, total_tokens


def _get_token_counts_from_mapping(
    usage_mapping: Mapping[str, Any],
) -> Iterator[Tuple[str, Any]]:
    """
    Yields token count attributes from a mapping (e.x. completion kwargs payload)
    """
    if (prompt_tokens := usage_mapping.get("prompt_tokens")) is not None:
        yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if (completion_tokens := usage_mapping.get("completion_tokens")) is not None:
        yield LLM_TOKEN_COUNT_COMPLETION, completion_tokens
    if (total_tokens := usage_mapping.get("total_tokens")) is not None:
        yield LLM_TOKEN_COUNT_TOTAL, total_tokens


def _template_attributes(payload: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Yields template attributes if present
    """
    if template := payload.get(EventPayload.TEMPLATE):
        yield LLM_PROMPT_TEMPLATE, template
    if template_vars := payload.get(EventPayload.TEMPLATE_VARS):
        yield LLM_PROMPT_TEMPLATE_VARIABLES, template_vars


def _get_tool_call(tool_call: object) -> Iterator[Tuple[str, Any]]:
    if function := getattr(tool_call, "function", None):
        if name := getattr(function, "name", None):
            assert isinstance(name, str), f"name must be str, found {type(name)}"
            yield TOOL_CALL_FUNCTION_NAME, name
        if arguments := getattr(function, "arguments", None):
            assert isinstance(arguments, str), f"arguments must be str, found {type(arguments)}"
            yield TOOL_CALL_FUNCTION_ARGUMENTS_JSON, arguments


def _is_streaming_response(response: Any) -> TypeGuard[StreamingResponse]:
    return isinstance(response, StreamingResponse)


def _flatten(mapping: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, List) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


_BILLION = 1_000_000_000

_SPAN_KINDS = MappingProxyType(
    {
        CBEventType.EMBEDDING: OpenInferenceSpanKindValues.EMBEDDING,
        CBEventType.LLM: OpenInferenceSpanKindValues.LLM,
        CBEventType.RETRIEVE: OpenInferenceSpanKindValues.RETRIEVER,
        CBEventType.FUNCTION_CALL: OpenInferenceSpanKindValues.TOOL,
        CBEventType.AGENT_STEP: OpenInferenceSpanKindValues.AGENT,
        CBEventType.RERANKING: OpenInferenceSpanKindValues.RERANKER,
    }
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
