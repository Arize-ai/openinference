import inspect
import logging
import weakref
from dataclasses import dataclass
from enum import Enum, auto
from functools import singledispatch, singledispatchmethod
from queue import SimpleQueue
from threading import RLock, Thread
from time import sleep, time, time_ns
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

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
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span, Status, StatusCode, Tracer, set_span_in_context
from opentelemetry.util.types import AttributeValue
from pydantic import PrivateAttr
from typing_extensions import assert_never

from llama_index.core import QueryBundle, Response
from llama_index.core.base.agent.types import BaseAgent, BaseAgentWorker
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    AsyncStreamingResponse,
    PydanticResponse,
    StreamingResponse,
)
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepEndEvent,
    AgentChatWithStepStartEvent,
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
    AgentToolCallEvent,
)
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatDeltaReceivedEvent,
    StreamChatEndEvent,
    StreamChatErrorEvent,
    StreamChatStartEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatInProgressEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionInProgressEvent,
    LLMCompletionStartEvent,
    LLMPredictEndEvent,
    LLMPredictStartEvent,
    LLMStructuredPredictEndEvent,
    LLMStructuredPredictStartEvent,
)
from llama_index.core.instrumentation.events.query import (
    QueryEndEvent,
    QueryStartEvent,
)
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
from llama_index.core.instrumentation.events.span import (
    SpanDropEvent,
)
from llama_index.core.instrumentation.events.synthesis import (
    GetResponseEndEvent,
    GetResponseStartEvent,
    SynthesizeEndEvent,
    SynthesizeStartEvent,
)
from llama_index.core.instrumentation.span import BaseSpan
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler
from llama_index.core.schema import NodeWithScore, QueryType
from llama_index.core.types import RESPONSE_TEXT_TYPE

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


STREAMING_FINISHED_EVENTS = (
    LLMChatEndEvent,
    LLMCompletionEndEvent,
    StreamChatEndEvent,
)
STREAMING_IN_PROGRESS_EVENTS = (
    LLMChatInProgressEvent,
    LLMCompletionInProgressEvent,
    StreamChatDeltaReceivedEvent,
)


class _StreamingStatus(Enum):
    FINISHED = auto()
    IN_PROGRESS = auto()


class _Span(
    BaseSpan,
    extra="allow",
    keep_untouched=(singledispatchmethod, property),
):
    _otel_span: Span = PrivateAttr()
    _active: bool = PrivateAttr()
    _span_kind: Optional[str] = PrivateAttr()
    _parent: Optional["_Span"] = PrivateAttr()
    _first_token_timestamp: Optional[int] = PrivateAttr()

    end_time: Optional[int] = PrivateAttr()
    last_updated_at: float = PrivateAttr()

    def __init__(
        self,
        otel_span: Span,
        span_kind: Optional[str] = None,
        parent: Optional["_Span"] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._otel_span = otel_span
        self._active = otel_span.is_recording()
        self._span_kind = span_kind
        self._parent = parent
        self._first_token_timestamp = None
        self.end_time = None
        self.last_updated_at = time()

    def __setitem__(self, key: str, value: AttributeValue) -> None:
        self._otel_span.set_attribute(key, value)

    def record_exception(self, exception: BaseException) -> None:
        self._otel_span.record_exception(exception)

    def end(self, exception: Optional[BaseException] = None) -> None:
        if not self._active:
            return
        self._active = False
        if exception is None:
            status = Status(status_code=StatusCode.OK)
        else:
            self._otel_span.record_exception(exception)
            # Follow the format in OTEL SDK for description, see:
            # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
            description = f"{type(exception).__name__}: {exception}"
            status = Status(status_code=StatusCode.ERROR, description=description)
        self[OPENINFERENCE_SPAN_KIND] = self._span_kind or CHAIN
        self._otel_span.set_status(status=status)
        self._otel_span.end(end_time=self.end_time)

    @property
    def waiting_for_streaming(self) -> bool:
        return self._active and bool(self.end_time)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def context(self) -> context_api.Context:
        return set_span_in_context(self._otel_span)

    @singledispatchmethod
    def process_instance(self, instance: Any) -> None: ...

    @process_instance.register
    def _(self, instance: BaseLLM) -> None:
        if params := instance.metadata:
            self[LLM_MODEL_NAME] = params.model_name
            self[LLM_INVOCATION_PARAMETERS] = params.json(exclude_unset=True)

    @process_instance.register
    def _(self, instance: BaseEmbedding) -> None:
        if name := instance.model_name:
            self[EMBEDDING_MODEL_NAME] = name

    def process_event(self, event: BaseEvent) -> None:
        self._process_event(event)
        if not self.waiting_for_streaming:
            return
        if isinstance(event, STREAMING_FINISHED_EVENTS):
            self.end()
            self.notify_parent(_StreamingStatus.FINISHED)
        elif isinstance(event, STREAMING_IN_PROGRESS_EVENTS):
            if self._first_token_timestamp is None:
                timestamp = time_ns()
                self._otel_span.add_event("First Token Stream Event", timestamp=timestamp)
                self._first_token_timestamp = timestamp
            self.last_updated_at = time()
            self.notify_parent(_StreamingStatus.IN_PROGRESS)

    def notify_parent(self, status: _StreamingStatus) -> None:
        if not (parent := self._parent) or not parent.waiting_for_streaming:
            return
        if status is _StreamingStatus.IN_PROGRESS:
            parent.last_updated_at = time()
        else:
            parent.end()
        parent.notify_parent(status)

    @singledispatchmethod
    def _process_event(self, event: BaseEvent) -> None:
        logger.warning(f"Unhandled event of type {event.__class__.__qualname__}")

    @_process_event.register
    def _(self, event: AgentChatWithStepStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = AGENT
        self[INPUT_VALUE] = event.user_msg

    @_process_event.register
    def _(self, event: AgentChatWithStepEndEvent) -> None:
        self[OUTPUT_VALUE] = str(event.response)

    @_process_event.register
    def _(self, event: AgentRunStepStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = AGENT
        if input := event.input:
            self[INPUT_VALUE] = input

    @_process_event.register
    def _(self, event: AgentRunStepEndEvent) -> None:
        # FIXME: not sure what to do here with interim outputs since
        # there is no corresponding semantic convention.
        ...

    @_process_event.register
    def _(self, event: AgentToolCallEvent) -> None:
        tool = event.tool
        if name := tool.name:
            self[TOOL_NAME] = name
        self[TOOL_DESCRIPTION] = tool.description
        self[TOOL_PARAMETERS] = safe_json_dumps(tool.get_parameters_dict())

    @_process_event.register
    def _(self, event: EmbeddingStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = EMBEDDING

    @_process_event.register
    def _(self, event: EmbeddingEndEvent) -> None:
        for i, (text, vector) in enumerate(zip(event.chunks, event.embeddings)):
            self[f"{EMBEDDING_EMBEDDINGS}.{i}.{EMBEDDING_TEXT}"] = text
            self[f"{EMBEDDING_EMBEDDINGS}.{i}.{EMBEDDING_VECTOR}"] = vector

    @_process_event.register
    def _(self, event: StreamChatStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = LLM

    @_process_event.register
    def _(self, event: StreamChatDeltaReceivedEvent) -> None: ...

    @_process_event.register
    def _(self, event: StreamChatErrorEvent) -> None:
        self.record_exception(event.exception)

    @_process_event.register
    def _(self, event: StreamChatEndEvent) -> None: ...

    @_process_event.register
    def _(self, event: LLMPredictStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = LLM
        template = event.template
        self[LLM_PROMPT_TEMPLATE] = template.get_template()
        variable_names: List[str] = template.template_vars
        argument_values: Dict[str, str] = {
            **template.kwargs,
            **(event.template_args if event.template_args else {}),
        }
        template_arguments = {
            variable_name: argument_value
            for variable_name in variable_names
            if (argument_value := argument_values.get(variable_name)) is not None
        }
        if template_arguments:
            self[LLM_PROMPT_TEMPLATE_VARIABLES] = safe_json_dumps(template_arguments)

    @_process_event.register
    def _(self, event: LLMPredictEndEvent) -> None:
        self[OUTPUT_VALUE] = event.output

    @_process_event.register
    def _(self, event: LLMStructuredPredictStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = LLM

    @_process_event.register
    def _(self, event: LLMStructuredPredictEndEvent) -> None:
        self[OUTPUT_VALUE] = event.output.json(exclude_unset=True)
        self[OUTPUT_MIME_TYPE] = JSON

    @_process_event.register
    def _(self, event: LLMCompletionStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = LLM
        self[LLM_PROMPTS] = [event.prompt]

    @_process_event.register
    def _(self, event: LLMCompletionInProgressEvent) -> None: ...

    @_process_event.register
    def _(self, event: LLMCompletionEndEvent) -> None:
        self[OUTPUT_VALUE] = event.response.text
        self._extract_token_counts(event.response)

    @_process_event.register
    def _(self, event: LLMChatStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = LLM
        self._process_messages(LLM_INPUT_MESSAGES, *event.messages)

    @_process_event.register
    def _(self, event: LLMChatInProgressEvent) -> None: ...

    @_process_event.register
    def _(self, event: LLMChatEndEvent) -> None:
        if (response := event.response) is None:
            return
        self[OUTPUT_VALUE] = str(response)
        self._process_messages(LLM_OUTPUT_MESSAGES, response.message)
        self._extract_token_counts(response)

    @_process_event.register
    def _(self, event: QueryStartEvent) -> None:
        self._process_query_type(event.query)

    @_process_event.register
    def _(self, event: QueryEndEvent) -> None:
        self._process_response_type(event.response)

    @_process_event.register
    def _(self, event: ReRankStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = RERANKER
        self._process_query_type(event.query)
        self[RERANKER_TOP_K] = event.top_n
        self[RERANKER_MODEL_NAME] = event.model_name
        self._process_nodes(RERANKER_INPUT_DOCUMENTS, *event.nodes)

    @_process_event.register
    def _(self, event: ReRankEndEvent) -> None:
        self._process_nodes(RERANKER_OUTPUT_DOCUMENTS, *event.nodes)

    @_process_event.register
    def _(self, event: RetrievalStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = RETRIEVER
        self._process_query_type(event.str_or_query_bundle)

    @_process_event.register
    def _(self, event: RetrievalEndEvent) -> None:
        self._process_nodes(RETRIEVAL_DOCUMENTS, *event.nodes)

    @_process_event.register
    def _(self, event: SpanDropEvent) -> None:
        # Not needed because `prepare_to_drop_span()` provides the same information.
        ...

    @_process_event.register
    def _(self, event: SynthesizeStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = CHAIN
        self._process_query_type(event.query)

    @_process_event.register
    def _(self, event: SynthesizeEndEvent) -> None:
        self._process_response_type(event.response)

    @_process_event.register
    def _(self, event: GetResponseStartEvent) -> None:
        if not self._span_kind:
            self._span_kind = CHAIN
        self[INPUT_VALUE] = event.query_str

    @_process_event.register
    def _(self, event: GetResponseEndEvent) -> None:
        self._process_response_text_type(event.response)

    def _extract_token_counts(self, response: Union[ChatResponse, CompletionResponse]) -> None:
        if (
            (raw := getattr(response, "raw", None))
            and hasattr(raw, "get")
            and (usage := raw.get("usage"))
        ):
            for k, v in _get_token_counts(usage):
                self[k] = v
        # Look for token counts in additional_kwargs of the completion payload
        # This is needed for non-OpenAI models
        if additional_kwargs := getattr(response, "additional_kwargs", None):
            for k, v in _get_token_counts(additional_kwargs):
                self[k] = v

    def _process_nodes(self, prefix: str, *nodes: NodeWithScore) -> None:
        for i, node in enumerate(nodes):
            self[f"{prefix}.{i}.{DOCUMENT_ID}"] = node.node_id
            if content := node.get_content():
                self[f"{prefix}.{i}.{DOCUMENT_CONTENT}"] = content
            if (score := node.get_score()) is not None:
                self[f"{prefix}.{i}.{DOCUMENT_SCORE}"] = score
            if metadata := node.metadata:
                self[f"{prefix}.{i}.{DOCUMENT_METADATA}"] = safe_json_dumps(metadata)

    def _process_messages(self, prefix: str, *messages: ChatMessage) -> None:
        for i, message in enumerate(messages):
            self[f"{prefix}.{i}.{MESSAGE_ROLE}"] = message.role.value
            if content := message.content:
                self[f"{prefix}.{i}.{MESSAGE_CONTENT}"] = str(content)
            additional_kwargs = message.additional_kwargs
            if name := additional_kwargs.get("name"):
                self[f"{prefix}.{i}.{MESSAGE_NAME}"] = name
            if tool_calls := additional_kwargs.get("tool_calls"):
                for j, tool_call in enumerate(tool_calls):
                    for k, v in _get_tool_call(tool_call):
                        self[f"{prefix}.{i}.{MESSAGE_TOOL_CALLS}.{j}.{k}"] = v

    def _process_query_type(self, query: Optional[QueryType]) -> None:
        if query is None:
            return
        if isinstance(query, str):
            self[INPUT_VALUE] = query
        elif isinstance(query, QueryBundle):
            query_dict = {k: v for k, v in query.to_dict().items() if v is not None}
            query_dict.pop("embedding", None)  # because it takes up too much space
            if len(query_dict) == 1 and query.query_str:
                self[INPUT_VALUE] = query.query_str
            else:
                self[INPUT_VALUE] = safe_json_dumps(query_dict)
                self[INPUT_MIME_TYPE] = JSON
        else:
            assert_never(query)

    def _process_response_type(self, response: Optional[RESPONSE_TYPE]) -> None:
        if response is None:
            return
        if isinstance(response, (Response, PydanticResponse)):
            self._process_response_text_type(response.response)
        elif isinstance(response, (StreamingResponse, AsyncStreamingResponse)):
            pass
        else:
            assert_never(response)

    def _process_response_text_type(self, response: Optional[RESPONSE_TEXT_TYPE]) -> None:
        if response is None:
            return
        if isinstance(response, str):
            self[OUTPUT_VALUE] = response
        elif isinstance(response, BaseModel):
            self[OUTPUT_VALUE] = response.json(exclude_unset=True)
            self[OUTPUT_MIME_TYPE] = JSON
        elif isinstance(response, (Generator, AsyncGenerator)):
            pass
        else:
            assert_never(response)


END_OF_QUEUE = None


@dataclass
class _QueueItem:
    last_touched_at: float
    span: _Span


class _ExportQueue:
    """
    Container for spans that have ended but are waiting for streaming events. The
    list is periodically swept to evict items that are no longer active or have not
    been updated for over 60 seconds.
    """

    def __init__(self) -> None:
        self.lock: RLock = RLock()
        self.spans: Dict[str, _Span] = {}
        self.queue: "SimpleQueue[Optional[_QueueItem]]" = SimpleQueue()
        weakref.finalize(self, self.queue.put, END_OF_QUEUE)
        Thread(target=self._sweep, args=(self.queue,), daemon=True).start()

    def put(self, span: _Span) -> None:
        with self.lock:
            self.spans[span.id_] = span
        self.queue.put(_QueueItem(time(), span))

    def find(self, id_: str) -> Optional[_Span]:
        with self.lock:
            return self.spans.get(id_)

    def _del(self, item: _QueueItem) -> None:
        with self.lock:
            del self.spans[item.span.id_]

    def _sweep(self, q: "SimpleQueue[Optional[_QueueItem]]") -> None:
        while True:
            t = time()
            while not q.empty():
                if (item := q.get()) is END_OF_QUEUE:
                    return
                if t == item.last_touched_at:
                    # we have gone through the whole list
                    q.put(item)
                    break
                span = item.span
                if not span.active:
                    self._del(item)
                    continue
                if t - span.last_updated_at > 60:
                    span.end()
                    self._del(item)
                    continue
                item.last_touched_at = t
                q.put(item)
            sleep(0.1)


class _SpanHandler(BaseSpanHandler[_Span], extra="allow"):
    _otel_tracer: Tracer = PrivateAttr()
    export_queue: _ExportQueue = PrivateAttr()

    def __init__(self, tracer: Tracer) -> None:
        super().__init__()
        self._otel_tracer = tracer
        self.export_queue = _ExportQueue()

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[_Span]:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return None
        parent = self.open_spans.get(parent_span_id) if parent_span_id else None
        otel_span = self._otel_tracer.start_span(
            name=id_.partition("-")[0],
            start_time=time_ns(),
            attributes=dict(get_attributes_from_context()),
            context=(parent.context if parent else None),
        )
        span = _Span(
            otel_span=otel_span,
            span_kind=_init_span_kind(instance),
            parent=parent,
            id_=id_,
            parent_id=parent_span_id,
        )
        span.process_instance(instance)
        return span

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return None
        if span := self.open_spans.get(id_):
            if isinstance(instance, BaseLLM) and isinstance(
                result,
                (Generator, AsyncGenerator),
            ):
                span.end_time = time_ns()
                self.export_queue.put(span)
                return span
            span.end()
        else:
            logger.warning(f"Open span is missing for {id_=}")
        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return None
        if span := self.open_spans.get(id_):
            span.end(err)
        else:
            logger.warning(f"Open span is missing for {id_=}")
        return span


class EventHandler(BaseEventHandler, extra="allow"):
    span_handler: _SpanHandler = PrivateAttr()

    def __init__(self, tracer: Tracer) -> None:
        super().__init__()
        self.span_handler = _SpanHandler(tracer=tracer)

    def handle(self, event: BaseEvent, **kwargs: Any) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return None
        if not event.span_id:
            return event
        span = self.span_handler.open_spans.get(event.span_id)
        if span is None:
            span = self.span_handler.export_queue.find(event.span_id)
        if span is None:
            logger.warning(f"Open span is missing for {event.span_id=}, {event.id_=}")
        else:
            try:
                span.process_event(event)
            except Exception:
                logger.exception(f"Error processing event of type {event.__class__.__qualname__}")
                pass
        return event


def _get_tool_call(tool_call: object) -> Iterator[Tuple[str, Any]]:
    if function := getattr(tool_call, "function", None):
        if name := getattr(function, "name", None):
            yield TOOL_CALL_FUNCTION_NAME, name
        if arguments := getattr(function, "arguments", None):
            yield TOOL_CALL_FUNCTION_ARGUMENTS_JSON, arguments


def _get_token_counts(usage: Union[object, Mapping[str, Any]]) -> Iterator[Tuple[str, Any]]:
    if isinstance(usage, Mapping):
        return _get_token_counts_from_mapping(usage)
    if isinstance(usage, object):
        return _get_token_counts_from_object(usage)


def _get_token_counts_from_object(usage: object) -> Iterator[Tuple[str, Any]]:
    if (prompt_tokens := getattr(usage, "prompt_tokens", None)) is not None:
        yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if (completion_tokens := getattr(usage, "completion_tokens", None)) is not None:
        yield LLM_TOKEN_COUNT_COMPLETION, completion_tokens
    if (total_tokens := getattr(usage, "total_tokens", None)) is not None:
        yield LLM_TOKEN_COUNT_TOTAL, total_tokens


def _get_token_counts_from_mapping(
    usage_mapping: Mapping[str, Any],
) -> Iterator[Tuple[str, Any]]:
    if (prompt_tokens := usage_mapping.get("prompt_tokens")) is not None:
        yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if (completion_tokens := usage_mapping.get("completion_tokens")) is not None:
        yield LLM_TOKEN_COUNT_COMPLETION, completion_tokens
    if (total_tokens := usage_mapping.get("total_tokens")) is not None:
        yield LLM_TOKEN_COUNT_TOTAL, total_tokens


@singledispatch
def _init_span_kind(_: Any) -> Optional[str]:
    return None


@_init_span_kind.register
def _(_: BaseAgent) -> str:
    return AGENT


@_init_span_kind.register
def _(_: BaseAgentWorker) -> str:
    return AGENT


@_init_span_kind.register
def _(_: BaseLLM) -> str:
    return LLM


@_init_span_kind.register
def _(_: BaseRetriever) -> str:
    return RETRIEVER


@_init_span_kind.register
def _(_: BaseEmbedding) -> str:
    return EMBEDDING


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

JSON = OpenInferenceMimeTypeValues.JSON.value

AGENT = OpenInferenceSpanKindValues.AGENT.value
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
LLM = OpenInferenceSpanKindValues.LLM.value
RERANKER = OpenInferenceSpanKindValues.RERANKER.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
