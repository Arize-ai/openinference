import json
import logging
from importlib import import_module
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

import openinference.instrumentation as oi
from openinference.instrumentation import (
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
    get_retriever_attributes,
    get_span_kind_attributes,
    get_tool_attributes,
    safe_json_dumps,
)
from openinference.semconv.trace import (
    OpenInferenceLLMProviderValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# qwen-agent uses role="function" for tool results, while OpenInference (and the
# OpenAI wire format qwen-agent converts to) use role="tool".
_FUNCTION_ROLE = "function"
_TOOL_ROLE = "tool"

# qwen-agent classes resolved lazily and cached: this module must be importable
# before qwen_agent is patched, and the lookups happen on every span.
_CLASS_CACHE: Dict[Tuple[str, str], Optional[type]] = {}


def _field(obj: Any, *names: str) -> Any:
    """Read the first present field from a mapping or an object.

    qwen-agent accepts a pydantic ``Message`` or a plain dict wherever a message
    or content item is expected, and returns whichever shape it was given, so
    both reach these wrappers.
    """
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj:
                return obj[name]
            continue
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _normalize_message(message: Any) -> Dict[str, Any]:
    """Convert a qwen-agent message to a plain dict."""
    if isinstance(message, Mapping):
        return dict(message)
    if callable(model_dump := getattr(message, "model_dump", None)):
        try:
            # Message.model_dump() defaults to exclude_none=True.
            dumped = model_dump()
            if isinstance(dumped, Mapping):
                return dict(dumped)
        except Exception:
            logger.debug("failed to dump qwen-agent message", exc_info=True)
    return {
        key: value
        for key in ("role", "content", "reasoning_content", "name", "function_call", "extra")
        if (value := getattr(message, key, None)) is not None
    }


def _normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    if messages is None:
        return []
    if isinstance(messages, (Mapping, str)) or not isinstance(messages, Sequence):
        messages = [messages]
    return [_normalize_message(message) for message in messages]


def _content_text(content: Any) -> str:
    """Flatten a message's content to plain text.

    Content is either a string or a list of ContentItem, each of which holds
    exactly one of text / image / file / audio / video.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "".join(text for item in content if isinstance(text := _field(item, "text"), str))
    return ""


def _message_contents(content: Any) -> List[oi.MessageContent]:
    """Convert a content field to OpenInference message content parts."""
    if isinstance(content, str):
        return [oi.TextMessageContent(type="text", text=content)] if content else []
    if not isinstance(content, Sequence):
        return []
    contents: List[oi.MessageContent] = []
    for item in content:
        if isinstance(text := _field(item, "text"), str) and text:
            contents.append(oi.TextMessageContent(type="text", text=text))
        elif isinstance(image := _field(item, "image"), str) and image:
            contents.append(oi.ImageMessageContent(type="image", image=oi.Image(url=image)))
    return contents


def _tool_call(message: Dict[str, Any]) -> Optional[oi.ToolCall]:
    if not (function_call := message.get("function_call")):
        return None
    tool_call = oi.ToolCall(
        function=oi.ToolCallFunction(
            name=_field(function_call, "name") or "",
            arguments=_field(function_call, "arguments") or "",
        )
    )
    if function_id := _field(message.get("extra"), "function_id"):
        tool_call["id"] = str(function_id)
    return tool_call


def _tool_call_id(message: Dict[str, Any]) -> Optional[str]:
    """The id of the tool call a role="function" message is answering."""
    if function_id := _field(message.get("extra"), "function_id"):
        return str(function_id)
    return None


def _to_oi_message(message: Dict[str, Any]) -> oi.Message:
    role = message.get("role") or "user"
    content = message.get("content")

    oi_message = oi.Message(role=_TOOL_ROLE if role == _FUNCTION_ROLE else str(role))

    contents: List[oi.MessageContent] = []
    if isinstance(reasoning := message.get("reasoning_content"), (str, list)):
        if reasoning_text := _content_text(reasoning):
            contents.append(oi.ReasoningMessageContent(type="reasoning", text=reasoning_text))

    if isinstance(content, str):
        # Plain-string content is by far the common case; keep it on `content`
        # so it renders as a simple message rather than a single-part list.
        if content:
            oi_message["content"] = content
    else:
        contents.extend(_message_contents(content))

    if contents:
        oi_message["contents"] = contents

    if role == _FUNCTION_ROLE and (tool_call_id := _tool_call_id(message)):
        oi_message["tool_call_id"] = tool_call_id

    if tool_call := _tool_call(message):
        oi_message["tool_calls"] = [tool_call]

    return oi_message


def _to_oi_messages(messages: Sequence[Dict[str, Any]]) -> List[oi.Message]:
    oi_messages: List[oi.Message] = []
    for message in messages:
        try:
            oi_messages.append(_to_oi_message(message))
        except Exception:
            logger.debug("failed to convert qwen-agent message", exc_info=True)
    return oi_messages


def _to_oi_tools(functions: Any) -> List[oi.Tool]:
    """Convert the `functions` argument of BaseChatModel.chat to OI tools.

    Entries are `BaseTool.function` dicts (`{name, description, parameters}`),
    or OpenAI-style `{"type": "function", "function": {...}}` wrappers when
    `use_raw_api` is in play.
    """
    if not isinstance(functions, Sequence) or isinstance(functions, (str, Mapping)):
        return []
    tools: List[oi.Tool] = []
    for function in functions:
        schema = function
        if isinstance(function, Mapping) and "function" in function:
            schema = function["function"]
        if not isinstance(schema, Mapping):
            continue
        tool = oi.Tool(json_schema=safe_json_dumps(schema))
        if isinstance(name := schema.get("name"), str) and name:
            tool["name"] = name
        if isinstance(description := schema.get("description"), str) and description:
            tool["description"] = description
        tools.append(tool)
    return tools


def _retrieved_documents(result: Any) -> Optional[List[oi.Document]]:
    """Convert the retrieval tool's output into OpenInference documents.

    `Retrieval.call` returns one entry per source document, shaped
    `{"url": ..., "text": [chunk, ...]}`, which `Agent._call_tool` then
    serialises to JSON. Every retrieved chunk becomes one document. qwen-agent
    computes relevance scores in `sort_by_scores` but discards them in
    `get_topk`, so no `document.score` is recorded.

    Returns None when the payload is not that shape, so the caller can fall
    back to treating the call as an ordinary tool.
    """
    payload = result
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(payload, Sequence) or isinstance(payload, (str, Mapping)):
        return None

    documents: List[oi.Document] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            return None
        texts = entry.get("text")
        if not isinstance(texts, Sequence) or isinstance(texts, str):
            return None
        url = entry.get("url")
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text:
                continue
            document = oi.Document(content=text)
            metadata: Dict[str, Any] = {"chunk_index": index}
            if isinstance(url, str) and url:
                document["id"] = url
                metadata["url"] = url
            document["metadata"] = metadata
            documents.append(document)
    return documents


def _int_field(usage: Any, *names: str) -> Optional[int]:
    for name in names:
        value = _field(usage, name)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return None


def _token_count(messages: Sequence[Dict[str, Any]]) -> Optional[oi.TokenCount]:
    """Extract token counts from a chat response, when qwen-agent exposes them.

    Only the DashScope backends stash the raw model response on
    ``Message.extra["model_service_info"]``; the OpenAI-compatible backends
    discard it, so on those the token counts live on the nested OpenAI-SDK span
    emitted by ``openinference-instrumentation-openai`` instead. Never counting
    the same tokens on two spans matters because trace-level totals are summed
    across every span in the trace.
    """
    for message in reversed(messages):
        info = _field(message.get("extra"), "model_service_info")
        if info is None:
            continue
        if (usage := _field(info, "usage")) is None:
            continue
        prompt = _int_field(usage, "input_tokens", "prompt_tokens")
        completion = _int_field(usage, "output_tokens", "completion_tokens")
        total = _int_field(usage, "total_tokens")
        if not any((prompt, completion, total)):
            # `BaseChatModel.quick_chat_oai` fabricates an all-zero usage block.
            continue
        token_count = oi.TokenCount()
        if prompt is not None:
            token_count["prompt"] = prompt
        if completion is not None:
            token_count["completion"] = completion
        if total is not None:
            token_count["total"] = total
        elif prompt is not None and completion is not None:
            token_count["total"] = prompt + completion
        return token_count
    return None


def _provider(model_type: Optional[str]) -> Optional[OpenInferenceLLMProviderValues]:
    """Infer the provider from the configured `model_type`.

    Only Azure is unambiguous. The `oai` family points at an arbitrary
    OpenAI-compatible server and qwen-agent does not retain the base URL on the
    model instance, so there is nothing to infer from; DashScope has no
    OpenInference provider value yet. `qwen_agent.llm.model_type` carries the
    raw value in both cases.
    """
    if model_type == "azure":
        return OpenInferenceLLMProviderValues.AZURE
    return None


def _invocation_parameters(instance: Any, extra_generate_cfg: Any) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {}
    if isinstance(base := getattr(instance, "generate_cfg", None), Mapping):
        parameters.update(base)
    if isinstance(extra_generate_cfg, Mapping):
        parameters.update(extra_generate_cfg)
    return parameters


def _final_answer(response: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The agent's answer: the last assistant message carrying text.

    An agent run yields the whole response history — intermediate assistant
    text, tool calls and tool results — so the answer is found from the end.
    """
    for message in reversed(response):
        role = message.get("role")
        if role in (_FUNCTION_ROLE, _TOOL_ROLE) or message.get("function_call"):
            continue
        if text := _content_text(message.get("content")):
            return text
    return None


def _json_or_text(value: Any) -> Tuple[str, OpenInferenceMimeTypeValues]:
    """Serialize a tool argument or result, preferring JSON when it is JSON."""
    if isinstance(value, str):
        try:
            json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value, OpenInferenceMimeTypeValues.TEXT
        return value, OpenInferenceMimeTypeValues.JSON
    return safe_json_dumps(value), OpenInferenceMimeTypeValues.JSON


def _is_suppressed() -> bool:
    return bool(context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY))


def _safe_attributes(
    build: Callable[[], Dict[str, AttributeValue]],
    fallback: Dict[str, AttributeValue],
) -> Dict[str, AttributeValue]:
    """Build span attributes without ever raising into user code."""
    try:
        return build()
    except Exception:
        logger.exception("failed to build qwen-agent span attributes")
        return fallback


class _RunWrapper:
    """Wraps `Agent.run`, producing one AGENT span per agent invocation.

    `Agent.run` is a generator function that yields the growing response
    message list, so the span is opened when iteration starts and closed when
    the caller drains (or abandons) the generator. Every step is pumped inside
    `use_span` so the LLM and TOOL spans created by the agent's own workflow
    nest underneath.

    `Agent.run_nonstream` is deliberately not wrapped: it calls `run`
    internally, so wrapping both would double every agent span.

    Token counts are deliberately not aggregated onto this span — they belong
    on LLM spans only, since trace totals are summed across all spans.
    """

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if _is_suppressed():
            return wrapped(*args, **kwargs)

        span_name, span_kind = _agent_span_name_and_kind(instance)
        attributes = _safe_attributes(
            lambda: {
                **get_span_kind_attributes(span_kind),
                **get_input_attributes(
                    _normalize_messages(args[0] if args else kwargs.get("messages")),
                    mime_type=OpenInferenceMimeTypeValues.JSON,
                ),
                **dict(_agent_attributes(instance)),
            },
            get_span_kind_attributes(span_kind),
        )

        def wrapped_generator() -> Iterator[Any]:
            span = self._tracer.start_span(span_name, attributes=attributes)
            response: List[Dict[str, Any]] = []

            def finish() -> None:
                try:
                    set_output(response)
                except Exception as error:
                    # The agent run itself succeeded, so the span stays OK; the
                    # failure to record its output is surfaced as an event.
                    logger.exception("failed to record qwen-agent agent output")
                    span.record_exception(error)
                span.set_status(trace_api.StatusCode.OK)

            def set_output(response: List[Dict[str, Any]]) -> None:
                if answer := _final_answer(response):
                    span.set_attributes(
                        get_output_attributes(answer, mime_type=OpenInferenceMimeTypeValues.TEXT)
                    )
                elif response:
                    span.set_attributes(
                        get_output_attributes(response, mime_type=OpenInferenceMimeTypeValues.JSON)
                    )

            try:
                with trace_api.use_span(span, end_on_exit=False):
                    stream = iter(wrapped(*args, **kwargs))
                while True:
                    with trace_api.use_span(span, end_on_exit=False):
                        try:
                            chunk = next(stream)
                        except StopIteration:
                            break
                    if chunk:
                        response = _normalize_messages(chunk)
                    yield chunk
            except GeneratorExit:
                with trace_api.use_span(span, end_on_exit=False):
                    if callable(close := getattr(stream, "close", None)):
                        close()
                    finish()
                raise
            except Exception as error:
                span.record_exception(error)
                span.set_status(
                    trace_api.Status(trace_api.StatusCode.ERROR, f"{type(error).__name__}: {error}")
                )
                raise
            else:
                with trace_api.use_span(span, end_on_exit=False):
                    finish()
            finally:
                span.end()

        return wrapped_generator()


def _agent_span_name_and_kind(instance: Any) -> Tuple[str, OpenInferenceSpanKindValues]:
    class_name = instance.__class__.__name__
    if _is_memory(instance):
        # Memory is an Agent subclass used for file management and RAG rather
        # than for reasoning, so it reads better as a CHAIN. Its `retrieval` and
        # `doc_parser` calls still surface as TOOL spans underneath.
        return f"{class_name}.run", OpenInferenceSpanKindValues.CHAIN
    name = getattr(instance, "name", None) or class_name
    return f"{name}.run", OpenInferenceSpanKindValues.AGENT


def _lookup_class(module_name: str, class_name: str) -> Optional[type]:
    key = (module_name, class_name)
    if key not in _CLASS_CACHE:
        try:
            candidate = getattr(import_module(module_name), class_name)
            _CLASS_CACHE[key] = candidate if isinstance(candidate, type) else None
        except Exception:  # pragma: no cover - these modules should always import
            logger.debug("could not resolve %s.%s", module_name, class_name, exc_info=True)
            _CLASS_CACHE[key] = None
    return _CLASS_CACHE[key]


def _is_memory(instance: Any) -> bool:
    memory_class = _lookup_class("qwen_agent.memory", "Memory")
    return memory_class is not None and isinstance(instance, memory_class)


def _is_retrieval(tool: Any) -> bool:
    """Whether a tool call is a document retrieval.

    Matched on `Retrieval` itself so subclasses and tools registered under
    another name are covered. Deliberately not matched on the name
    `"retrieval"`: a tool absent from `function_map` never really runs
    (`Agent._call_tool` returns "Tool ... does not exists."), and a differently
    implemented tool registered under that name would not return the document
    shape a retriever span needs.
    """
    retrieval_class = _lookup_class("qwen_agent.tools.retrieval", "Retrieval")
    return retrieval_class is not None and isinstance(tool, retrieval_class)


def _agent_attributes(instance: Any) -> Iterator[Tuple[str, AttributeValue]]:
    if name := getattr(instance, "name", None):
        yield SpanAttributes.AGENT_NAME, name
    if description := getattr(instance, "description", None):
        yield "qwen_agent.agent.description", description
    if isinstance(function_map := getattr(instance, "function_map", None), Mapping):
        if tool_names := list(function_map):
            yield "qwen_agent.agent.tools", tool_names
    if (llm := getattr(instance, "llm", None)) is not None:
        if model := getattr(llm, "model", None):
            yield SpanAttributes.LLM_MODEL_NAME, model
        if model_type := getattr(llm, "model_type", None):
            yield "qwen_agent.llm.model_type", model_type


class _ChatWrapper:
    """Wraps `BaseChatModel.chat`, producing one LLM span per model call.

    Every backend routes through this one method — the subclasses override
    `_chat_stream` / `_chat_no_stream` / `_chat_with_functions` instead — so
    wrapping it here covers DashScope, the OpenAI-compatible backends, Azure,
    `transformers` and OpenVINO with a single span and no duplication.

    `chat` returns a list when `stream=False` and an iterator of cumulative
    message lists when `stream=True` (which is what the agents use).
    """

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if _is_suppressed():
            return wrapped(*args, **kwargs)

        arguments = _bind_chat_arguments(args, kwargs)
        attributes = _safe_attributes(
            lambda: _chat_attributes(instance, arguments),
            get_span_kind_attributes(OpenInferenceSpanKindValues.LLM),
        )

        span = self._tracer.start_span(f"{instance.__class__.__name__}.chat", attributes=attributes)

        def finish(response: Sequence[Dict[str, Any]]) -> None:
            try:
                set_output(span, response)
            except Exception as error:
                # The model call itself succeeded, so the span stays OK; the
                # failure to record its output is surfaced as an event.
                logger.exception("failed to record qwen-agent llm output")
                span.record_exception(error)
            span.set_status(trace_api.StatusCode.OK)

        def set_output(span: trace_api.Span, response: Sequence[Dict[str, Any]]) -> None:
            span.set_attributes(
                get_llm_attributes(
                    output_messages=_to_oi_messages(response),
                    token_count=_token_count(response),
                )
            )
            if text := _content_text_of_response(response):
                span.set_attributes(
                    get_output_attributes(text, mime_type=OpenInferenceMimeTypeValues.TEXT)
                )
            elif response:
                span.set_attributes(
                    get_output_attributes(response, mime_type=OpenInferenceMimeTypeValues.JSON)
                )

        try:
            with trace_api.use_span(span, end_on_exit=False):
                result = wrapped(*args, **kwargs)
        except Exception as error:
            span.record_exception(error)
            span.set_status(
                trace_api.Status(trace_api.StatusCode.ERROR, f"{type(error).__name__}: {error}")
            )
            span.end()
            raise
        except BaseException:
            # KeyboardInterrupt / SystemExit: end the span so it is still
            # exported, but do not record it as a failure of the model call.
            span.end()
            raise

        if isinstance(result, list):
            with trace_api.use_span(span, end_on_exit=False):
                finish(_normalize_messages(result))
            span.end()
            return result

        return _ChatStream(
            result, span, finish, delta_stream=bool(arguments.get("delta_stream", False))
        )


def _merge_delta(
    accumulated: List[Dict[str, Any]], chunk: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Accumulate `delta_stream=True` fragments into whole messages.

    With `delta_stream=False` — the default, and the only mode the agents use —
    every yield is the full response so far and can simply replace the previous
    one. With `delta_stream=True` each yield is a fragment, so the text has to
    be concatenated or only the final fragment would be recorded.
    """
    for index, message in enumerate(chunk):
        if index >= len(accumulated):
            accumulated.append(dict(message))
            continue
        merged = accumulated[index]
        for field in ("content", "reasoning_content"):
            previous, addition = merged.get(field), message.get(field)
            if isinstance(previous, str) and isinstance(addition, str):
                merged[field] = previous + addition
            elif addition is not None and previous is None:
                merged[field] = addition
        previous_call = merged.get("function_call")
        addition_call = message.get("function_call")
        if previous_call is not None and addition_call is not None:
            merged["function_call"] = {
                "name": (_field(previous_call, "name") or "")
                + (_field(addition_call, "name") or ""),
                "arguments": (_field(previous_call, "arguments") or "")
                + (_field(addition_call, "arguments") or ""),
            }
        elif addition_call is not None:
            merged["function_call"] = addition_call
    return accumulated


class _ChatStream(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    """Keeps the LLM span open until the caller drains the response stream.

    An ``ObjectProxy`` rather than a generator so the caller still receives an
    object that behaves like qwen-agent's own iterator, and so ``__del__`` can
    close the span if the stream is abandoned without ever being iterated —
    otherwise that span would be created but never exported.
    """

    __slots__ = (
        "_self_span",
        "_self_finish",
        "_self_response",
        "_self_finished",
        "_self_delta_stream",
    )

    def __init__(
        self,
        stream: Any,
        span: trace_api.Span,
        finish: Callable[[Sequence[Dict[str, Any]]], None],
        delta_stream: bool,
    ) -> None:
        super().__init__(stream)
        self._self_span = span
        self._self_finish = finish
        self._self_response: List[Dict[str, Any]] = []
        self._self_finished = False
        self._self_delta_stream = delta_stream

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        return self._self_step(self.__wrapped__.__next__)

    def send(self, value: Any) -> Any:
        """Forward `send` through the span.

        On its normal streaming path `BaseChatModel.chat` returns a real
        generator, so the object handed back has to advance the span for every
        way of driving it, not just `__next__`, or the span outlives the stream.

        Defining `send`/`throw` here means `hasattr` reports them even when the
        wrapped object is a plain iterator (the response-cache path returns
        one); calling them then raises AttributeError, exactly as attribute
        access on that iterator would.
        """
        send = self.__wrapped__.send
        return self._self_step(lambda: send(value))

    def throw(self, *args: Any, **kwargs: Any) -> Any:
        """Forward `throw` through the span. See `send`."""
        throw = self.__wrapped__.throw
        return self._self_step(lambda: throw(*args, **kwargs))

    def _self_step(self, advance: Callable[[], Any]) -> Any:
        # The chunk is produced inside the span so that spans created by the
        # underlying model client nest beneath this one.
        with trace_api.use_span(self._self_span, end_on_exit=False):
            try:
                chunk = advance()
            except StopIteration:
                self._self_complete()
                raise
            except Exception as error:
                self._self_fail(error)
                raise
            except BaseException:
                # KeyboardInterrupt / SystemExit / a thrown GeneratorExit: end
                # the span without recording it as a failure of the model call.
                self._self_finished = True
                self._self_span.end()
                raise
        if chunk:
            messages = _normalize_messages(chunk)
            if self._self_delta_stream:
                self._self_response = _merge_delta(self._self_response, messages)
            else:
                self._self_response = messages
        return chunk

    def close(self) -> None:
        try:
            if callable(close := getattr(self.__wrapped__, "close", None)):
                # Closing runs the underlying generator's own cleanup, which may
                # create spans, so it happens inside this span.
                with trace_api.use_span(self._self_span, end_on_exit=False):
                    close()
        finally:
            # The span ends even if cleanup raises, rather than being left to
            # the garbage collector.
            self._self_complete()

    def _self_complete(self) -> None:
        """Record the response and end the span.

        Deliberately does not enter the span's context: it only sets attributes,
        and this runs from `__del__` too, where touching the OTel context would
        mean mutating whatever happens to be executing when the garbage
        collector fires.
        """
        if self._self_finished:
            return
        self._self_finished = True
        try:
            self._self_finish(self._self_response)
        finally:
            self._self_span.end()

    def _self_fail(self, error: BaseException) -> None:
        if self._self_finished:
            return
        self._self_finished = True
        self._self_span.record_exception(error)
        self._self_span.set_status(
            trace_api.Status(trace_api.StatusCode.ERROR, f"{type(error).__name__}: {error}")
        )
        self._self_span.end()

    def __del__(self) -> None:
        # Safety net for a stream that is dropped without being drained.
        # `wrapt.ObjectProxy` defines no __del__, so there is nothing to chain.
        try:
            if not getattr(self, "_self_finished", True):
                self._self_complete()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass


def _content_text_of_response(response: Sequence[Dict[str, Any]]) -> str:
    return "\n\n".join(
        text
        for message in response
        if not message.get("function_call") and (text := _content_text(message.get("content")))
    )


def _bind_chat_arguments(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind `BaseChatModel.chat(messages, functions, stream, delta_stream, extra_generate_cfg)`.

    Bound by position rather than with `inspect.signature` so that a signature
    change upstream degrades to missing attributes instead of raising.
    """
    names = ("messages", "functions", "stream", "delta_stream", "extra_generate_cfg")
    arguments: Dict[str, Any] = dict(kwargs)
    for name, value in zip(names, args):
        arguments[name] = value
    return arguments


class _ToolCallWrapper:
    """Wraps `Agent._call_tool`, producing one span per tool invocation.

    The span is a TOOL span, except for qwen-agent's document `retrieval` tool,
    which becomes a RETRIEVER span carrying the retrieved chunks as documents.

    Only `FnCallAgent` overrides `_call_tool`, and it delegates to
    `super()._call_tool(...)`, so wrapping the base method catches every tool
    call exactly once. The one path it misses is `FnCallAgent`'s early return
    for a tool that is not registered, which never reaches the base method.

    Note that `Agent._call_tool` swallows tool exceptions and returns the error
    text as the tool result, so a failing tool produces an OK span whose output
    is the error message. Only `ToolServiceError` and `DocParserError`
    propagate, and those are recorded on the span.
    """

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if _is_suppressed():
            return wrapped(*args, **kwargs)

        tool_name = str(args[0] if args else kwargs.get("tool_name", ""))
        tool_args = args[1] if len(args) > 1 else kwargs.get("tool_args", "{}")
        tool = None
        if isinstance(function_map := getattr(instance, "function_map", None), Mapping):
            tool = function_map.get(tool_name)
        is_retrieval = _is_retrieval(tool)
        span_kind = (
            OpenInferenceSpanKindValues.RETRIEVER
            if is_retrieval
            else OpenInferenceSpanKindValues.TOOL
        )
        attributes = _safe_attributes(
            lambda: _tool_call_attributes(tool, span_kind, instance, tool_name, tool_args, kwargs),
            get_span_kind_attributes(span_kind),
        )

        with self._tracer.start_as_current_span(f"{tool_name}.call", attributes=attributes) as span:
            result = wrapped(*args, **kwargs)
            try:
                output, output_mime_type = _json_or_text(result)
                span.set_attributes(get_output_attributes(output, mime_type=output_mime_type))
                if is_retrieval and (documents := _retrieved_documents(result)) is not None:
                    span.set_attributes(get_retriever_attributes(documents=documents))
            except Exception:
                logger.exception("failed to record qwen-agent tool output")
            span.set_status(trace_api.StatusCode.OK)
        return result


def _chat_attributes(instance: Any, arguments: Mapping[str, Any]) -> Dict[str, AttributeValue]:
    messages = _normalize_messages(arguments.get("messages"))
    model_type = getattr(instance, "model_type", None)
    attributes: Dict[str, AttributeValue] = {
        **get_span_kind_attributes(OpenInferenceSpanKindValues.LLM),
        **get_input_attributes(messages, mime_type=OpenInferenceMimeTypeValues.JSON),
        **get_llm_attributes(
            provider=_provider(model_type),
            model_name=getattr(instance, "model", None),
            invocation_parameters=_invocation_parameters(
                instance, arguments.get("extra_generate_cfg")
            ),
            input_messages=_to_oi_messages(messages),
            tools=_to_oi_tools(arguments.get("functions")),
        ),
    }
    if model_type:
        attributes["qwen_agent.llm.model_type"] = model_type
    return attributes


def _tool_call_attributes(
    tool: Any,
    span_kind: OpenInferenceSpanKindValues,
    instance: Any,
    tool_name: str,
    tool_args: Any,
    kwargs: Mapping[str, Any],
) -> Dict[str, AttributeValue]:
    tool_input, tool_input_mime_type = _json_or_text(tool_args)
    attributes: Dict[str, AttributeValue] = {
        **get_span_kind_attributes(span_kind),
        **dict(_tool_attributes(tool_name, tool)),
        **get_input_attributes(tool_input, mime_type=tool_input_mime_type),
        ToolCallAttributes.TOOL_CALL_FUNCTION_NAME: tool_name,
        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: tool_input,
    }
    if agent_name := getattr(instance, "name", None):
        attributes[SpanAttributes.AGENT_NAME] = agent_name
    if tool_call_id := _tool_call_id_from_messages(kwargs.get("messages"), tool_name):
        attributes[ToolCallAttributes.TOOL_CALL_ID] = tool_call_id
    return attributes


def _tool_attributes(tool_name: str, tool: Any) -> Iterator[Tuple[str, AttributeValue]]:
    raw_parameters = getattr(tool, "parameters", None)
    parameters: Union[str, Dict[str, Any]]
    if isinstance(raw_parameters, Mapping):
        parameters = dict(raw_parameters)
    elif isinstance(raw_parameters, str):
        parameters = raw_parameters
    else:
        # BaseTool.parameters may also be a list of parameter descriptors.
        parameters = safe_json_dumps(raw_parameters) if raw_parameters else "{}"
    attributes = get_tool_attributes(
        name=tool_name,
        description=getattr(tool, "description", None) or None,
        parameters=parameters,
    )
    yield from attributes.items()


def _tool_call_id_from_messages(messages: Any, tool_name: str) -> Optional[str]:
    """Best-effort lookup of the id of the tool call being executed.

    `Agent._call_tool` receives the tool name and arguments but not the id, so
    it is recovered from the message history `FnCallAgent` passes along: the
    call being executed is the first `function_call` for `tool_name` whose
    `extra["function_id"]` has no answering `role="function"` message yet.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, Mapping)):
        return None

    answered: Set[str] = set()
    pending: List[str] = []
    for message in messages:
        function_id = _field(_field(message, "extra"), "function_id")
        role = _field(message, "role")
        if role in (_FUNCTION_ROLE, _TOOL_ROLE):
            if function_id:
                answered.add(str(function_id))
            continue
        function_call = _field(message, "function_call")
        if function_call and _field(function_call, "name") == tool_name and function_id:
            pending.append(str(function_id))

    for pending_id in pending:
        if pending_id not in answered:
            return pending_id
    return None


__all__ = [
    "_ChatWrapper",
    "_RunWrapper",
    "_ToolCallWrapper",
]
