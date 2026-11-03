"""Tracing for Gemini Live sessions opened with ``client.aio.live.connect``.

Each interaction (from the user's input until the model hands control back to the
user) is a CHAIN span. Every model generation inside it is an LLM child span, and
every tool call is a TOOL child span that lasts until the client sends the tool
response. Interactions of the same session are tied together by ``session.id``.

Audio is never recorded. Input and output transcriptions, when enabled on the
session, are recorded as message text. URIs of remote audio and video files are
kept as references.
"""

import asyncio
import copy
import logging
import re
import time
import weakref
from contextvars import ContextVar
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_attributes_from_context,
    safe_json_dumps,
)
from openinference.instrumentation.google_genai._image_utils import (
    redact_images_from_output_value,
    redact_images_from_request_parameters,
)
from openinference.instrumentation.google_genai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.google_genai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.google_genai._utils import (
    _get_token_count_attributes_from_usage_metadata,
)
from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_CHAIN_SPAN_NAME = "AsyncSession"
_LLM_SPAN_NAME = "AsyncSession.generate"

CHAIN = OpenInferenceSpanKindValues.CHAIN.value
LLM = OpenInferenceSpanKindValues.LLM.value
TOOL = OpenInferenceSpanKindValues.TOOL.value

# Config fields that are not generation parameters, or are recorded elsewhere. The
# resumption handle lets a client resume the session, so it is not recorded.
_NON_INVOCATION_CONFIG_FIELDS = frozenset(
    ("system_instruction", "tools", "http_options", "session_resumption")
)

_sessions: "weakref.WeakKeyDictionary[Any, _LiveSessionTracer]" = weakref.WeakKeyDictionary()

# Set while connect() runs, to receive the connect config after the SDK has resolved it
# (MCP sessions turned into tools, system instruction turned into content).
_resolved_connect_config: ContextVar[Optional[list[Any]]] = ContextVar(
    "_resolved_connect_config", default=None
)


def _is_suppressed() -> bool:
    return bool(context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY))


def _is_interaction_complete(server_content: Any) -> bool:
    # Newer SDKs report the end of an interaction with ``interaction_status``; older
    # ones (and models that do not send it) only have ``turn_complete``.
    status = getattr(server_content, "interaction_status", None)
    status = getattr(status, "value", status)
    if status and status != "INTERACTION_STATUS_UNSPECIFIED":
        return bool(status == "IDLE")
    return bool(getattr(server_content, "turn_complete", None))


def _error_status(exception: BaseException) -> trace_api.Status:
    return trace_api.Status(
        status_code=trace_api.StatusCode.ERROR,
        description=f"{type(exception).__name__}: {exception}",
    )


_OK = trace_api.Status(status_code=trace_api.StatusCode.OK)

_AfterSend = Callable[[Optional[BaseException]], None]


def _is_dropped_media(part: Any) -> bool:
    # Only inline images are recorded; audio and video bytes are dropped. The check matches
    # image redaction, so that nothing kept here can get past it.
    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None and not (inline_data.mime_type or "").startswith("image/"):
        return True
    # A URI is kept as a reference to the file, unless it embeds the content itself.
    # Embedded images need an image mime type, which is what image redaction looks for.
    file_data = getattr(part, "file_data", None)
    file_uri = (getattr(file_data, "file_uri", None) or "").strip()
    if file_uri[:5].lower() != "data:":
        return False
    mime_type = getattr(file_data, "mime_type", None) or ""
    return not (file_uri.startswith("data:image/") and mime_type.startswith("image/"))


def _response_without_media(response: Any) -> Any:
    parts = response.parts or []
    kept = [part for part in parts if not _is_dropped_media(part)]
    return response if len(kept) == len(parts) else response.model_copy(update={"parts": kept})


def _without_media(content: Any) -> Any:
    parts = []
    for part in content.parts or ():
        if _is_dropped_media(part):
            continue
        if part.function_response is not None:
            part = part.model_copy(
                update={"function_response": _response_without_media(part.function_response)}
            )
        parts.append(part)
    return content.model_copy(update={"parts": parts}) if content.parts else content


# Keys and indexes of a JSON path such as $.days[0].city or $['city']
_JSON_PATH_TOKEN = re.compile(r"\[(\d+)\]|\[['\"]([^'\"]+)['\"]\]|([^.\[\]$]+)")


def _set_partial_arg(container: Any, keys: Sequence[Any], value: Any) -> None:
    key, *rest = keys
    if isinstance(container, list) and isinstance(key, int):
        container.extend([None] * (key + 1 - len(container)))
        current = container[key]
    elif isinstance(container, dict) and isinstance(key, str):
        current = container.get(key)
    else:
        return  # the path does not fit what was streamed so far
    if rest:
        if not isinstance(current, (dict, list)):
            current = [] if isinstance(rest[0], int) else {}
            container[key] = current
        _set_partial_arg(current, rest, value)
    elif isinstance(value, str) and isinstance(current, str):
        container[key] = current + value  # strings arrive in pieces
    else:
        container[key] = value


def _merge_function_call(stored: Any, frame: Any) -> Any:
    # Streamed arguments arrive as several frames for the same call, either as partial
    # values or as complete arguments on a later frame.
    args = frame.args if frame.args is not None else getattr(stored, "args", None)
    if partial_args := getattr(frame, "partial_args", None):
        args = copy.deepcopy(args) if args else {}
        for partial in partial_args:
            keys = [
                int(index) if index else (quoted or name)
                for index, quoted, name in _JSON_PATH_TOKEN.findall(partial.json_path or "")
            ]
            value: Any = None
            if partial.string_value is not None:
                value = partial.string_value
            elif partial.number_value is not None:
                value = partial.number_value
            elif partial.bool_value is not None:
                value = partial.bool_value
            elif partial.null_value is None:
                continue  # no value, only the end of this path
            if keys:
                _set_partial_arg(args, keys, value)
    return frame.model_copy(
        update={
            "id": frame.id or getattr(stored, "id", None),
            "name": frame.name or getattr(stored, "name", None),
            "args": args,
            "partial_args": None,
        }
    )


def _tool_error(response: Any) -> Any:
    # FunctionResponse reports a failed function with an "error" key in its response.
    value = response.response
    return value.get("error") if isinstance(value, Mapping) else None


def _usage_attributes(usage_metadata: Any) -> Iterator[tuple[str, AttributeValue]]:
    # Live reports usage once per model turn as ``UsageMetadata``, which names the
    # output counts ``response_*`` instead of ``candidates_*``.
    from google.genai import types

    usage = usage_metadata.model_dump(exclude_none=True)
    usage["candidates_token_count"] = usage.pop("response_token_count", None)
    usage["candidates_tokens_details"] = usage.pop("response_tokens_details", None)
    fields = types.GenerateContentResponseUsageMetadata.model_fields
    yield from _get_token_count_attributes_from_usage_metadata(
        types.GenerateContentResponseUsageMetadata.model_validate(
            {key: value for key, value in usage.items() if key in fields}
        )
    )


class _Generation:
    __slots__ = (
        "span",
        "system_contents",
        "texts",
        "thoughts",
        "transcript",
        "function_calls",
        "usage",
        "reason",
        "completed_at",
    )

    def __init__(self, span: _WithSpan, system_contents: Sequence[Any]) -> None:
        self.span = span
        # the system instruction in effect for this generation
        self.system_contents = system_contents
        self.texts: list[str] = []
        self.thoughts: list[str] = []
        self.transcript: list[str] = []
        self.function_calls: list[Any] = []
        self.usage: Any = None
        self.reason: Optional[str] = None
        # When the model finished. turn_complete can come later, once audio has played.
        self.completed_at: Optional[int] = None

    def visible_text(self) -> str:
        return "".join(self.texts) or "".join(self.transcript)

    def content(self) -> Any:
        from google.genai import types

        parts = []
        if thoughts := "".join(self.thoughts):
            parts.append(types.Part(text=thoughts, thought=True))
        if text := self.visible_text():
            parts.append(types.Part(text=text))
        parts.extend(types.Part(function_call=call) for call in self.function_calls)
        return types.Content(role="model", parts=parts)


class _EndedGeneration:
    """A generation that stopped at a tool call, kept open until the model continues,
    since usage for it may follow."""

    __slots__ = ("span", "status", "attributes", "end_time")

    def __init__(
        self,
        span: _WithSpan,
        status: Optional[trace_api.Status],
        attributes: dict[str, AttributeValue],
        end_time: Optional[int],
    ) -> None:
        self.span = span
        self.status = status
        self.attributes = attributes
        self.end_time = end_time or time.time_ns()


class _ToolCall:
    __slots__ = ("id", "name", "span", "answered", "error")

    def __init__(self, id_: Optional[str], name: Optional[str], span: _WithSpan) -> None:
        self.id = id_
        self.name = name
        self.span = span
        # the final response is being sent; the span ends once the send finishes
        self.answered = False
        # what the function reported as its failure, if it did
        self.error: Any = None


def _find_tool(
    tools: Iterable[_ToolCall],
    id_: Optional[str],
    name: Optional[str],
    include_answered: bool = False,
) -> Optional[_ToolCall]:
    # IDs are optional (Vertex AI does not require them); without one, match the
    # oldest open call with the same name.
    for tool in tools:
        if tool.answered and not include_answered:
            continue
        if (tool.id == id_) if id_ else (bool(name) and tool.name == name):
            return tool
    return None


def _is_silent(response: Any) -> bool:
    # SILENT responses to NON_BLOCKING calls are added to the context without prompting
    # the model; the other schedulings (WHEN_IDLE, the default, and INTERRUPT) prompt it.
    scheduling = (
        response.get("scheduling")
        if isinstance(response, Mapping)
        else getattr(response, "scheduling", None)
    )
    return bool(getattr(scheduling, "value", scheduling) == "SILENT")


class _Interaction:
    __slots__ = (
        "span",
        "context",
        "history",
        "user_texts",
        "input_transcript",
        "generation",
        "ended_generation",
        "tools",
        "last_output",
        "pending_usage",
        "input_started_at",
    )

    def __init__(
        self,
        span: _WithSpan,
        context: context_api.Context,
        input_started_at: Optional[int] = None,
    ) -> None:
        self.span = span
        self.context = context
        # when audio or video input began; the first generation starts from it as well
        self.input_started_at = input_started_at
        self.history: list[Any] = []
        self.user_texts: list[str] = []
        self.input_transcript: list[str] = []
        self.generation: Optional[_Generation] = None
        self.ended_generation: Optional[_EndedGeneration] = None
        self.tools: list[_ToolCall] = []
        self.last_output = ""
        # usage that arrived while no generation was open; it goes on the next one
        self.pending_usage: Any = None

    def find_tool(self, id_: Optional[str], name: Optional[str]) -> Optional[_ToolCall]:
        return _find_tool(self.tools, id_, name)

    def flush_input_transcript(self) -> None:
        if not (text := "".join(self.input_transcript)):
            return
        from google.genai import types

        self.input_transcript.clear()
        self.history.append(types.Content(role="user", parts=[types.Part(text=text)]))
        self.user_texts.append(text)


class _LiveSessionTracer:
    """Turns the client and server messages of one Live session into spans."""

    def __init__(
        self,
        tracer: OITracer,
        trace_config: Optional[TraceConfig],
        model: Any,
        config: Any,
        parent_context: context_api.Context,
        provider_session_id: Optional[str],
        api_client: Any = None,
    ) -> None:
        self._tracer = tracer
        self._trace_config = trace_config
        self._parent_context = parent_context
        self._provider_session_id = provider_session_id
        # When audio or video input began. A span is only started once the input leads to
        # something (a server message or a failure), since a microphone may stream silence.
        self._input_started_at: Optional[int] = None
        # context attributes active when that input began; only valid together with it
        self._input_attributes: dict[str, AttributeValue] = {}
        # set while the current turn was started under suppress_tracing(), so that what
        # follows it is not traced whichever context it is sent or received in
        self.suppressed_turn = False
        self._api_client = api_client
        self.closing = False
        # set once the session is closed; anything sent or received later is not traced
        self.closed = False
        self._interaction: Optional[_Interaction] = None
        # names of NON_BLOCKING functions, which keep running after the interaction ends
        self._non_blocking: set[str] = set()
        self._background_tools: list[_ToolCall] = []
        # set by a client message that only answers calls left over from an earlier
        # interaction, so a failure to send it does not fail the current interaction
        self.detached_send = False
        self._request_extractor = _RequestAttributesExtractor()
        self._response_extractor = _ResponseAttributesExtractor(config=trace_config)
        self._system_contents: list[Any] = []
        self._llm_attributes: dict[str, AttributeValue] = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: LLM,
            SpanAttributes.LLM_PROVIDER: OpenInferenceLLMProviderValues.GOOGLE.value,
        }
        if isinstance(model, str) and model:
            self._llm_attributes[SpanAttributes.LLM_MODEL_NAME] = model
        try:
            self._read_connect_config(config)
        except Exception:
            logger.exception("Failed to read Live connect config")

    def _read_connect_config(self, config: Any) -> None:
        if config is None:
            return
        from google.genai import types
        from google.genai._transformers import t_contents

        if isinstance(config, Mapping):
            config = types.LiveConnectConfig.model_validate(config)
        if config.system_instruction:
            self._system_contents = list(t_contents(config.system_instruction))
        parameters = config.model_dump(
            mode="json", exclude_none=True, exclude=set(_NON_INVOCATION_CONFIG_FIELDS)
        )
        if parameters:
            self._llm_attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS] = safe_json_dumps(
                parameters
            )
        self._set_tools(config.tools or ())

    def _set_tools(self, tools: Iterable[Any]) -> None:
        tools = list(tools)
        if tools and self._api_client is not None:
            try:
                from google.genai._transformers import t_tool, t_tools

                # the same conversion the SDK applies before sending (e.g. Python functions)
                tools = [
                    t_tool(self._api_client, tool) for tool in t_tools(self._api_client, tools)
                ]
            except Exception:
                logger.debug("Failed to convert Live tools", exc_info=True)
        self._llm_attributes = {
            key: value
            for key, value in self._llm_attributes.items()
            if not key.startswith(f"{SpanAttributes.LLM_TOOLS}.")
        }
        self._llm_attributes.update(self._tool_attributes(tools))
        self._non_blocking = set()
        for tool in tools:
            for declaration in getattr(tool, "function_declarations", None) or ():
                behavior = getattr(declaration.behavior, "value", declaration.behavior)
                if behavior == "NON_BLOCKING" and declaration.name:
                    self._non_blocking.add(declaration.name)

    @staticmethod
    def _tool_attributes(tools: Iterable[Any]) -> Iterator[tuple[str, AttributeValue]]:
        index = 0
        for tool in tools:
            dump = getattr(tool, "model_dump", None)
            if not callable(dump):
                continue  # anything the SDK conversion did not turn into a Tool
            tool_dict = dump(mode="json", exclude_none=True)
            declarations = tool_dict.get("function_declarations")
            for schema in declarations if declarations else [tool_dict]:
                if "parameters" not in schema and "parameters_json_schema" in schema:
                    schema = {**schema, "parameters": schema["parameters_json_schema"]}
                    del schema["parameters_json_schema"]
                yield (
                    f"{SpanAttributes.LLM_TOOLS}.{index}.{ToolAttributes.TOOL_JSON_SCHEMA}",
                    safe_json_dumps(schema),
                )
                index += 1

    def _find_tool(
        self, id_: Optional[str], name: Optional[str], include_answered: bool = False
    ) -> Optional[_ToolCall]:
        if self._interaction is not None and (
            tool := _find_tool(self._interaction.tools, id_, name, include_answered)
        ):
            return tool
        return _find_tool(self._background_tools, id_, name, include_answered)

    def _remove_tool(self, tool: _ToolCall) -> None:
        for tools in (
            self._interaction.tools if self._interaction is not None else [],
            self._background_tools,
        ):
            if tool in tools:
                tools.remove(tool)

    # ----- span helpers -----

    def _start_span(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue],
        context: context_api.Context,
        parent: Optional[_WithSpan] = None,
        start_time: Optional[int] = None,
    ) -> _WithSpan:
        # The interaction's context is attached so that the span's parent and context
        # attributes such as session.id do not depend on which task sends or receives.
        # Suppression is the exception: it applies to wherever this message is handled.
        if _is_suppressed():
            context = context_api.set_value(
                context_api._SUPPRESS_INSTRUMENTATION_KEY, True, context
            )
        token = context_api.attach(context)
        span: trace_api.Span
        try:
            span = self._tracer.start_span(
                name=name,
                context=(
                    trace_api.set_span_in_context(parent._span, context)
                    if parent is not None
                    else context
                ),
                attributes=attributes,
                start_time=start_time,
            )
        except Exception:
            logger.exception("Failed to start span")
            span = trace_api.INVALID_SPAN
        finally:
            context_api.detach(token)
        return _WithSpan(span=span)

    def _ensure_interaction(self) -> _Interaction:
        if self._interaction is None:
            context = self._interaction_context()
            input_started_at, self._input_started_at = self._input_started_at, None
            self._interaction = _Interaction(
                self._start_span(
                    _CHAIN_SPAN_NAME,
                    {SpanAttributes.OPENINFERENCE_SPAN_KIND: CHAIN},
                    context,
                    start_time=input_started_at,
                ),
                context,
                input_started_at,
            )
        return self._interaction

    def _interaction_context(self) -> context_api.Context:
        # Context attributes active when the interaction starts apply on top of those from
        # connect(), so using_session() and the like can be used per interaction.
        context = self._parent_context
        attributes = dict(get_attributes_from_context())
        if self._input_started_at is not None:
            attributes.update(self._input_attributes)
        for key, value in attributes.items():
            context = context_api.set_value(key, value, context)
        if (
            self._provider_session_id
            and context_api.get_value(SpanAttributes.SESSION_ID, context) is None
        ):
            context = context_api.set_value(
                SpanAttributes.SESSION_ID, self._provider_session_id, context
            )
        return context

    def _ensure_generation(self) -> _Generation:
        interaction = self._ensure_interaction()
        if interaction.generation is None:
            start_time, interaction.input_started_at = interaction.input_started_at, None
            interaction.generation = _Generation(
                self._start_span(
                    _LLM_SPAN_NAME,
                    self._llm_attributes,
                    interaction.context,
                    parent=interaction.span,
                    start_time=start_time,
                ),
                self._system_contents,
            )
            interaction.generation.usage, interaction.pending_usage = (
                interaction.pending_usage,
                None,
            )
        return interaction.generation

    def _input_value(self, contents: Sequence[Any]) -> str:
        request: dict[str, Any] = {
            "contents": [content.model_dump(mode="json", exclude_none=True) for content in contents]
        }
        if self._trace_config is not None:
            request = redact_images_from_request_parameters(
                request,
                hide_input_images=bool(self._trace_config.hide_input_images),
                base64_image_max_length=int(self._trace_config.base64_image_max_length or 0),
            )
        return safe_json_dumps(request)

    def _input_message_attributes(
        self, system_contents: Sequence[Any], contents: Sequence[Any]
    ) -> Iterator[tuple[str, AttributeValue]]:
        index = 0
        for content in system_contents:
            for key, value in self._request_extractor._get_attributes_from_content(content):
                yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{key}", value
            yield (
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{MessageAttributes.MESSAGE_ROLE}",
                "system",
            )
            index += 1
        for content in contents:
            for message in self._request_extractor._iter_messages_from_content(content):
                for key, value in message:
                    yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{key}", value
                index += 1

    def _output_message_attributes(
        self, generation: _Generation, content: Any
    ) -> Iterator[tuple[str, AttributeValue]]:
        prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
        for key, value in self._response_extractor._get_attributes_from_generate_content_content(
            content
        ):
            yield f"{prefix}.{key}", value
        for index, call in enumerate(generation.function_calls):
            if call.id:
                yield (
                    f"{prefix}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                    f"{ToolCallAttributes.TOOL_CALL_ID}",
                    call.id,
                )

    def _end_generation(
        self,
        status: Optional[trace_api.Status],
        exception: Optional[BaseException] = None,
        await_usage: bool = False,
    ) -> None:
        interaction = self._interaction
        if interaction is None or (generation := interaction.generation) is None:
            return
        interaction.generation = None
        attributes: dict[str, AttributeValue] = {}
        content = None
        try:
            interaction.flush_input_transcript()
            content = generation.content()
            # nothing to record when the user spoke and no transcription is enabled
            if interaction.history:
                attributes[SpanAttributes.INPUT_VALUE] = self._input_value(interaction.history)
                attributes[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
            attributes.update(
                self._input_message_attributes(generation.system_contents, interaction.history)
            )
            if content.parts:
                attributes[SpanAttributes.OUTPUT_VALUE] = content.model_dump_json(exclude_none=True)
                attributes[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
                attributes.update(self._output_message_attributes(generation, content))
            if generation.reason:
                attributes[SpanAttributes.LLM_FINISH_REASON] = generation.reason
            if generation.usage is not None:
                attributes.update(_usage_attributes(generation.usage))
        except Exception:
            logger.exception("Failed to get Live generation attributes")
        if content is not None and content.parts:
            interaction.history.append(content)
            if text := generation.visible_text():
                interaction.last_output = text
        if exception is not None:
            generation.span.record_exception(exception)
        if await_usage:
            interaction.ended_generation = _EndedGeneration(
                generation.span, status, attributes, generation.completed_at
            )
        else:
            generation.span.finish_tracing(
                status=status, attributes=attributes, end_time=generation.completed_at
            )

    def _finish_ended_generation(self, interaction: _Interaction, usage: Any = None) -> None:
        if (ended := interaction.ended_generation) is None:
            return
        interaction.ended_generation = None
        if usage is not None:
            try:
                ended.attributes.update(_usage_attributes(usage))
            except Exception:
                logger.exception("Failed to get Live usage attributes")
        ended.span.finish_tracing(
            status=ended.status, attributes=ended.attributes, end_time=ended.end_time
        )

    def _end_interaction(
        self,
        status: Optional[trace_api.Status],
        exception: Optional[BaseException] = None,
    ) -> None:
        if (interaction := self._interaction) is None:
            return
        self._end_generation(status, exception)
        self._finish_ended_generation(interaction)
        for tool in interaction.tools:
            if exception is not None:
                tool.span.finish_tracing(status=status)
            elif status is not None and not tool.answered and tool.name in self._non_blocking:
                # a NON_BLOCKING function keeps running after the model finished its turn
                self._background_tools.append(tool)
            elif tool.answered and status is not None:
                # the server finished the interaction, so the response got through even
                # though its send() has not returned yet
                tool.span.finish_tracing(status=self._answered_status(tool))
            else:
                # no response yet (cancelled, or the session closed)
                tool.span.finish_tracing()
        interaction.tools.clear()
        self._interaction = None
        interaction.flush_input_transcript()
        attributes: dict[str, AttributeValue] = {}
        if user_text := "\n".join(interaction.user_texts):
            attributes[SpanAttributes.INPUT_VALUE] = user_text
            attributes[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.TEXT.value
        if interaction.last_output:
            attributes[SpanAttributes.OUTPUT_VALUE] = interaction.last_output
            attributes[SpanAttributes.OUTPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.TEXT.value
        if exception is not None:
            interaction.span.record_exception(exception)
        interaction.span.finish_tracing(status=status, attributes=attributes)

    # ----- client messages -----

    # Each returns an optional callback, called once the SDK call has finished (with the
    # exception if it failed), for spans that should only end after the message is sent.

    def on_client_content(self, kwargs: Mapping[str, Any]) -> Optional[_AfterSend]:
        from google.genai._transformers import t_contents

        self.suppressed_turn = False
        self._input_started_at = None
        interaction = self._ensure_interaction()
        interaction.flush_input_transcript()
        if turns := kwargs.get("turns"):
            for content in t_contents(turns):
                interaction.history.append(_without_media(content))
                if text := "".join(part.text for part in content.parts or () if part.text):
                    interaction.user_texts.append(text)
        if kwargs.get("turn_complete", True):
            self._ensure_generation()
        return None

    def on_realtime_input(self, kwargs: Mapping[str, Any]) -> Optional[_AfterSend]:
        # Audio, video and activity signals are not recorded, only when they began. An
        # explicit activity_start is exact; otherwise it is the first chunk after the
        # previous interaction, which includes any silence a microphone streams before
        # the user speaks.
        if kwargs.get("activity_start") is not None:
            self.suppressed_turn = False
            self._mark_input_started()
            return self._fail_input_if_send_fails
        if any(kwargs.get(key) is not None for key in ("media", "audio", "video")):
            return self._on_media_input()
        if not (text := kwargs.get("text")):
            return None
        from google.genai import types

        self.suppressed_turn = False
        self._input_started_at = None
        interaction = self._ensure_interaction()
        interaction.flush_input_transcript()
        interaction.history.append(types.Content(role="user", parts=[types.Part(text=text)]))
        interaction.user_texts.append(text)
        self._ensure_generation()
        return None

    def _on_media_input(self) -> _AfterSend:
        self.suppressed_turn = False
        if self._interaction is None and self._input_started_at is None:
            self._mark_input_started()
        return self._fail_input_if_send_fails

    def _mark_input_started(self) -> None:
        self._input_started_at = time.time_ns()
        self._input_attributes = dict(get_attributes_from_context())

    def _fail_input_if_send_fails(self, exception: Optional[BaseException]) -> None:
        # A failure while an interaction is open is handled by the send wrapper.
        if (
            exception is None
            or isinstance(exception, asyncio.CancelledError)
            or self._interaction is not None
            or self._input_started_at is None
        ):
            return
        self._ensure_interaction()
        self._end_interaction(_error_status(exception), exception)

    def on_suppressed_send(self, session: Any, kwargs: Mapping[str, Any]) -> None:
        # What the server replies with is suppressed too. Spans that already started are
        # not affected, and a SILENT tool response gets no reply.
        if self._interaction is not None:
            return
        responses = kwargs.get("function_responses")
        if "input" in kwargs:  # the deprecated send()
            try:
                message = session._parse_client_message(kwargs["input"], kwargs.get("end_of_turn"))
            except Exception:
                return  # invalid input; send() raises the same error
            responses = (message.get("tool_response") or {}).get("function_responses")
        if responses is not None:
            if not isinstance(responses, Sequence) or isinstance(responses, Mapping):
                responses = [responses]
            if all(_is_silent(response) for response in responses):
                return
        self._input_started_at = None
        self.suppressed_turn = True

    def on_context_update(self, kwargs: Mapping[str, Any]) -> None:
        # only the fields that are set are updated; an empty tool list clears the tools
        if (system_instruction := kwargs.get("system_instruction")) is not None:
            from google.genai._transformers import t_contents

            self._system_contents = list(t_contents(system_instruction))
        if (tools := kwargs.get("tools")) is not None:
            self._set_tools(tools)

    def on_tool_response(self, kwargs: Mapping[str, Any]) -> Optional[_AfterSend]:
        from google.genai import types

        responses = kwargs.get("function_responses")
        if responses is None or self.suppressed_turn:
            return None
        if not isinstance(responses, Sequence) or isinstance(responses, Mapping):
            responses = [responses]
        responses = [
            _response_without_media(
                response
                if isinstance(response, types.FunctionResponse)
                else types.FunctionResponse.model_validate(response)
            )
            for response in responses
        ]
        answered: list[_ToolCall] = []
        # silent responses to NON_BLOCKING calls that outlived their interaction
        detached = bool(responses)
        for response in responses:
            tool = self._find_tool(response.id, response.name)
            if tool is None or tool not in self._background_tools or not _is_silent(response):
                detached = False
            if tool is not None:
                if (output := self._tool_output_value(response)) is not None:
                    tool.span.set_attributes(
                        {
                            SpanAttributes.OUTPUT_VALUE: output,
                            SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                        }
                    )
                # a response with will_continue is partial; the call stays open
                if not getattr(response, "will_continue", None):
                    tool.answered = True
                    tool.error = _tool_error(response)
                    answered.append(tool)
        if detached:
            self.detached_send = True
        else:
            if self._interaction is None:
                self._input_started_at = None
            interaction = self._ensure_interaction()
            interaction.history.append(
                types.Content(
                    role="user",
                    parts=[types.Part(function_response=response) for response in responses],
                )
            )
            if any(
                response.name not in self._non_blocking or not _is_silent(response)
                for response in responses
            ):
                self._ensure_generation()

        def end_tool_spans(exception: Optional[BaseException]) -> None:
            for tool in answered:
                # the interaction may have ended (and ended the span) in the meantime
                self._remove_tool(tool)
                if exception is None:
                    tool.span.finish_tracing(status=self._answered_status(tool))
                elif isinstance(exception, asyncio.CancelledError):
                    tool.span.finish_tracing()
                else:
                    tool.span.record_exception(exception)
                    tool.span.finish_tracing(status=_error_status(exception))

        return end_tool_spans

    def _answered_status(self, tool: _ToolCall) -> trace_api.Status:
        if tool.error is None:
            return _OK
        # the error is part of the tool's output, so it is hidden along with it
        hidden = self._trace_config is not None and self._trace_config.hide_outputs
        return trace_api.Status(
            status_code=trace_api.StatusCode.ERROR,
            description=(
                None
                if hidden
                else tool.error
                if isinstance(tool.error, str)
                else safe_json_dumps(tool.error)
            ),
        )

    def _tool_output_value(self, response: Any) -> Optional[str]:
        if not response.parts:
            return None if response.response is None else safe_json_dumps(response.response)
        # multimodal results; audio and video were already removed
        value: dict[str, Any] = {
            "parts": [part.model_dump(mode="json", exclude_none=True) for part in response.parts]
        }
        if response.response is not None:
            value = {"response": response.response, **value}
        if self._trace_config is not None:
            value, _ = redact_images_from_output_value(
                value, base64_image_max_length=int(self._trace_config.base64_image_max_length or 0)
            )
        return safe_json_dumps(value)

    def on_send(self, session: Any, kwargs: Mapping[str, Any]) -> Optional[_AfterSend]:
        # The deprecated send() takes many input shapes; the SDK's own parser sorts
        # them into the same three kinds of client message.
        try:
            message = session._parse_client_message(kwargs.get("input"), kwargs.get("end_of_turn"))
        except Exception:
            return None  # invalid input; send() raises the same error
        if (content := message.get("client_content")) is not None:
            return self.on_client_content(
                {"turns": content.get("turns"), "turn_complete": content.get("turn_complete")}
            )
        if (tool_response := message.get("tool_response")) is not None:
            return self.on_tool_response(
                {"function_responses": tool_response.get("function_responses")}
            )
        if message.get("realtime_input") is not None:
            return self._on_media_input()
        return None

    # ----- server messages -----

    def on_server_message(self, message: Any, suppressed: bool = False) -> None:
        server_content = getattr(message, "server_content", None)
        tool_call = getattr(message, "tool_call", None)
        cancellation = getattr(message, "tool_call_cancellation", None)
        usage = getattr(message, "usage_metadata", None)
        # Only a new turn is suppressed here; one that already started needs its messages
        # to finish its spans. Audio input that no span exists for yet counts as new.
        if (
            suppressed
            and self._interaction is None
            and (server_content is not None or tool_call is not None)
        ):
            self._input_started_at = None
            self.suppressed_turn = True
        if self.suppressed_turn:
            if server_content is not None and _is_interaction_complete(server_content):
                self.suppressed_turn = False
            server_content = tool_call = usage = None
        if server_content is None and tool_call is None and cancellation is None and not usage:
            return  # setup, resumption, go-away and voice-activity messages
        if (
            usage
            and self._interaction is None
            and self._input_started_at is not None
            and getattr(server_content, "turn_complete", None)
        ):
            # the model spent tokens on the input without any output, e.g. it was interrupted
            self._ensure_generation()
        if server_content is not None:
            self._on_server_content(server_content)
        if tool_call is not None and tool_call.function_calls:
            self._on_tool_call(tool_call.function_calls)
        if cancellation is not None and cancellation.ids:
            for id_ in cancellation.ids:
                # include calls whose response is still being sent
                if tool := self._find_tool(id_, None, include_answered=True):
                    self._remove_tool(tool)
                    tool.span.add_event("cancelled")
                    tool.span.finish_tracing()
        if usage and (interaction := self._interaction) is not None:
            # Usage normally arrives with turn_complete. After a tool call it belongs to
            # the generation that made the call, even if the client has already answered,
            # until the model continues. If there is no generation at all, keep it for the
            # next one instead of starting an empty LLM span.
            if interaction.ended_generation is not None and (
                server_content is None or interaction.generation is None
            ):
                self._finish_ended_generation(interaction, usage)
            elif interaction.generation is not None:
                interaction.generation.usage = usage
            else:
                interaction.pending_usage = usage
        if server_content is not None:
            if getattr(server_content, "turn_complete", None):
                self._end_generation(_OK)
            if _is_interaction_complete(server_content):
                self._end_interaction(_OK)

    def _on_server_content(self, server_content: Any) -> None:
        if (transcription := server_content.input_transcription) and transcription.text:
            self._ensure_interaction().input_transcript.append(transcription.text)
        model_turn = server_content.model_turn
        if model_turn is not None and model_turn.parts:
            # any output starts a generation, including audio that is not recorded
            generation = self._ensure_generation()
            self._finish_ended_generation(self._ensure_interaction())
            for part in model_turn.parts:
                if part.text:
                    (generation.thoughts if part.thought else generation.texts).append(part.text)
        if (transcription := server_content.output_transcription) and transcription.text:
            self._ensure_generation().transcript.append(transcription.text)
            self._finish_ended_generation(self._ensure_interaction())
        if reason := getattr(server_content, "turn_complete_reason", None):
            if self._interaction is not None and self._interaction.generation is not None:
                self._interaction.generation.reason = getattr(reason, "value", str(reason))
        if getattr(server_content, "generation_complete", None):
            if self._interaction is not None and (open_generation := self._interaction.generation):
                open_generation.completed_at = open_generation.completed_at or time.time_ns()

    @staticmethod
    def _streamed_call_index(calls: Sequence[Any], frame: Any) -> Optional[int]:
        # IDs are optional; without one, a frame continues the call being streamed.
        for index in range(len(calls) - 1, -1, -1):
            call = calls[index]
            if frame.id:
                if call.id == frame.id:
                    return index
            elif call.will_continue and (not frame.name or call.name == frame.name):
                return index
        return None

    def _on_tool_call(self, function_calls: Sequence[Any]) -> None:
        new_calls = []
        continued_calls = []
        open_generation = self._interaction.generation if self._interaction is not None else None
        calls = open_generation.function_calls if open_generation is not None else []
        for frame in function_calls:
            index = self._streamed_call_index(calls, frame)
            stored = calls[index] if index is not None else None
            call = _merge_function_call(stored, frame)
            tool = (
                self._find_tool(call.id, None if call.id else call.name)
                if frame.id or stored is not None
                else None
            )
            if tool is None:
                new_calls.append(call)
                continue
            if index is not None:
                calls[index] = call
                continued_calls.append(call)
            if call.args is not None:
                tool.span.set_attributes(
                    {
                        SpanAttributes.INPUT_VALUE: safe_json_dumps(call.args),
                        SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                    }
                )
        if not new_calls and not continued_calls:
            return
        generation = self._ensure_generation()
        if new_calls:
            self._finish_ended_generation(self._ensure_interaction())
            generation.function_calls.extend(new_calls)
        # A blocking call ends the generation; the model continues in a new one once
        # the client sends the tool response. With only NON_BLOCKING calls the model
        # keeps generating. A call still being streamed keeps the generation open too.
        if not any(call.will_continue for call in generation.function_calls) and any(
            call.name not in self._non_blocking for call in (*new_calls, *continued_calls)
        ):
            self._end_generation(_OK, await_usage=True)
        interaction = self._ensure_interaction()
        for call in new_calls:
            attributes: dict[str, AttributeValue] = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: TOOL,
            }
            if call.name:
                attributes[SpanAttributes.TOOL_NAME] = call.name
            if call.id:
                attributes[SpanAttributes.TOOL_ID] = call.id
            if call.args is not None:
                attributes[SpanAttributes.INPUT_VALUE] = safe_json_dumps(call.args)
                attributes[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
            interaction.tools.append(
                _ToolCall(
                    call.id,
                    call.name,
                    self._start_span(
                        call.name or "tool",
                        attributes,
                        interaction.context,
                        parent=interaction.span,
                    ),
                )
            )

    # ----- session lifecycle -----

    @property
    def interaction(self) -> Optional[_Interaction]:
        return self._interaction

    def fail(self, exception: BaseException) -> None:
        if self._input_started_at is not None:
            self._ensure_interaction()
        self._end_interaction(_error_status(exception), exception)
        self._end_background_tools(exception)

    def fail_interaction(
        self, exception: BaseException, interaction: Optional[_Interaction]
    ) -> None:
        # A send can fail after the server already finished its interaction; a newer
        # interaction is not affected.
        if interaction is not None and interaction is self._interaction:
            self._end_interaction(_error_status(exception), exception)

    def end(self) -> None:
        # the session closed in the middle of an interaction; leave the status unset
        self._end_interaction(None)
        self._end_background_tools(None)

    def _end_background_tools(self, exception: Optional[BaseException]) -> None:
        for tool in self._background_tools:
            if exception is not None:
                tool.span.record_exception(exception)
                tool.span.finish_tracing(status=_error_status(exception))
            else:
                tool.span.finish_tracing()
        self._background_tools.clear()


class _TracedConnect(ObjectProxy):  # type: ignore[misc,type-arg,unused-ignore]
    """Async context manager returned by the wrapped ``AsyncLive.connect``."""

    def __init__(
        self,
        wrapped: Any,
        tracer: OITracer,
        trace_config: Optional[TraceConfig],
        model: Any,
        config: Any,
        api_client: Any,
    ) -> None:
        super().__init__(wrapped)
        self._self_tracer = tracer
        self._self_trace_config = trace_config
        self._self_model = model
        self._self_config = config
        self._self_api_client = api_client
        self._self_session: Any = None

    async def __aenter__(self) -> Any:
        resolved: list[Any] = []
        token = _resolved_connect_config.set(resolved)
        try:
            session = await self.__wrapped__.__aenter__()
        finally:
            _resolved_connect_config.reset(token)
        try:
            _sessions[session] = _LiveSessionTracer(
                tracer=self._self_tracer,
                trace_config=self._self_trace_config,
                model=self._self_model,
                config=resolved[0] if resolved else self._self_config,
                parent_context=context_api.get_current(),
                provider_session_id=getattr(session, "session_id", None),
                api_client=self._self_api_client,
            )
            self._self_session = session
        except Exception:
            logger.exception("Failed to start tracing the Live session")
        return session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        session, self._self_session = self._self_session, None
        state = _sessions.pop(session, None) if session is not None else None
        if exc is not None:
            _finish_session(state, exc)
        try:
            return await _close_session(state, self.__wrapped__.__aexit__(exc_type, exc, tb))
        finally:
            if state is not None:
                state.closed = True


async def _close_session(state: Optional[_LiveSessionTracer], close: Any) -> Any:
    """Awaits the SDK's close and then ends whatever is still open.

    While closing, a receive that fails because the socket closed is not an error.
    """
    if state is not None:
        state.closing = True
    try:
        result = await close
    except BaseException as exception:
        if state is not None:
            # the session may still be usable after a failed close
            state.closing = False
        _finish_session(state, exception)
        raise
    _finish_session(state, None)
    if state is not None:
        state.closed = True
    return result


def _finish_session(
    state: Optional[_LiveSessionTracer], exception: Optional[BaseException]
) -> None:
    if state is None:
        return
    try:
        if exception is not None and not isinstance(exception, asyncio.CancelledError):
            state.fail(exception)
        else:
            state.end()
    except Exception:
        logger.exception("Failed to finish tracing the Live session")


class _AsyncLiveConnectWrapper:
    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer
        config = getattr(tracer, "_self_config", None)
        self._config = config if isinstance(config, TraceConfig) else None

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        connect = wrapped(*args, **kwargs)
        if _is_suppressed():
            return connect
        return _TracedConnect(
            connect,
            tracer=self._tracer,
            trace_config=self._config,
            model=kwargs.get("model"),
            config=kwargs.get("config"),
            api_client=getattr(instance, "_api_client", None),
        )


class _ConnectConfigWrapper:
    """Hands the connect config resolved by the SDK to the connect() being traced."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        config = await wrapped(*args, **kwargs)
        if (resolved := _resolved_connect_config.get()) is not None:
            resolved.append(config)
        return config


class _AsyncSessionSendWrapper:
    """Records what the client sends before handing it to the session."""

    def __init__(self, record: "_Record") -> None:
        self._record = record

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if (state := _sessions.get(instance)) is None or state.closed:
            return await wrapped(*args, **kwargs)
        if _is_suppressed():
            state.on_suppressed_send(instance, kwargs)
            try:
                return await wrapped(*args, **kwargs)
            except BaseException:
                state.suppressed_turn = False  # no reply is coming
                raise
        # Recorded before sending: the reply may be received by another task before
        # this call returns.
        after_send: Optional[_AfterSend] = None
        state.detached_send = False
        try:
            after_send = self._record(state, instance, kwargs)
        except Exception:
            logger.exception("Failed to record Live client message")
        interaction = None if state.detached_send else state.interaction
        try:
            result = await wrapped(*args, **kwargs)
        except BaseException as exception:
            _call_after_send(after_send, exception)
            if not isinstance(exception, asyncio.CancelledError):
                state.fail_interaction(exception, interaction)
            raise
        _call_after_send(after_send, None)
        return result


class _AsyncSessionContextUpdateWrapper:
    """Applies a change of system instruction or tools to the spans that follow it."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        result = await wrapped(*args, **kwargs)
        # A suppressed update must not show up on later spans, so they keep the previous one.
        if (
            (state := _sessions.get(instance)) is not None
            and not state.closed
            and not _is_suppressed()
        ):
            try:
                state.on_context_update(kwargs)
            except Exception:
                logger.exception("Failed to record Live context update")
        return result


def _call_after_send(after_send: Optional[_AfterSend], exception: Optional[BaseException]) -> None:
    if after_send is None:
        return
    try:
        after_send(exception)
    except Exception:
        logger.exception("Failed to finish tracing a Live client message")


class _AsyncSessionReceiveWrapper:
    """Observes every server message, whether read through receive() or start_stream()."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if (state := _sessions.get(instance)) is None:
            return await wrapped(*args, **kwargs)
        try:
            message = await wrapped(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exception:
            # while closing, the close itself ends the spans once it finishes
            if not state.closing and not state.closed:
                state.fail(exception)
            raise
        # a receive that was already waiting can return after the session closed
        if not state.closed:
            try:
                state.on_server_message(message, suppressed=_is_suppressed())
            except Exception:
                logger.exception("Failed to record Live server message")
        return message


class _AsyncSessionCloseWrapper:
    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        return await _close_session(_sessions.get(instance), wrapped(*args, **kwargs))


_Record = Callable[[_LiveSessionTracer, Any, Mapping[str, Any]], Optional[_AfterSend]]

SEND_WRAPPERS: Mapping[str, _Record] = {
    "send_client_content": lambda state, session, kwargs: state.on_client_content(kwargs),
    "send_realtime_input": lambda state, session, kwargs: state.on_realtime_input(kwargs),
    "send_tool_response": lambda state, session, kwargs: state.on_tool_response(kwargs),
    # deprecated, but still public
    "send": _LiveSessionTracer.on_send,
}
