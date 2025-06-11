import inspect
import json
import logging
from abc import ABC
from contextlib import ExitStack
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    OrderedDict,
    TypedDict,
    TypeVar,
)

import wrapt
from google.adk import Runner
from google.adk.agents import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.telemetry import _build_llm_request_for_trace
from google.adk.tools import BaseTool
from google.genai import types
from google.genai.types import MediaModality
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import StatusCode, get_current_span
from opentelemetry.util.types import AttributeValue
from typing_extensions import NotRequired, ParamSpec

from openinference.instrumentation import (
    get_attributes_from_context,
    safe_json_dumps,
    using_session,
    using_user,
)
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

P = ParamSpec("P")
T = TypeVar("T")


class _WithTracer(ABC):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer


class _RunnerRunAsyncKwargs(TypedDict):
    user_id: str
    session_id: str
    new_message: types.Content
    run_config: NotRequired[RunConfig]


class _RunnerRunAsync(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[Event, None]],
        instance: Runner,
        args: tuple[Any, ...],
        kwargs: _RunnerRunAsyncKwargs,
    ) -> Any:
        generator = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return generator

        tracer = self._tracer
        name = f"invocation [{instance.app_name}]"
        attributes = dict(get_attributes_from_context())
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.CHAIN.value

        arguments = bind_args_kwargs(wrapped, *args, **kwargs)
        try:
            attributes[SpanAttributes.INPUT_VALUE] = json.dumps(
                arguments,
                default=_default,
                ensure_ascii=False,
            )
            attributes[SpanAttributes.INPUT_MIME_TYPE] = OpenInferenceMimeTypeValues.JSON.value
        except Exception:
            logger.exception(f"Failed to get attribute: {SpanAttributes.INPUT_VALUE}.")

        if (user_id := kwargs.get("user_id")) is not None:
            attributes[SpanAttributes.USER_ID] = user_id
        if (session_id := kwargs.get("session_id")) is not None:
            attributes[SpanAttributes.SESSION_ID] = session_id

        class _AsyncGenerator(wrapt.ObjectProxy):  # type: ignore[misc]
            __wrapped__: AsyncGenerator[Event, None]

            async def __aiter__(self) -> Any:
                with ExitStack() as stack:
                    span = stack.enter_context(
                        tracer.start_as_current_span(
                            name=name,
                            attributes=attributes,
                        )
                    )
                    if user_id is not None:
                        stack.enter_context(using_user(user_id))
                    if session_id is not None:
                        stack.enter_context(using_session(session_id))
                    async for event in self.__wrapped__:
                        if event.is_final_response():
                            try:
                                span.set_attribute(
                                    SpanAttributes.OUTPUT_VALUE,
                                    event.model_dump_json(exclude_none=True),
                                )
                                span.set_attribute(
                                    SpanAttributes.OUTPUT_MIME_TYPE,
                                    OpenInferenceMimeTypeValues.JSON.value,
                                )
                            except Exception:
                                logger.exception(
                                    f"Failed to get attribute: {SpanAttributes.OUTPUT_VALUE}."
                                )
                        yield event
                    span.set_status(StatusCode.OK)

        return _AsyncGenerator(generator)


class _BaseAgentRunAsync(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., AsyncGenerator[Event, None]],
        instance: BaseAgent,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        generator = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return generator

        tracer = self._tracer
        name = f"agent_run [{instance.name}]"
        attributes = dict(get_attributes_from_context())
        attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.AGENT.value

        class _AsyncGenerator(wrapt.ObjectProxy):  # type: ignore[misc]
            __wrapped__: AsyncGenerator[Event, None]

            async def __aiter__(self) -> Any:
                with tracer.start_as_current_span(
                    name=name,
                    attributes=attributes,
                ) as span:
                    async for event in self.__wrapped__:
                        if event.is_final_response():
                            try:
                                span.set_attribute(
                                    SpanAttributes.OUTPUT_VALUE,
                                    event.model_dump_json(exclude_none=True),
                                )
                                span.set_attribute(
                                    SpanAttributes.OUTPUT_MIME_TYPE,
                                    OpenInferenceMimeTypeValues.JSON.value,
                                )
                            except Exception:
                                logger.exception(
                                    f"Failed to get attribute: {SpanAttributes.OUTPUT_VALUE}."
                                )
                        yield event
                    span.set_status(StatusCode.OK)

        return _AsyncGenerator(generator)


class _TraceCallLlm(_WithTracer):
    @wrapt.decorator  # type: ignore[misc]
    def __call__(
        self,
        wrapped: Callable[..., T],
        _: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> T:
        ans = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return ans
        span = get_current_span()
        span.set_status(StatusCode.OK)  # Pre-emptively set status to OK
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        arguments = bind_args_kwargs(wrapped, *args, **kwargs)
        llm_request = next((arg for arg in arguments.values() if isinstance(arg, LlmRequest)), None)
        llm_response = next(
            (arg for arg in arguments.values() if isinstance(arg, LlmResponse)), None
        )
        input_messages_index = 0
        if llm_request:
            span.set_attribute(
                SpanAttributes.LLM_PROVIDER,
                OpenInferenceLLMProviderValues.GOOGLE.value,
            )  # TODO: other providers may also be possible

            try:
                span.set_attribute(
                    SpanAttributes.INPUT_VALUE,
                    safe_json_dumps(_build_llm_request_for_trace(llm_request)),
                )
                span.set_attribute(
                    SpanAttributes.INPUT_MIME_TYPE,
                    OpenInferenceMimeTypeValues.JSON.value,
                )
            except Exception:
                logger.exception(f"Failed to get attribute: {SpanAttributes.INPUT_VALUE}.")

            if llm_request.tools_dict:
                for i, tool in enumerate(llm_request.tools_dict.values()):
                    for k, v in _get_attributes_from_base_tool(
                        tool,
                        prefix=f"{SpanAttributes.LLM_TOOLS}.{i}.",
                    ):
                        span.set_attribute(k, v)

            if llm_request.model:
                span.set_attribute(SpanAttributes.LLM_MODEL_NAME, llm_request.model)

            if config := llm_request.config:
                for k, v in _get_attributes_from_generate_content_config(config):
                    span.set_attribute(k, v)

                if system_instruction := config.system_instruction:
                    span.set_attribute(
                        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_ROLE}",
                        "system",
                    )
                    if isinstance(system_instruction, str):
                        span.set_attribute(
                            f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_CONTENT}",
                            system_instruction,
                        )
                    elif isinstance(system_instruction, types.Content):
                        if system_instruction.parts:
                            for k, v in _get_attributes_from_parts(
                                system_instruction.parts,
                                prefix=f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.",
                                text_only=True,
                            ):
                                span.set_attribute(k, v)
                    elif isinstance(system_instruction, list):
                        # TODO
                        pass
                    input_messages_index += 1

            if contents := llm_request.contents:
                for i, content in enumerate(contents, input_messages_index):
                    for k, v in _get_attributes_from_content(
                        content,
                        prefix=f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}.",
                    ):
                        span.set_attribute(k, v)
        if llm_response:
            for k, v in _get_attributes_from_llm_response(llm_response):
                span.set_attribute(k, v)
        return ans


class _TraceToolCall(_WithTracer):
    @wrapt.decorator  # type: ignore[misc]
    def __call__(
        self,
        wrapped: Callable[..., T],
        _: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> T:
        ans = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return ans
        span = get_current_span()
        span.set_status(StatusCode.OK)  # Pre-emptively set status to OK
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.TOOL.value,
        )
        arguments = bind_args_kwargs(wrapped, *args, **kwargs)
        if base_tool := next(
            (arg for arg in arguments.values() if isinstance(arg, BaseTool)), None
        ):
            span.set_attribute(SpanAttributes.TOOL_NAME, base_tool.name)
            span.set_attribute(SpanAttributes.TOOL_DESCRIPTION, base_tool.description)
            if args_dict := next(
                (arg for arg in arguments.values() if isinstance(arg, Mapping)), None
            ):
                try:
                    span.set_attribute(
                        SpanAttributes.TOOL_PARAMETERS,
                        safe_json_dumps(args_dict),
                    )
                    span.set_attribute(
                        SpanAttributes.INPUT_VALUE,
                        safe_json_dumps(args_dict),
                    )
                    span.set_attribute(
                        SpanAttributes.INPUT_MIME_TYPE,
                        OpenInferenceMimeTypeValues.JSON.value,
                    )
                except Exception:
                    logger.exception(f"Failed to get attribute: {SpanAttributes.INPUT_VALUE}.")
        if event := next((arg for arg in arguments.values() if isinstance(arg, Event)), None):
            if responses := event.get_function_responses():
                try:
                    span.set_attribute(
                        SpanAttributes.OUTPUT_VALUE,
                        responses[0].model_dump_json(exclude_none=True),
                    )
                    span.set_attribute(
                        SpanAttributes.OUTPUT_MIME_TYPE,
                        OpenInferenceMimeTypeValues.JSON.value,
                    )
                except Exception:
                    logger.exception(f"Failed to get attribute in {wrapped.__name__}.")
        return ans


def stop_on_exception(
    wrapped: Callable[P, Iterator[tuple[str, AttributeValue]]],
) -> Callable[P, Iterator[tuple[str, AttributeValue]]]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[tuple[str, AttributeValue]]:
        try:
            yield from wrapped(*args, **kwargs)
        except Exception:
            logger.exception(f"Failed to get attribute in {wrapped.__name__}.")

    return wrapper


@stop_on_exception
def _get_attributes_from_generate_content_config(
    obj: types.GenerateContentConfig,
) -> Iterator[tuple[str, AttributeValue]]:
    yield SpanAttributes.LLM_INVOCATION_PARAMETERS, obj.model_dump_json(exclude_none=True)


@stop_on_exception
def _get_attributes_from_llm_response(
    obj: LlmResponse,
) -> Iterator[tuple[str, AttributeValue]]:
    yield SpanAttributes.OUTPUT_VALUE, obj.model_dump_json(exclude_none=True)
    yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
    if obj.usage_metadata:
        yield from _get_attributes_from_usage_metadata(obj.usage_metadata)
    if obj.content:
        yield from _get_attributes_from_content(
            obj.content, prefix=f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0."
        )


@stop_on_exception
def _get_attributes_from_usage_metadata(
    obj: types.GenerateContentResponseUsageMetadata,
) -> Iterator[tuple[str, AttributeValue]]:
    if obj.candidates_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, obj.candidates_token_count
    if obj.candidates_tokens_details:
        completion_details_audio = 0
        for modality_token_count in obj.candidates_tokens_details:
            if (
                modality_token_count.modality is MediaModality.AUDIO
                and modality_token_count.token_count
            ):
                completion_details_audio += modality_token_count.token_count
        if completion_details_audio:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO, completion_details_audio
    if obj.prompt_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, obj.prompt_token_count
    if obj.total_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, obj.total_token_count


@stop_on_exception
def _get_attributes_from_content(
    obj: types.Content,
    /,
    *,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    role = obj.role or "user"
    yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", role
    if parts := obj.parts:
        yield from _get_attributes_from_parts(parts, prefix=prefix)


@stop_on_exception
def _get_attributes_from_parts(
    obj: Iterable[types.Part],
    /,
    *,
    prefix: str = "",
    text_only: bool = False,
) -> Iterator[tuple[str, AttributeValue]]:
    for i, part in enumerate(obj):
        if (text := part.text) is not None:
            yield from _get_attributes_from_text_part(
                text,
                prefix=f"{prefix}{MessageAttributes.MESSAGE_CONTENTS}.{i}.",
            )
        elif text_only:
            continue
        elif (function_call := part.function_call) is not None:
            yield from _get_attributes_from_function_call(
                function_call,
                prefix=f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALLS}.{i}.",
            )
        elif (function_response := part.function_response) is not None:
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "tool"
            if function_response.name:
                yield f"{prefix}{MessageAttributes.MESSAGE_NAME}", function_response.name
            if function_response.response:
                yield (
                    f"{prefix}{MessageAttributes.MESSAGE_CONTENT}",
                    safe_json_dumps(function_response.response),
                )


@stop_on_exception
def _get_attributes_from_text_part(
    obj: str,
    /,
    *,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", obj
    yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"


@stop_on_exception
def _get_attributes_from_function_call(
    obj: types.FunctionCall,
    /,
    *,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    if id_ := obj.id:
        yield f"{prefix}{ToolCallAttributes.TOOL_CALL_ID}", id_
    if name := obj.name:
        yield f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", name
    if function_arguments := obj.args:
        yield (
            f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
            safe_json_dumps(function_arguments),
        )


@stop_on_exception
def _get_attributes_from_function_response(
    obj: types.FunctionResponse,
    /,
    *,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    if id_ := obj.id:
        yield f"{prefix}{ToolCallAttributes.TOOL_CALL_ID}", id_
    if name := obj.name:
        yield f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", name
    if response := obj.response:
        yield (
            f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
            safe_json_dumps(response),
        )


@stop_on_exception
def _get_attributes_from_base_tool(
    obj: BaseTool,
    /,
    *,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    if declaration := obj._get_declaration():
        tool_json_schema = declaration.model_dump_json(exclude_none=True)
    else:
        tool_json_schema = json.dumps({"name": obj.name, "description": obj.description})
    yield f"{prefix}{ToolAttributes.TOOL_JSON_SCHEMA}", tool_json_schema


def bind_args_kwargs(func: Any, *args: Any, **kwargs: Any) -> OrderedDict[str, Any]:
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _default(obj: Any) -> str:
    from pydantic import BaseModel

    if isinstance(obj, BaseModel):
        return json.dumps(
            obj.model_dump(exclude=None),
            ensure_ascii=False,
            default=str,
        )
    else:
        return str(obj)
