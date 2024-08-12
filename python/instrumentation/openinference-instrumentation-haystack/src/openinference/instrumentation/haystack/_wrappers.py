from abc import ABC
from enum import Enum, auto
from inspect import BoundArguments, signature
from typing import Any, Callable, Iterator, List, Mapping, Optional, Tuple

import opentelemetry.context as context_api
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
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
from opentelemetry.util.types import AttributeValue
from typing_extensions import assert_never

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.core.component import Component
from haystack.dataclasses import ChatRole


class ComponentType(Enum):
    GENERATOR = auto()
    EMBEDDER = auto()
    RETRIEVER = auto()
    PROMPT_BUILDER = auto()
    UNKNOWN = auto()


def get_component_type(component_name: str) -> ComponentType:
    if "Generator" in component_name or "VertexAIImage" in component_name:
        return ComponentType.GENERATOR
    elif "Embedder" in component_name:
        return ComponentType.EMBEDDER
    elif "Retriever" in component_name:
        return ComponentType.RETRIEVER
    elif "PromptBuilder" in component_name:
        return ComponentType.PROMPT_BUILDER
    return ComponentType.UNKNOWN


class _WithTracer(ABC):
    """
    Base class for wrappers that need a tracer.
    """

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer


class _ComponentWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _get_bound_arguments(wrapped, *args, **kwargs).arguments
        component_type = arguments["name"]
        input_data = arguments["inputs"]

        if (component := _get_component_by_name(instance, component_type)) is None:
            return wrapped(*args, **kwargs)

        component_name = component.__class__.__name__

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        with self._tracer.start_as_current_span(name=component_name) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            if (component_type := get_component_type(component_name)) is ComponentType.GENERATOR:
                span.set_attributes(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM,
                        INPUT_VALUE: safe_json_dumps(input_data),
                        INPUT_MIME_TYPE: JSON,
                    }
                )
                if "Chat" in component_name:
                    for i, msg in enumerate(input_data["messages"]):
                        span.set_attributes(
                            {
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}": msg.content,
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}": msg.role,
                            }
                        )
                else:
                    span.set_attributes(
                        {
                            f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}": input_data["prompt"],
                            f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}": ChatRole.USER,
                        }
                    )

                if "prompt_builder" in str(instance):
                    if "ChatPromptBuilder" in str(instance):
                        span.set_attributes(
                            {
                                LLM_INPUT_MESSAGES: input_data["messages"],
                            }
                        )
                    else:
                        span.set_attribute(
                            LLM_PROMPT_TEMPLATE,
                            instance.graph.nodes._nodes["prompt_builder"][
                                "instance"
                            ]._template_string,
                        )

            elif component_type is ComponentType.EMBEDDER:
                span.set_attributes(
                    {
                        EMBEDDING_MODEL_NAME: component.model,
                        OPENINFERENCE_SPAN_KIND: EMBEDDING,
                        INPUT_VALUE: safe_json_dumps(invocation_parameters),
                        INPUT_MIME_TYPE: JSON,
                    }
                )
            elif component_type is ComponentType.RETRIEVER:
                span.set_attributes(
                    {
                        OPENINFERENCE_SPAN_KIND: RETRIEVER,
                    }
                )
                if "query_embedding" in input_data:
                    emb_len = len(input_data["query_embedding"])
                    span.set_attributes(
                        {
                            INPUT_MIME_TYPE: TEXT,
                            INPUT_VALUE: f"<{emb_len} dimensional vector>",
                        }
                    )
                elif "query" in input_data:
                    span.set_attributes(
                        {
                            INPUT_MIME_TYPE: TEXT,
                            INPUT_VALUE: input_data["query"],
                        }
                    )
            elif component_type is ComponentType.PROMPT_BUILDER:
                span.set_attributes(
                    {
                        OPENINFERENCE_SPAN_KIND: LLM,
                        INPUT_VALUE: safe_json_dumps(invocation_parameters),
                        INPUT_MIME_TYPE: JSON,
                    }
                )
                if "ChatPromptBuilder" in component_name:
                    temp_vars = safe_json_dumps(input_data["template_variables"])
                    msg_conts = [m.content for m in input_data["template"]]
                    span.set_attributes(
                        {
                            LLM_INPUT_MESSAGES: msg_conts,
                            LLM_PROMPT_TEMPLATE_VARIABLES: temp_vars,
                        }
                    )
                else:
                    span.set_attributes(
                        dict(_get_llm_prompt_template_attributes(component, arguments)),
                    )
                if "documents" in invocation_parameters:
                    span.set_attributes(
                        {
                            RETRIEVAL_DOCUMENTS: safe_json_dumps(
                                invocation_parameters["documents"]
                            ),
                        }
                    )
            elif component_type is ComponentType.UNKNOWN:
                span.set_attributes(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        INPUT_VALUE: safe_json_dumps(invocation_parameters),
                        INPUT_MIME_TYPE: JSON,
                    }
                )
            else:
                assert_never(component_type)

            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)

            if component_type is ComponentType.GENERATOR:
                if "Chat" in component.__class__.__name__:
                    replies = response.get("replies")
                    if replies is None or len(replies) == 0:
                        pass
                    reply = replies[0]
                    usage = reply.meta.get("usage", {})
                    if "meta" in str(reply):
                        span.set_attributes(
                            {
                                **dict(_get_token_counts(usage)),
                                OUTPUT_VALUE: safe_json_dumps(response),
                                OUTPUT_MIME_TYPE: JSON,
                                LLM_MODEL_NAME: reply.meta["model"],
                            }
                        )
                    for i, reply in enumerate(response["replies"]):
                        span.set_attributes(
                            {
                                f"{LLM_OUTPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}": reply.content,
                                f"{LLM_OUTPUT_MESSAGES}.{i}.{MESSAGE_ROLE}": reply.role,
                            }
                        )
                else:
                    span.set_attributes(
                        {
                            **dict(_get_token_counts(response["meta"][0]["usage"])),
                            LLM_MODEL_NAME: response["meta"][0]["model"],
                            OUTPUT_VALUE: safe_json_dumps(response["replies"]),
                            OUTPUT_MIME_TYPE: JSON,
                        }
                    )
                    span.set_attributes(
                        {
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}": response["replies"][0],
                            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": ChatRole.ASSISTANT,
                        }
                    )
            elif component_type is ComponentType.EMBEDDER:
                emb_len = len(response["embedding"])
                emb_vec_0 = f"{EMBEDDING_EMBEDDINGS}.0."

                span.set_attributes(
                    {
                        f"{emb_vec_0}{EMBEDDING_VECTOR}": f"<{emb_len} dimensional vector>",
                        f"{emb_vec_0}{EMBEDDING_TEXT}": invocation_parameters["text"],
                    }
                )
            elif component_type is ComponentType.RETRIEVER:
                if "documents" in response:
                    span.set_attributes(
                        {
                            OUTPUT_VALUE: safe_json_dumps(response["documents"]),
                            OUTPUT_MIME_TYPE: JSON,
                        }
                    )

                for i, document in enumerate(response["documents"]):
                    span.set_attributes(
                        {
                            f"{RETRIEVAL_DOCUMENTS}.{i}." f"{DOCUMENT_CONTENT}": document.content,
                            f"{RETRIEVAL_DOCUMENTS}.{i}." f"{DOCUMENT_ID}": document.id,
                            f"{RETRIEVAL_DOCUMENTS}.{i}." f"{DOCUMENT_SCORE}": document.score,
                            f"{RETRIEVAL_DOCUMENTS}.{i}." f"{DOCUMENT_METADATA}": safe_json_dumps(
                                document.meta
                            ),
                        }
                    )
            elif component_type in (ComponentType.UNKNOWN, ComponentType.PROMPT_BUILDER):
                span.set_attributes(
                    {
                        OUTPUT_VALUE: safe_json_dumps(response),
                        OUTPUT_MIME_TYPE: JSON,
                    }
                )
            else:
                assert_never(component_type)

        return response


class _PipelineWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        span_name = "Pipeline"
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: CHAIN,
                INPUT_VALUE: safe_json_dumps(invocation_parameters),
                INPUT_MIME_TYPE: JSON,
            },
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                {
                    OUTPUT_VALUE: safe_json_dumps(response),
                    OUTPUT_MIME_TYPE: JSON,
                }
            )
            span.set_status(trace_api.StatusCode.OK)

        return response


def _get_token_counts(usage: Any) -> Iterator[Tuple[str, Any]]:
    """
    Extract token counts from the usage.
    """
    if not isinstance(usage, dict):
        return
    if (completion_tokens := usage.get("completion_tokens")) is not None:
        yield LLM_TOKEN_COUNT_COMPLETION, completion_tokens
    if (prompt_tokens := usage.get("prompt_tokens")) is not None:
        yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if (total_tokens := usage.get("total_tokens")) is not None:
        yield LLM_TOKEN_COUNT_TOTAL, total_tokens


def _get_llm_prompt_template_attributes(
    component: Component, arguments: Any
) -> Iterator[Tuple[str, str]]:
    """
    Extracts prompt template attributes from a prompt builder component.

    This duplicates logic from `PromptBuilder.run`. See:
    https://github.com/deepset-ai/haystack/blob/21c507331c98c76aed88cd8046373dfa2a3590e7/haystack/components/builders/prompt_builder.py#L194
    """
    bound_arguments = _get_bound_arguments(
        PromptBuilder.run,
        component,
        **(inputs if isinstance((inputs := arguments.get("inputs")), dict) else {}),
    )
    template = (
        t
        if (t := bound_arguments.arguments.get("template")) is not None
        else getattr(component, "_template_string", None)
    )
    if template is not None:
        yield LLM_PROMPT_TEMPLATE, template
    if (
        template_variables := {
            **bound_arguments.kwargs,
            **(
                tvs
                if isinstance(tvs := bound_arguments.arguments.get("template_variables"), dict)
                else {}
            ),
        }
    ) is not None:
        yield LLM_PROMPT_TEMPLATE_VARIABLES, safe_json_dumps(template_variables)


def _get_component_by_name(pipeline: Pipeline, component_name: str) -> Optional[Component]:
    """
    Gets the component invoked by `Pipeline._run_component` (if one exists).
    """
    if (node := pipeline.graph.nodes.get(component_name)) is None or (
        component := node.get("instance")
    ) is None:
        return None
    return component


def _get_bound_arguments(function: Callable[..., Any], *args: Any, **kwargs: Any) -> BoundArguments:
    """
    Binds arguments and keyword arguments to a function.
    """
    return signature(function).bind(*args, **kwargs)


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


CHAIN = OpenInferenceSpanKindValues.CHAIN.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
LLM = OpenInferenceSpanKindValues.LLM.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value

JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
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
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
USER_ID = SpanAttributes.USER_ID
