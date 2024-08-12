from abc import ABC
from enum import Enum, auto
from inspect import BoundArguments, signature
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

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
from typing_extensions import assert_never

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.core.component import Component
from haystack.dataclasses import ChatRole


class _WithTracer(ABC):
    """
    Base class for wrappers that need a tracer.
    """

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer


class _ComponentWrapper(_WithTracer):
    """
    Components in Haystack are defined as a duck-typed protocol with a `run`
    method, so we wrap `haystack.Pipeline._run_component`, which invokes the
    component's `run` method. From here, we can gain access to the component
    itself.

    See:
    https://github.com/deepset-ai/haystack/blob/21c507331c98c76aed88cd8046373dfa2a3590e7/haystack/core/component/component.py#L129
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

        pipe_args = signature(wrapped).bind(*args, **kwargs).arguments
        component = _get_component_by_name(instance, pipe_args["name"])
        if component is None or not hasattr(component, "run") or not callable(component.run):
            return wrapped(*args, **kwargs)
        component_class_name = _get_component_class_name(component)

        run_bound_args = signature(component.run).bind(**pipe_args["inputs"])
        run_args = run_bound_args.arguments

        with self._tracer.start_as_current_span(name=component_class_name) as span:
            span.set_attributes(
                {**dict(get_attributes_from_context()), **dict(_get_input_attributes(run_args))}
            )
            if (component_type := _get_component_type(component)) is ComponentType.GENERATOR:
                span.set_attributes(dict(_get_span_kind_attributes(LLM)))
                if "Chat" in component_class_name:
                    for i, msg in enumerate(run_args["messages"]):
                        span.set_attributes(
                            {
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}": msg.content,
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}": msg.role,
                            }
                        )
                else:
                    span.set_attributes(
                        {
                            f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}": run_args["prompt"],
                            f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}": ChatRole.USER,
                        }
                    )

            elif component_type is ComponentType.EMBEDDER:
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(EMBEDDING)),
                        EMBEDDING_MODEL_NAME: component.model,
                    }
                )
            elif component_type is ComponentType.RETRIEVER:
                span.set_attributes(dict(_get_span_kind_attributes(RETRIEVER)))
                # todo: implement retriever input attributes
            elif component_type is ComponentType.PROMPT_BUILDER:
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(LLM)),
                        **dict(
                            _get_llm_prompt_template_attributes_from_prompt_builder(
                                component, run_bound_args
                            )
                        ),
                    }
                )
            elif component_type is ComponentType.UNKNOWN:
                span.set_attributes(dict(_get_span_kind_attributes(CHAIN)))
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
                if "Chat" in component_class_name:
                    replies = response.get("replies")
                    if replies is None or len(replies) == 0:
                        pass
                    reply = replies[0]
                    usage = reply.meta.get("usage", {})
                    if "meta" in str(reply):
                        span.set_attributes(
                            {
                                **dict(_get_llm_token_count_attributes(usage)),
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
                            **dict(_get_llm_token_count_attributes(response["meta"][0]["usage"])),
                            LLM_MODEL_NAME: response["meta"][0]["model"],
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
                        f"{emb_vec_0}{EMBEDDING_TEXT}": run_args["text"],
                    }
                )
            elif component_type is ComponentType.RETRIEVER:
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
            elif (
                component_type is ComponentType.UNKNOWN
                or component_type is ComponentType.PROMPT_BUILDER
            ):
                span.set_attributes(dict(_get_output_attributes(response)))
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

        arguments = signature(wrapped).bind(*args, **kwargs).arguments

        span_name = "Pipeline"
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                **dict(get_attributes_from_context()),
                **dict(_get_span_kind_attributes(CHAIN)),
                **dict(_get_input_attributes(arguments)),
            },
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(dict(_get_output_attributes(response)))
            span.set_status(trace_api.StatusCode.OK)

        return response


class ComponentType(Enum):
    GENERATOR = auto()
    EMBEDDER = auto()
    RETRIEVER = auto()
    PROMPT_BUILDER = auto()
    UNKNOWN = auto()


def _get_component_by_name(pipeline: Pipeline, component_name: str) -> Optional[Component]:
    """
    Gets the component invoked by `haystack.Pipeline._run_component` (if one exists).
    """
    if (node := pipeline.graph.nodes.get(component_name)) is None or (
        component := node.get("instance")
    ) is None:
        return None
    return component


def _get_component_class_name(component: Component) -> str:
    """
    Gets the name of the component.
    """
    return str(component.__class__.__name__)


def _get_component_type(component: Component) -> ComponentType:
    """
    Haystack has a single `Component` interface that produces unstructured
    outputs. In the absence of typing information, we make a best-effort attempt
    to infer the component type.
    """
    component_name = _get_component_class_name(component)
    if (run_method := getattr(component, "run", None)) is None or not callable(run_method):
        return ComponentType.UNKNOWN
    if "Generator" in component_name or "VertexAIImage" in component_name:
        return ComponentType.GENERATOR
    elif "Embedder" in component_name:
        return ComponentType.EMBEDDER
    elif "Retriever" in component_name or _has_retriever_run_method(run_method):
        return ComponentType.RETRIEVER
    elif isinstance(component, PromptBuilder):
        return ComponentType.PROMPT_BUILDER
    return ComponentType.UNKNOWN


def _has_retriever_run_method(run_method: Callable[..., Any]) -> bool:
    """
    Uses heuristics to infer if a component has a retriever-like `run` method.

    This is used to find unusual retrievers such as `SerperDevWebSearch`. See:
    https://github.com/deepset-ai/haystack/blob/21c507331c98c76aed88cd8046373dfa2a3590e7/haystack/components/websearch/serper_dev.py#L93
    """

    # Find types defined with the `output_types` decorator. See
    # https://github.com/deepset-ai/haystack/blob/21c507331c98c76aed88cd8046373dfa2a3590e7/haystack/core/component/component.py#L398
    output_types = (
        ot if isinstance((ot := getattr(run_method, "_output_types_cache", None)), dict) else {}
    )
    run_method_signature = signature(run_method)
    has_string_query_parameter = (
        query_parameter := run_method_signature.parameters.get("query")
    ) is not None and query_parameter.annotation is str
    outputs_list_of_documents = output_types.get("documents") is List[Document]
    if has_string_query_parameter and outputs_list_of_documents:
        return True
    return False


def _get_span_kind_attributes(span_kind: str) -> Iterator[Tuple[str, Any]]:
    """
    Yields span kind attributes.
    """
    yield OPENINFERENCE_SPAN_KIND, span_kind


def _get_input_attributes(arguments: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Yields input attributes.
    """
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _get_output_attributes(response: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Yields output attributes.
    """
    yield OUTPUT_MIME_TYPE, JSON
    yield OUTPUT_VALUE, safe_json_dumps(response)


def _get_llm_token_count_attributes(usage: Any) -> Iterator[Tuple[str, Any]]:
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


def _get_llm_prompt_template_attributes_from_prompt_builder(
    component: Component, run_bound_args: BoundArguments
) -> Iterator[Tuple[str, str]]:
    """
    Extracts prompt template attributes from a prompt builder component.

    This duplicates logic from `PromptBuilder.run`. See:
    https://github.com/deepset-ai/haystack/blob/21c507331c98c76aed88cd8046373dfa2a3590e7/haystack/components/builders/prompt_builder.py#L194
    """
    template = (
        t
        if (t := run_bound_args.arguments.get("template")) is not None
        else getattr(component, "_template_string", None)
    )
    if template is not None:
        yield LLM_PROMPT_TEMPLATE, template
    if (
        template_variables := {
            **run_bound_args.kwargs,
            **(
                tvs
                if isinstance(tvs := run_bound_args.arguments.get("template_variables"), dict)
                else {}
            ),
        }
    ) is not None:
        yield LLM_PROMPT_TEMPLATE_VARIABLES, safe_json_dumps(template_variables)


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
