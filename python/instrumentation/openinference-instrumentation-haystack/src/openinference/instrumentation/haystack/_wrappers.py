import json
from abc import ABC
from enum import Enum, auto
from inspect import BoundArguments, signature
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

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
from typing_extensions import TypeGuard, assert_never

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.core.component import Component
from haystack.dataclasses import ChatMessage


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
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(LLM)),
                        **dict(_get_llm_input_message_attributes(run_args)),
                    }
                )
            elif component_type is ComponentType.EMBEDDER:
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(EMBEDDING)),
                        **dict(_get_embedding_model_attributes(component.model)),
                    }
                )
            elif component_type is ComponentType.RETRIEVER:
                span.set_attributes(dict(_get_span_kind_attributes(RETRIEVER)))
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
            span.set_attributes(dict(_get_component_output_attributes(response, component_type)))
            span.set_status(trace_api.StatusCode.OK)
            if component_type is ComponentType.GENERATOR:
                span.set_attributes(
                    {
                        **dict(_get_llm_model_attributes(response)),
                        **dict(_get_llm_output_message_attributes(response)),
                        **dict(_get_llm_token_count_attributes(response)),
                    }
                )
            elif component_type is ComponentType.EMBEDDER:
                span.set_attributes(dict(_get_embedding_attributes(run_args, response)))
            elif component_type is ComponentType.RETRIEVER:
                span.set_attributes(dict(_get_retriever_response_attributes(response)))
            elif component_type is ComponentType.PROMPT_BUILDER:
                pass
            elif component_type is ComponentType.UNKNOWN:
                pass
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
    outputs_list_of_documents = (
        output_socket := output_types.get("documents")
    ) is not None and output_socket.type is List[Document]
    if has_string_query_parameter and outputs_list_of_documents:
        return True
    return False


def _get_span_kind_attributes(span_kind: str) -> Iterator[Tuple[str, Any]]:
    """
    Yields span kind attributes.
    """
    yield OPENINFERENCE_SPAN_KIND, span_kind


def _get_input_attributes(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Yields input attributes.
    """
    masked_arguments = dict(_mask_embedding_vectors(key, value) for key, value in arguments.items())
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(masked_arguments)


def _get_component_output_attributes(
    response: Mapping[str, Any], component_type: ComponentType
) -> Iterator[Tuple[str, Any]]:
    """
    Yields output attributes.
    """
    if component_type is ComponentType.PROMPT_BUILDER:
        yield from _get_output_attributes_for_prompt_builder(response)
    else:
        yield from _get_output_attributes(response)


def _get_output_attributes(response: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Yields output attributes.
    """
    yield OUTPUT_MIME_TYPE, JSON
    yield OUTPUT_VALUE, safe_json_dumps(response)


def _get_llm_input_message_attributes(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts input messages.
    """
    if isinstance(messages := arguments.get("messages"), Sequence) and all(
        map(lambda x: isinstance(x, ChatMessage), messages)
    ):
        for message_index, message in enumerate(messages):
            if (content := message.content) is not None:
                yield f"{LLM_INPUT_MESSAGES}.{message_index}.{MESSAGE_CONTENT}", content
            if (role := message.role) is not None:
                yield f"{LLM_INPUT_MESSAGES}.{message_index}.{MESSAGE_ROLE}", role
            if (name := message.name) is not None:
                yield f"{LLM_INPUT_MESSAGES}.{message_index}.{MESSAGE_NAME}", name
    elif isinstance(prompt := arguments.get("prompt"), str):
        yield f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}", prompt
        yield f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}", USER


def _get_llm_output_message_attributes(response: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts output messages.
    """

    if not isinstance(replies := response.get("replies"), Sequence):
        return
    for reply_index, reply in enumerate(replies):
        if isinstance(reply, ChatMessage):
            if (
                (reply_meta := getattr(reply, "meta", None)) is None
                or not isinstance(reply_meta, dict)
                or (finish_reason := reply_meta.get("finish_reason")) is None
            ):
                continue
            if finish_reason == "tool_calls":
                try:
                    tool_calls = json.loads(reply.content)
                except json.JSONDecodeError:
                    continue
                for tool_call_index, tool_call in enumerate(tool_calls):
                    if (function_call := tool_call.get("function")) is None:
                        continue
                    if (tool_call_arguments_json := function_call.get("arguments")) is not None:
                        yield (
                            f"{LLM_OUTPUT_MESSAGES}.{reply_index}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            tool_call_arguments_json,
                        )
                    if (tool_name := function_call.get("name")) is not None:
                        yield (
                            f"{LLM_OUTPUT_MESSAGES}.{reply_index}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_NAME}",
                            tool_name,
                        )
            else:
                yield f"{LLM_OUTPUT_MESSAGES}.{reply_index}.{MESSAGE_CONTENT}", reply.content
            yield f"{LLM_OUTPUT_MESSAGES}.{reply_index}.{MESSAGE_ROLE}", reply.role.value
        elif isinstance(reply, str):
            yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}", reply
            yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", ASSISTANT


def _get_llm_model_attributes(response: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts LLM model attributes from response.
    """
    if (
        isinstance(response_meta := response.get("meta"), Sequence)
        and response_meta
        and (model := response_meta[0].get("model")) is not None
    ):
        yield LLM_MODEL_NAME, model
    elif (
        isinstance(replies := response.get("replies"), Sequence)
        and replies
        and isinstance(reply := replies[0], ChatMessage)
        and (model := reply.meta.get("model")) is not None
    ):
        yield LLM_MODEL_NAME, model


def _get_llm_token_count_attributes(response: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts token counts from response.
    """
    token_usage = None
    if (
        isinstance(response_meta := response.get("meta"), Sequence)
        and response_meta
        and isinstance(response_meta[0], dict)
        and isinstance(usage := response_meta[0].get("usage"), dict)
    ):
        token_usage = usage
    elif (
        isinstance(replies := response.get("replies"), Sequence)
        and replies
        and isinstance(reply := replies[0], ChatMessage)
        and isinstance(usage := reply.meta.get("usage"), dict)
    ):
        token_usage = usage
    if token_usage is not None:
        if (completion_tokens := token_usage.get("completion_tokens")) is not None:
            yield LLM_TOKEN_COUNT_COMPLETION, completion_tokens
        if (prompt_tokens := token_usage.get("prompt_tokens")) is not None:
            yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens
        if (total_tokens := token_usage.get("total_tokens")) is not None:
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


def _get_output_attributes_for_prompt_builder(
    response: Mapping[str, Any],
) -> Iterator[Tuple[str, Any]]:
    """
    Yields output attributes for prompt builder.
    """
    if isinstance(prompt := response.get("prompt"), str):
        yield OUTPUT_MIME_TYPE, TEXT
        yield OUTPUT_VALUE, prompt
    else:
        yield from _get_output_attributes(response)


def _get_retriever_response_attributes(response: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts retriever-related attributes from the response.
    """
    if (
        (documents := response.get("documents")) is None
        or not isinstance(documents, Sequence)
        or not all(map(lambda x: isinstance(x, Document), documents))
    ):
        return
    for doc_index, doc in enumerate(documents):
        if (content := doc.content) is not None:
            yield f"{RETRIEVAL_DOCUMENTS}.{doc_index}." f"{DOCUMENT_CONTENT}", content
        if (id := doc.id) is not None:
            yield f"{RETRIEVAL_DOCUMENTS}.{doc_index}." f"{DOCUMENT_ID}", id
        if (score := doc.score) is not None:
            yield f"{RETRIEVAL_DOCUMENTS}.{doc_index}." f"{DOCUMENT_SCORE}", score
        if (metadata := doc.meta) is not None:
            yield (
                f"{RETRIEVAL_DOCUMENTS}.{doc_index}." f"{DOCUMENT_METADATA}",
                safe_json_dumps(metadata),
            )


def _get_embedding_model_attributes(model: Any) -> Iterator[Tuple[str, Any]]:
    """
    Yields attributes for embedding model.
    """
    if isinstance(model, str):
        yield EMBEDDING_MODEL_NAME, model


def _get_embedding_attributes(
    arguments: Mapping[str, Any], response: Mapping[str, Any]
) -> Iterator[Tuple[str, Any]]:
    """
    Extracts embedding attributes from an embedder response.
    """
    if isinstance(documents := response.get("documents"), Sequence) and all(
        map(_is_embedding_doc, documents)
    ):
        for doc_index, doc in enumerate(documents):
            yield f"{EMBEDDING_EMBEDDINGS}.{doc_index}.{EMBEDDING_TEXT}", doc.content
            yield (
                f"{EMBEDDING_EMBEDDINGS}.{doc_index}.{EMBEDDING_VECTOR}",
                list(doc.embedding),
            )
    elif _is_vector(embedding := response.get("embedding")) and isinstance(
        text := arguments.get("text"), str
    ):
        yield f"{EMBEDDING_EMBEDDINGS}.0.{EMBEDDING_TEXT}", text
        yield (
            f"{EMBEDDING_EMBEDDINGS}.0.{EMBEDDING_VECTOR}",
            list(embedding),
        )


def _is_embedding_doc(maybe_doc: Any) -> bool:
    """
    Returns true if the input is a `haystack.Document` with embedding
    attributes.
    """
    return (
        isinstance(maybe_doc, Document)
        and isinstance(maybe_doc.content, str)
        and _is_vector(maybe_doc.embedding)
    )


def _mask_embedding_vectors(key: str, value: Any) -> Tuple[str, Any]:
    """
    Masks embeddings.
    """
    if isinstance(key, str) and "embedding" in key and _is_vector(value):
        return key, f"<{len(value)}-dimensional vector>"
    return key, value


def _is_vector(
    value: Any,
) -> TypeGuard[Sequence[Union[int, float]]]:
    """
    Checks for sequences of numbers.
    """

    is_sequence_of_numbers = isinstance(value, Sequence) and all(
        map(lambda x: isinstance(x, (int, float)), value)
    )
    return is_sequence_of_numbers


CHAIN = OpenInferenceSpanKindValues.CHAIN.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
LLM = OpenInferenceSpanKindValues.LLM.value
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER.value

JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

ASSISTANT = "assistant"
USER = "user"

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
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
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
