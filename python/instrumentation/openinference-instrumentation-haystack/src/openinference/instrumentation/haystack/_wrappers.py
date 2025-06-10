from enum import Enum, auto
from inspect import BoundArguments, Parameter, signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    get_type_hints,
)

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from typing_extensions import TypeGuard, assert_never

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

if TYPE_CHECKING:
    from haystack import Document, Pipeline
    from haystack.core.component import Component


class _PipelineRunComponentWrapper:
    """
    Components in Haystack are defined as a duck-typed protocol with a `run`
    method invoked by `haystack.Pipeline._run_component`. We dynamically wrap
    the component `run` method if it is not already wrapped from within
    `haystack.Pipeline._run_component`.

    This wrapper handles the static method signature of Pipeline._run_component:
    @staticmethod
    def _run_component(
        component_name: str,
        component: Dict[str, Any],
        inputs: Dict[str, Any],
        component_visits: Dict[str, int],
        parent_span: Optional[tracing.Span] = None,
    ) -> Dict[str, Any]:
    """

    def __init__(
        self,
        tracer: trace_api.Tracer,
        wrap_component_run_method: Callable[[type[Any], Callable[..., Any]], None],
    ) -> None:
        self._tracer = tracer
        self._wrap_component_run_method = wrap_component_run_method

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        arguments = _get_bound_arguments(wrapped, *args, **kwargs).arguments
        if (component := arguments.get("component")) is not None and (
            component_instance := component.get("instance")
        ) is not None:
            component_cls = component_instance.__class__
            self._wrap_component_run_method(component_cls, component_instance.run)
        return wrapped(*args, **kwargs)


class _ComponentRunWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        component = instance
        component_class_name = _get_component_class_name(component)
        bound_arguments = _get_bound_arguments(wrapped, *args, **kwargs)
        arguments = bound_arguments.arguments

        with self._tracer.start_as_current_span(
            name=_get_component_span_name(component_class_name)
        ) as span:
            span.set_attributes(
                {**dict(get_attributes_from_context()), **dict(_get_input_attributes(arguments))}
            )
            if (component_type := _get_component_type(component)) is ComponentType.GENERATOR:
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(LLM)),
                        **dict(_get_llm_input_message_attributes(arguments)),
                    }
                )
            elif component_type is ComponentType.EMBEDDER:
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(EMBEDDING)),
                        **dict(_get_embedding_model_attributes(component)),
                    }
                )
            elif component_type is ComponentType.RANKER:
                span.set_attributes(
                    {
                        **dict(_get_span_kind_attributes(RERANKER)),
                        **dict(_get_reranker_model_attributes(component)),
                        **dict(_get_reranker_request_attributes(arguments)),
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
                                component, bound_arguments
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
                span.set_attributes(dict(_get_embedding_attributes(arguments, response)))
            elif component_type is ComponentType.RANKER:
                span.set_attributes(dict(_get_reranker_response_attributes(response)))
            elif component_type is ComponentType.RETRIEVER:
                span.set_attributes(dict(_get_retriever_response_attributes(response)))
            elif component_type is ComponentType.PROMPT_BUILDER:
                pass
            elif component_type is ComponentType.UNKNOWN:
                pass
            else:
                assert_never(component_type)

        return response


class _PipelineWrapper:
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
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
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _get_bound_arguments(wrapped, *args, **kwargs).arguments

        span_name = "Pipeline.run"
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
                raise
            span.set_attributes(dict(_get_output_attributes(response)))
            span.set_status(trace_api.StatusCode.OK)

        return response


class ComponentType(Enum):
    GENERATOR = auto()
    EMBEDDER = auto()
    RANKER = auto()
    RETRIEVER = auto()
    PROMPT_BUILDER = auto()
    UNKNOWN = auto()


def _get_component_by_name(pipeline: "Pipeline", component_name: str) -> Optional["Component"]:
    """
    Gets the component invoked by `haystack.Pipeline._run_component` (if one exists).
    """
    if (node := pipeline.graph.nodes.get(component_name)) is None or (
        component := node.get("instance")
    ) is None:
        return None
    return cast(Optional["Component"], component)


def _get_component_class_name(component: "Component") -> str:
    """
    Gets the name of the component.
    """
    return str(component.__class__.__name__)


def _get_component_span_name(component_class_name: str) -> str:
    """
    Gets the name of the span for a component.
    """
    return f"{component_class_name}.run"


def _get_component_type(component: "Component") -> ComponentType:
    """
    Haystack has a single `Component` interface that produces unstructured
    outputs. In the absence of typing information, we make a best-effort attempt
    to infer the component type.
    """
    from haystack.components.builders import PromptBuilder

    component_name = _get_component_class_name(component)
    if (run_method := _get_component_run_method(component)) is None:
        return ComponentType.UNKNOWN
    if "Generator" in component_name or _has_generator_output_type(run_method):
        return ComponentType.GENERATOR
    elif "Embedder" in component_name:
        return ComponentType.EMBEDDER
    elif "Ranker" in component_name and _has_ranker_io_types(run_method):
        return ComponentType.RANKER
    elif (
        "Retriever" in component_name or "WebSearch" in component_name
    ) and _has_retriever_io_types(run_method):
        return ComponentType.RETRIEVER
    elif isinstance(component, PromptBuilder):
        return ComponentType.PROMPT_BUILDER
    return ComponentType.UNKNOWN


def _get_component_run_method(component: "Component") -> Optional[Callable[..., Any]]:
    """
    Gets the `run` method for a component (if one exists).
    """
    if callable(run_method := getattr(component, "run", None)):
        return cast(Callable[..., Any], run_method)
    return None


def _get_run_method_output_types(run_method: Callable[..., Any]) -> Optional[Dict[str, type]]:
    """
    Haystack components are decorated with an `output_type` decorator that is
    useful for inferring the component type.

    https://github.com/deepset-ai/haystack/blob/21c507331c98c76aed88cd8046373dfa2a3590e7/haystack/core/component/component.py#L398
    """

    if isinstance((output_types_cache := getattr(run_method, "_output_types_cache", None)), dict):
        return {key: value.type for key, value in output_types_cache.items()}
    return None


def _get_run_method_input_types(run_method: Callable[..., Any]) -> Optional[Dict[str, type]]:
    """
    Gets input types of parameters to the `run` method.
    """
    return get_type_hints(run_method)


def _has_generator_output_type(run_method: Callable[..., Any]) -> bool:
    """
    Uses heuristics to infer if a component has a generator-like `run` method.
    """
    from haystack.dataclasses import ChatMessage

    if (output_types := _get_run_method_output_types(run_method)) is None or (
        replies := output_types.get("replies")
    ) is None:
        return False
    return replies == List[ChatMessage] or replies == List[str]


def _has_ranker_io_types(run_method: Callable[..., Any]) -> bool:
    """
    Uses heuristics to infer if a component has a ranker-like `run` method.
    """
    from haystack import Document

    if (input_types := _get_run_method_input_types(run_method)) is None or (
        output_types := _get_run_method_output_types(run_method)
    ) is None:
        return False
    has_documents_parameter = input_types.get("documents") == List[Document]
    outputs_list_of_documents = output_types.get("documents") == List[Document]
    return has_documents_parameter and outputs_list_of_documents


def _has_retriever_io_types(run_method: Callable[..., Any]) -> bool:
    """
    Uses heuristics to infer if a component has a retriever-like `run` method.

    This is used to find unusual retrievers such as `SerperDevWebSearch`. See:
    https://github.com/deepset-ai/haystack/blob/21c507331c98c76aed88cd8046373dfa2a3590e7/haystack/components/websearch/serper_dev.py#L93
    """
    from haystack import Document

    if (input_types := _get_run_method_input_types(run_method)) is None or (
        output_types := _get_run_method_output_types(run_method)
    ) is None:
        return False
    has_documents_parameter = "documents" in input_types
    outputs_list_of_documents = output_types.get("documents") == List[Document]
    return not has_documents_parameter and outputs_list_of_documents


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
    masked_response = dict(_mask_embedding_vectors(key, value) for key, value in response.items())
    yield OUTPUT_MIME_TYPE, JSON
    yield OUTPUT_VALUE, safe_json_dumps(masked_response)


def _get_llm_input_message_attributes(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts input messages.
    """
    from haystack.dataclasses import ChatMessage

    if isinstance(messages := arguments.get("messages"), Sequence) and all(
        map(lambda x: isinstance(x, ChatMessage), messages)
    ):
        for message_index, message in enumerate(messages):
            if (content := message.text) is not None:
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
    from haystack.dataclasses import ChatMessage

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
                tool_calls = reply.tool_calls
                for tool_call_index, tool_call in enumerate(tool_calls):
                    if (tool_call_arguments := tool_call.arguments) is not None:
                        yield (
                            f"{LLM_OUTPUT_MESSAGES}.{reply_index}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                            safe_json_dumps(tool_call_arguments),
                        )
                    if (tool_name := tool_call.tool_name) is not None:
                        yield (
                            f"{LLM_OUTPUT_MESSAGES}.{reply_index}.{MESSAGE_TOOL_CALLS}.{tool_call_index}.{TOOL_CALL_FUNCTION_NAME}",
                            tool_name,
                        )
            else:
                yield f"{LLM_OUTPUT_MESSAGES}.{reply_index}.{MESSAGE_CONTENT}", reply.text
            yield f"{LLM_OUTPUT_MESSAGES}.{reply_index}.{MESSAGE_ROLE}", reply.role.value
        elif isinstance(reply, str):
            yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}", reply
            yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", ASSISTANT


def _get_llm_model_attributes(response: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts LLM model attributes from response.
    """
    from haystack.dataclasses import ChatMessage

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
    from haystack.dataclasses import ChatMessage

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
    component: "Component", run_bound_args: BoundArguments
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


def _get_reranker_model_attributes(component: "Component") -> Iterator[Tuple[str, Any]]:
    """
    A best-effort attempt to get the model name from a ranker component.
    """
    if isinstance(
        model_name := (getattr(component, "model_name", None) or getattr(component, "model", None)),
        str,
    ):
        yield RERANKER_MODEL_NAME, model_name


def _get_reranker_request_attributes(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts re-ranker attributes from arguments.
    """
    if isinstance(query := arguments.get("query"), str):
        yield RERANKER_QUERY, query
    if isinstance(top_k := arguments.get("top_k"), int):
        yield RERANKER_TOP_K, top_k
    if _is_list_of_documents(documents := arguments.get("documents")):
        for doc_index, doc in enumerate(documents):
            if (id := doc.id) is not None:
                yield f"{RERANKER_INPUT_DOCUMENTS}.{doc_index}.{DOCUMENT_ID}", id
            if (content := doc.content) is not None:
                yield f"{RERANKER_INPUT_DOCUMENTS}.{doc_index}.{DOCUMENT_CONTENT}", content


def _get_reranker_response_attributes(response: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts re-ranker attributes from response.
    """
    if _is_list_of_documents(documents := response.get("documents")):
        for doc_index, doc in enumerate(documents):
            if (id := doc.id) is not None:
                yield f"{RERANKER_OUTPUT_DOCUMENTS}.{doc_index}.{DOCUMENT_ID}", id
            if (content := doc.content) is not None:
                yield f"{RERANKER_OUTPUT_DOCUMENTS}.{doc_index}.{DOCUMENT_CONTENT}", content
            if (score := doc.score) is not None:
                yield f"{RERANKER_OUTPUT_DOCUMENTS}.{doc_index}.{DOCUMENT_SCORE}", score


def _get_retriever_response_attributes(response: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Extracts retriever-related attributes from the response.
    """
    from haystack import Document

    if (
        (documents := response.get("documents")) is None
        or not isinstance(documents, Sequence)
        or not all(map(lambda x: isinstance(x, Document), documents))
    ):
        return
    for doc_index, doc in enumerate(documents):
        if (content := doc.content) is not None:
            yield f"{RETRIEVAL_DOCUMENTS}.{doc_index}.{DOCUMENT_CONTENT}", content
        if (id := doc.id) is not None:
            yield f"{RETRIEVAL_DOCUMENTS}.{doc_index}.{DOCUMENT_ID}", id
        if (score := doc.score) is not None:
            yield f"{RETRIEVAL_DOCUMENTS}.{doc_index}.{DOCUMENT_SCORE}", score
        if (metadata := doc.meta) is not None:
            yield (
                f"{RETRIEVAL_DOCUMENTS}.{doc_index}.{DOCUMENT_METADATA}",
                safe_json_dumps(metadata),
            )


def _get_embedding_model_attributes(component: "Component") -> Iterator[Tuple[str, Any]]:
    """
    Yields attributes for embedding model.
    """

    if (
        model := (getattr(component, "model", None) or getattr(component, "model_name", None))
    ) and isinstance(model, str):
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
    from haystack import Document

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


def _is_list_of_documents(value: Any) -> TypeGuard[Sequence["Document"]]:
    """
    Checks for a list of documents.
    """

    from haystack import Document

    return isinstance(value, Sequence) and all(map(lambda x: isinstance(x, Document), value))


def _get_bound_arguments(function: Callable[..., Any], *args: Any, **kwargs: Any) -> BoundArguments:
    """
    Safely returns bound arguments from the current context.
    """
    sig = signature(function)
    accepts_arbitrary_kwargs = any(
        param.kind == Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )
    valid_kwargs = {
        key: value
        for key, value in kwargs.items()
        if accepts_arbitrary_kwargs or key in sig.parameters
    }
    return sig.bind(*args, **valid_kwargs)


CHAIN = OpenInferenceSpanKindValues.CHAIN.value
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING.value
LLM = OpenInferenceSpanKindValues.LLM.value
RERANKER = OpenInferenceSpanKindValues.RERANKER.value
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
RERANKER_INPUT_DOCUMENTS = RerankerAttributes.RERANKER_INPUT_DOCUMENTS
RERANKER_MODEL_NAME = RerankerAttributes.RERANKER_MODEL_NAME
RERANKER_OUTPUT_DOCUMENTS = RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS
RERANKER_QUERY = RerankerAttributes.RERANKER_QUERY
RERANKER_TOP_K = RerankerAttributes.RERANKER_TOP_K
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
USER_ID = SpanAttributes.USER_ID
