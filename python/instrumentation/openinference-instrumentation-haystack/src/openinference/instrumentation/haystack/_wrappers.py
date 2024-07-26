import json
from abc import ABC
from typing import Any, Callable, Iterator, List, Mapping, Tuple
from enum import Enum
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
    EmbeddingAttributes,
    OpenInferenceMimeTypeValues,
    DocumentAttributes
)
from opentelemetry import trace as trace_api
import opentelemetry.context as context_api
from opentelemetry.util.types import AttributeValue
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps


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

class _WithTracer(ABC):
    """
    Base class for wrappers that need a tracer. Acts as a trait for the wrappers
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

        # Diving into the instance to retrieve the Component name
        component = instance.graph.nodes._nodes[args[0]]['instance']

        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
               invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        span_name = component.__class__.__name__

        with self._tracer.start_as_current_span(
                span_name,
                attributes={}
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)

            attributes = dict(
                _flatten(
                    {
                        "parameters": safe_json_dumps(invocation_parameters),
                    }
                )
            )
            match args[0]:
                case 'llm':
                    if isinstance(response['meta'][0]['usage'], dict):
                        attributes = dict(
                                    _flatten(
                                        {
                                            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM,
                                            SpanAttributes.LLM_MODEL_NAME: response['meta'][0]['model'],
                                            SpanAttributes.INPUT_VALUE: safe_json_dumps(args[1]),
                                            SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                                            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: response['meta'][0]['usage']['completion_tokens'],
                                            SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response['meta'][0]['usage']['prompt_tokens'],
                                            SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response['meta'][0]['usage']['total_tokens'],
                                            SpanAttributes.LLM_OUTPUT_MESSAGES: response["replies"],
                                            SpanAttributes.LLM_PROMPT_TEMPLATE: instance.graph.nodes._nodes["prompt_builder"]["instance"]._template_string,
                                        }
                                    )
                                )
                case 'text_embedder':
                    attributes = dict(
                        _flatten(
                            {
                                SpanAttributes.EMBEDDING_MODEL_NAME: component.model,
                                f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}" : f"<{len(response['embedding'])} dimensional vector>",
                                f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}" : invocation_parameters['text'],
                                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.EMBEDDING,
                                SpanAttributes.INPUT_VALUE: safe_json_dumps(invocation_parameters),
                                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON
                            }
                        )
                    )
                case 'retriever':
                    attributes = dict(
                        _flatten(
                            {
                                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER,
                                # Display full vector?
                                SpanAttributes.INPUT_VALUE: safe_json_dumps({"dimensions": len(args[1]['query_embedding']),
                                                                             "embedding": args[1]['query_embedding']}),
                                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                                SpanAttributes.OUTPUT_VALUE: safe_json_dumps(response['documents']),
                                SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON
                            }
                        )
                    )

                    i = 0
                    for document in response['documents']:
                        attributes[f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}"] = document.content
                        attributes[f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_ID}"] = document.id
                        attributes[f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_SCORE}"] = document.score
                        attributes[f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_METADATA}"] = safe_json_dumps(document.meta)
                        i += 1

                case 'prompt_builder':
                    attributes = dict(
                        _flatten(
                            {
                                SpanAttributes.OPENINFERENCE_SPAN_KIND : OpenInferenceSpanKindValues.CHAIN,
                                SpanAttributes.LLM_PROMPT_TEMPLATE: instance.graph.nodes._nodes["prompt_builder"]["instance"]._template_string,
                                SpanAttributes.INPUT_VALUE: attributes['parameters'],
                                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                                SpanAttributes.RETRIEVAL_DOCUMENTS: safe_json_dumps(invocation_parameters['documents']),
                            }
                        )
                    )

            span.set_attributes(attributes)

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
                attributes={}
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)

            attributes = dict(
                _flatten(
                    {
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN,
                        SpanAttributes.INPUT_VALUE: safe_json_dumps(invocation_parameters),
                        SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON
                    }
                )
            )
            span.set_attributes(attributes)

        return response

'''

class _FetcherWrapper(_WithTracer):
    """
    Captures all calls to the fetcher
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

        span_name = "Fetcher"
        with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            "pipeline.invocation.parameters": safe_json_dumps(invocation_parameters),
                        }
                    )
                ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class _OpenAIGeneratorWrapper(_WithTracer):
    """
    Captures all calls to the OpenAIGenerator
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

        span_name = "OpenAIGenerator"
        with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            "llm.invocation_parameters": safe_json_dumps(invocation_parameters),
                            "llm.model_name": instance.model,
                            "llm.input_messages" : kwargs["prompt"]
                        }
                    )
                ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class _PromptBuilderWrapper(_WithTracer):
    """
    Captures all calls to the OpenAIGenerator
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

        span_name = "PromptBuilder"
        with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            "parameters": safe_json_dumps(invocation_parameters),
                            #"template": instance.template._template_string
                            #"llm.model_name": instance.model,
                            #"llm.input_messages" : kwargs["prompt"]
                        }
                    )
                ),
        ) as span:
            print(f"Template String: {instance.template._template_string}")
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class _HTMLToDocumentWrapper(_WithTracer):
    """
    Captures all calls to the OpenAIGenerator
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

        span_name = "HTMLToDocument"
        with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            "parameters": safe_json_dumps(invocation_parameters),
                            #"llm.model_name": instance.model,
                            #"llm.input_messages" : kwargs["prompt"]
                        }
                    )
                ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class _SentenceTransformersTextEmbedderWrapper(_WithTracer):
    """
    Captures all calls to the OpenAIGenerator
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

        span_name = "SentenceTransformersTextEmbedder"
        with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            "parameters": safe_json_dumps(invocation_parameters),
                            #"llm.model_name": instance.model,
                            #"llm.input_messages" : kwargs["prompt"]
                        }
                    )
                ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class _SentenceTransformersDocumentEmbedderWrapper(_WithTracer):
    """
    Captures all calls to the OpenAIGenerator
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

        span_name = "SentenceTransformersDocumentEmbedder"
        with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            "parameters": safe_json_dumps(invocation_parameters),
                            #"llm.model_name": instance.model,
                            #"llm.input_messages" : kwargs["prompt"]
                        }
                    )
                ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class _InMemoryEmbeddingRetrieverWrapper(_WithTracer):
    """
    Captures all calls to the OpenAIGenerator
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

        span_name = "InMemoryEmbeddingRetriever"
        with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            "parameters": safe_json_dumps(invocation_parameters),
                            #"llm.model_name": instance.model,
                            #"llm.input_messages" : kwargs["prompt"]
                        }
                    )
                ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class _InMemoryDocumentStoreWrapper(_WithTracer):
    """
    Captures all calls to the OpenAIGenerator
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

        span_name = "InMemoryDocumentStore"
        with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            "parameters": safe_json_dumps(invocation_parameters),
                            #"llm.model_name": instance.model,
                            #"llm.input_messages" : kwargs["prompt"]
                        }
                    )
                ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response
'''