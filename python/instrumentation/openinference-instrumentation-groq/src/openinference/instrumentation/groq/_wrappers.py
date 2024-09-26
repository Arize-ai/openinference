from abc import ABC
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Tuple
from inspect import Signature, signature

import json
import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from opentelemetry.trace import INVALID_SPAN

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    EmbeddingAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from openinference.instrumentation.groq._with_span import _WithSpan
from openinference.instrumentation.groq._request_attributes_extractor import _RequestAttributesExtractor


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
    Base class for wrappers that need a tracer.
    """

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
            self,
            span_name: str,
            attributes: Iterable[Tuple[str, AttributeValue]],
            context_attributes: Iterable[Tuple[str, AttributeValue]],
            extra_attributes: Iterable[Tuple[str, AttributeValue]],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our
        # attributes into two tiers, where "extra_attributes" are added first to
        # ensure that the most important "attributes" are added last and are not
        # dropped.
        try:
            span = self._tracer.start_span(name=span_name, attributes=dict(extra_attributes))
        except Exception:
            span = INVALID_SPAN
        with trace_api.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
        ) as span:
            yield _WithSpan(
                span=span,
                context_attributes=dict(context_attributes),
                extra_attributes=dict(attributes),
            )

def _parse_args(
        signature: Signature,
        *args: Tuple[Any],
        **kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    bound_signature = signature.bind(*args, **kwargs)
    bound_signature.apply_defaults()
    bound_arguments = bound_signature.arguments
    request_data: Dict[str, Any] = {}
    for key, value in bound_arguments.items():
        try:
            if value is not None:
                try:
                    # ensure the value is JSON-serializable
                    safe_json_dumps(value)
                    request_data[key] = value
                except json.JSONDecodeError:
                    request_data[key] = str(value)
        except Exception:
            request_data[key] = str(value)
    return request_data

class _CompletionsWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_extractor = _RequestAttributesExtractor()

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        llm_invocation_params = kwargs
        llm_messages = dict(kwargs).pop("messages", None)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)
        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        span_name = "Completions"
        with self._start_as_current_span(
                span_name=span_name,
                attributes=self._request_extractor.get_attributes_from_request(request_parameters),
                context_attributes=get_attributes_from_context(),
                extra_attributes=[],
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            SpanAttributes.OUTPUT_VALUE: response.choices[0].message.content,
                            SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT,
                            LLM_TOKEN_COUNT_COMPLETION: response.usage.completion_tokens,
                            SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response.usage.prompt_tokens,
                            SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response.usage.total_tokens,
                        }
                    )
                )
            )
            span.set_status(trace_api.StatusCode.OK)

        return response


class _AsyncCompletionsWrapper(_WithTracer):
    """
    Wrapper for the pipeline processing
    Captures all calls to the pipeline
    """

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        llm_invocation_params = kwargs
        llm_messages = dict(kwargs).pop("messages", None)

        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)

        span_name = "AsyncCompletions"
        with self._tracer.start_as_current_span(
            span_name,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))

            span.set_attributes(
                dict(
                    _flatten(
                        {
                            SpanAttributes.OPENINFERENCE_SPAN_KIND: LLM,
                            SpanAttributes.LLM_INPUT_MESSAGES: llm_messages,
                            SpanAttributes.LLM_INVOCATION_PARAMETERS: safe_json_dumps(
                                llm_invocation_params
                            ),
                            SpanAttributes.LLM_MODEL_NAME: llm_invocation_params.get("model"),
                            SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON,
                        }
                    )
                )
            )
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            SpanAttributes.OUTPUT_VALUE: response.choices[0].message.content,
                            SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT,
                            LLM_TOKEN_COUNT_COMPLETION: response.usage.completion_tokens,
                            SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response.usage.prompt_tokens,
                            SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response.usage.total_tokens,
                        }
                    )
                )
            )

            span.set_status(trace_api.StatusCode.OK)

        return response


CHAIN = OpenInferenceSpanKindValues.CHAIN
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER
EMBEDDING = OpenInferenceSpanKindValues.EMBEDDING
LLM = OpenInferenceSpanKindValues.LLM
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
