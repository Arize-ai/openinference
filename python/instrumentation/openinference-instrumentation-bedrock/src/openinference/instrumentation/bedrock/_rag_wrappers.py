from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
)

from botocore.client import BaseClient
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Status, StatusCode, Tracer

from openinference.instrumentation import (
    get_attributes_from_context,
)
from openinference.instrumentation.bedrock._attribute_extractor import AttributeExtractor


def _retrieve_wrapper(tracer: Tracer) -> Callable[[BaseClient], Callable[..., Any]]:
    def _invocation_wrapper(wrapped_client: BaseClient) -> Callable[..., Any]:
        @wraps(wrapped_client.retrieve)
        def instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return wrapped_client._unwrapped_retrieve(*args, **kwargs)  # type: ignore

            with tracer.start_as_current_span("bedrock.retrieve") as span:
                span.set_attributes(
                    AttributeExtractor.extract_bedrock_retrieve_input_attributes(kwargs)
                )
                response = wrapped_client._unwrapped_retrieve(*args, **kwargs)
                span.set_attributes(dict(get_attributes_from_context()))
                span.set_attributes(
                    AttributeExtractor.extract_bedrock_retrieve_response_attributes(response)
                )
                span.set_status(Status(StatusCode.OK))
                return response  # type: ignore

        return instrumented_response

    return _invocation_wrapper


def _retrieve_and_generate_wrapper(tracer: Tracer) -> Callable[[BaseClient], Callable[..., Any]]:
    def _invocation_wrapper(wrapped_client: BaseClient) -> Callable[..., Any]:
        @wraps(wrapped_client.retrieve_and_generate)
        def instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return wrapped_client._unwrapped_retrieve_and_generate(*args, **kwargs)  # type: ignore

            with tracer.start_as_current_span("bedrock.retrieve_and_generate") as span:
                span.set_attributes(AttributeExtractor.extract_bedrock_rag_input_attributes(kwargs))
                response = wrapped_client._unwrapped_retrieve_and_generate(*args, **kwargs)
                span.set_attributes(dict(get_attributes_from_context()))
                span.set_attributes(
                    AttributeExtractor.extract_bedrock_rag_response_attributes(response)
                )
                span.set_status(Status(StatusCode.OK))
                return response  # type: ignore

        return instrumented_response

    return _invocation_wrapper
