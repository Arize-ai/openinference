"""
Wrapper functions for instrumenting AWS Bedrock RAG (Retrieval-Augmented Generation) operations.

This module provides OpenTelemetry instrumentation for Bedrock's retrieve and retrieve_and_generate
operations, enabling distributed tracing and observability for RAG workflows.
"""

from functools import wraps
from typing import Any, Callable, Dict, List, Mapping, TypeVar

from botocore.client import BaseClient
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from openinference.instrumentation import (
    get_attributes_from_context,
)
from openinference.instrumentation.bedrock._attribute_extractor import AttributeExtractor

_AnyT = TypeVar("_AnyT")  # Type variable for generic return type


def _retrieve_wrapper(tracer: Tracer) -> Callable[[BaseClient], Callable[..., Any]]:
    """
    Create a wrapper for instrumenting Bedrock's retrieve operation with OpenTelemetry tracing.

    This function creates a decorator that wraps the Bedrock client's retrieve method to add
    distributed tracing capabilities. It captures input parameters, response data, and creates
    spans for observability.

    Args:
        tracer (Tracer): OpenTelemetry tracer instance used to create spans for the retrieve
        operation.

    Returns:
        Callable[[BaseClient], Callable[..., Any]]: A decorator function that takes a BaseClient
        and returns an instrumented version of the retrieve method.

    Note:
        The wrapper respects the OpenTelemetry suppression context key and will skip instrumentation
        if suppression is active.
    """

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
    """
    Create a wrapper for instrumenting Bedrock's retrieve_and_generate operation with
    OpenTelemetry tracing.

    This function creates a decorator that wraps the Bedrock client's retrieve_and_generate method
    to add distributed tracing capabilities. This method combines retrieval and generation in a
    single RAG operation, and the wrapper captures both input parameters and response data.

    Args:
        tracer (Tracer): OpenTelemetry tracer instance used to create spans for the
        retrieve_and_generate operation.

    Returns:
        Callable[[BaseClient], Callable[..., Any]]: A decorator function that takes a BaseClient
        and returns an instrumented version of the retrieve_and_generate method.

    Note:
        The wrapper respects the OpenTelemetry suppression context key and will skip instrumentation
        if suppression is active.
    """

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


class _RagEventStream:
    """
    Event stream handler for RAG (Retrieval-Augmented Generation) streaming responses.

    This class processes streaming events from Bedrock RAG operations, accumulating output text
    and citations while managing the associated OpenTelemetry span lifecycle. It acts as a
    callable that processes each event in the stream and finalizes tracing when the stream ends.

    Attributes:
        _span (Span): The OpenTelemetry span associated with this RAG operation.
        _request_parameters (Mapping[str, Any]): The original request parameters.
        tracer (Tracer): OpenTelemetry tracer instance.
        output (str): Accumulated output text from the streaming response.
        citations (List): List of citations collected from the streaming response.
        start_index (int): Starting index for processing (currently unused).
    """

    def __init__(
        self, span: Span, tracer: Tracer, request: Mapping[str, Any], idx: int = 0
    ) -> None:
        """
        Initialize the RAG event stream handler.

        Args:
            span (Span): OpenTelemetry span to associate with this RAG operation.
            tracer (Tracer): OpenTelemetry tracer instance.
            request (Mapping[str, Any]): The original request parameters for the RAG operation.
            idx (int, optional): Starting index for processing. Defaults to 0.
        """
        self._span = span
        self._request_parameters = request
        self.tracer = tracer
        self.output: str = ""
        self.citations: List[Dict[str, Any]] = []
        self.start_index = 0
        self.output = ""

    def __call__(self, obj: _AnyT) -> _AnyT:
        """
        Process a streaming event object from the RAG operation.

        This method handles different types of objects in the event stream:
        - Dict objects containing output text and citations
        - StopIteration/StopAsyncIteration indicating stream completion
        - BaseException objects indicating errors

        Args:
            obj (_AnyT): The event object to process. Can be a dict with response data,
                        a stop iteration signal, or an exception.

        Returns:
            _AnyT: The same object that was passed in, allowing for transparent processing.

        Raises:
            Exception: Re-raises any exception that occurs during processing after recording
                      it in the span.

        Note:
            When the stream ends (StopIteration/StopAsyncIteration), this method finalizes
            the span with accumulated output and citations. On exceptions, it records the
            error and sets the span status to ERROR before ending the span.
        """
        try:
            if isinstance(obj, dict):
                if output := obj.get("output", {}).get("text"):
                    self.output += output
                if citation := obj.get("citation"):
                    self.citations += [citation]
                print("CITATIONS LEN", len(self.citations))
                print(self.output)
            elif isinstance(obj, (StopIteration, StopAsyncIteration)):
                self._span.set_attributes(dict(get_attributes_from_context()))
                self._span.set_attributes(
                    AttributeExtractor.extract_bedrock_rag_response_attributes(
                        {"citations": self.citations, "output": {"text": self.output}}
                    )
                )
                self._span.set_status(Status(StatusCode.OK))
                self._span.end()
            elif isinstance(obj, BaseException):
                self._span.set_attributes(dict(get_attributes_from_context()))
                self._span.record_exception(obj)
                self._span.set_status(Status(StatusCode.ERROR))
                self._span.end()
        except Exception as e:
            self._span.record_exception(obj)  # type: ignore
            self._span.set_status(Status(StatusCode.ERROR))
            self._span.end()
            raise e
        return obj
