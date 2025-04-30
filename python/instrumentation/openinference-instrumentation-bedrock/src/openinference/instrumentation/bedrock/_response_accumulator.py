"""Response accumulator module for processing Bedrock service responses."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, TypeVar

from opentelemetry.trace import Span, Status, StatusCode, Tracer

from openinference.instrumentation import (
    get_output_attributes,
)
from openinference.instrumentation.bedrock.attribute_extractor import AttributeExtractor
from openinference.instrumentation.bedrock.span_manager import SpanManager
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_AnyT = TypeVar("_AnyT")  # Type variable for generic return type

logger = logging.getLogger(__name__)


class _ResponseAccumulator:
    """
    Accumulates and processes responses from Bedrock service.

    This class handles the processing of trace events, creating spans for different
    types of events, and managing the lifecycle of these spans.
    """

    def __init__(
        self, span: Span, tracer: Tracer, request: Mapping[str, Any], idx: int = 0
    ) -> None:
        """
        Initialize the ResponseAccumulator.

        Args:
            span (Span): The parent span for tracing.
            tracer (Tracer): The tracer instance.
            request (Mapping[str, Any]): The request parameters.
            idx (int, optional): Index for the accumulator. Defaults to 0.
        """
        self._span = span
        self._request_parameters = request
        self.tracer = tracer
        self._is_finished: bool = False

        # Initialize span manager
        self.span_manager = SpanManager(span, tracer, request)

        # Track model invocation inputs for matching with outputs
        self.model_inputs: dict[str, dict[str, Any]] = {}

    def __call__(self, obj: _AnyT) -> _AnyT:
        """
        Process an object received from the Bedrock service.

        Args:
            obj (_AnyT): The object to process.

        Returns:
            _AnyT: The processed object.
        """
        try:
            if isinstance(obj, dict):
                self._process_dict_object(obj)
            elif isinstance(obj, (StopIteration, StopAsyncIteration)):
                self._finish_tracing()
            elif isinstance(obj, BaseException):
                self._handle_exception(obj)
        except Exception as e:
            logger.exception(e)
            self._span.record_exception(e)
            self._span.set_status(Status(StatusCode.ERROR))
            self._span.end()
            raise e
        return obj

    def _process_dict_object(self, obj: Dict[str, Any]) -> None:
        """
        Process a dictionary object received from the Bedrock service.

        Args:
            obj (dict): The dictionary object to process.
        """
        if "chunk" in obj:
            if "bytes" in obj["chunk"]:
                output_text = obj["chunk"]["bytes"].decode("utf-8")
                self._span.set_attributes(get_output_attributes(output_text))
        elif "trace" in obj:
            self._process_trace_event(obj["trace"]["trace"])

    def _handle_exception(self, obj: BaseException) -> None:
        """
        Handle an exception object.

        Args:
            obj (BaseException): The exception to handle.
        """
        self.span_manager.handle_exception(obj)

    def _handle_model_invocation_input_trace(
        self, trace_data: dict[str, Any], trace_event: str, trace_id: str
    ) -> None:
        """
        Handle a model invocation input trace.

        Args:
            trace_data (dict[str, Any]): The trace data.
            trace_event (str): The trace event type.
            trace_id (str): The trace ID.
        """

        model_span = self.span_manager.create_child_span(
            trace_id=trace_id, name="LLM", span_kind=OpenInferenceSpanKindValues.LLM
        )

        # Store the model invocation input for later matching with output
        model_invocation_input = trace_data.get("modelInvocationInput", {})
        request_attributes = AttributeExtractor.get_attributes_from_model_invocation_input(
            model_invocation_input
        )
        if model_span:
            model_span.set_attributes(request_attributes)

        # Add input attributes to the parent span
        self.span_manager.add_model_invocation_attributes_to_parent_span(
            trace_id, model_invocation_input
        )

    def _handle_model_invocation_output_trace(
        self, trace_data: dict[str, Any], trace_event: str, trace_id: str
    ) -> None:
        """
        Handle a model invocation output trace.

        Args:
            trace_data (dict[str, Any]): The trace data.
            trace_event (str): The trace event type.
            trace_id (str): The trace ID.
        """
        # Get the model invocation output
        model_invocation_output = trace_data.get("modelInvocationOutput", {})
        model_span = self.span_manager.get_child_span(trace_id=trace_id)
        request_attributes = AttributeExtractor.get_attributes_from_model_invocation_output(
            model_invocation_output
        )
        if model_span:
            model_span.set_attributes(request_attributes)
            model_span.set_status(Status(StatusCode.OK))

        # Set output on the parent span
        self.span_manager.set_parent_trace_output(trace_id, model_invocation_output)

    def _handle_invocation_input_trace(
        self, trace_data: dict[str, Any], trace_event: str, trace_id: str
    ) -> None:
        """
        Handle an invocation input trace.

        Args:
            trace_data (dict[str, Any]): The trace data.
            trace_event (str): The trace event type.
            trace_id (str): The trace ID.
        """

        # Get the invocation input
        invocation_input = trace_data.get("invocationInput", {})
        if "agentCollaboratorInvocationInput" in invocation_input:
            invocation_span = self.span_manager.create_chain_span(
                trace_id=f"{trace_id}_agent",
                trace_event=trace_event,
                trace_name=invocation_input.get("invocationType", "").lower(),
                span_kind=OpenInferenceSpanKindValues.AGENT,
            )
        else:
            invocation_span = self.span_manager.create_child_span(
                trace_id=trace_id,
                name=invocation_input.get("invocationType", "").lower(),
                span_kind=OpenInferenceSpanKindValues.TOOL,
            )
        attributes = AttributeExtractor.get_attributes_from_invocation_input(invocation_input)
        if invocation_span and attributes is not None:
            invocation_span.set_attributes(attributes)

        # Add input attributes to the parent span
        self.span_manager.add_invocation_attributes_to_parent_span(trace_id, invocation_input)

    def _handle_observation_trace(
        self, trace_data: dict[str, Any], trace_event: str, trace_id: str
    ) -> None:
        """
        Handle an observation trace.

        Args:
            trace_data (dict[str, Any]): The trace data.
            trace_event (str): The trace event type.
            trace_id (str): The trace ID.
        """
        # Get the observation
        observation = trace_data.get("observation", {})
        parent_span = self.span_manager.get_chain_span(trace_id)
        if "agentCollaboratorInvocationOutput" in observation:
            invocation_span = self.span_manager.get_chain_span(f"{trace_id}_agent")
        else:
            invocation_span = self.span_manager.get_child_span(trace_id)
        if invocation_span:
            attributes = AttributeExtractor.get_attributes_from_observation(observation)
            if attributes is not None:
                invocation_span.set_attributes(attributes)
            invocation_span.set_status(Status(StatusCode.OK))
        # Check for final response
        if parent_span and (final_response := observation.get("finalResponse")):
            parent_span.set_attributes(get_output_attributes(final_response.get("text", "")))
            parent_span.set_status(Status(StatusCode.OK))
            return

    def _handle_rationale_trace(
        self, trace_data: dict[str, Any], trace_event: str, trace_id: str
    ) -> None:
        """
        Handle a rationale trace.

        Args:
            trace_data (dict[str, Any]): The trace data.
            trace_event (str): The trace event type.
            trace_id (str): The trace ID.
        """
        # Initialize the parent chain span if it doesn't exist
        parent_span = self.span_manager.get_chain_span(trace_id)
        # Add output attributes to the parent span
        if parent_span and (rationale_text := trace_data.get("rationale", {}).get("text", "")):
            parent_span.set_attributes(get_output_attributes(rationale_text))
            parent_span.set_status(Status(StatusCode.OK))

    def _process_trace_event(self, trace_data: dict[str, Any]) -> None:
        """
        Process a trace event and delegate it to the appropriate handler based on event type.

        This method extracts the trace event type, generates a trace ID, and routes the event
        to the appropriate handler method based on the content of the event data.

        Args:
            trace_data (dict[str, Any]): The trace data containing information about the event.
        """
        trace_event = AttributeExtractor.get_event_type(trace_data)
        if not trace_event:
            return

        # Extract trace ID
        trace_id = self.span_manager.extract_trace_id(trace_data)
        trace_id = f"{trace_event}-{'-'.join(trace_id.split('-')[:5])}"
        # Uncomment for debugging
        # print(trace_id_old, trace_id)
        # Get the event data
        event_data = trace_data.get(trace_event, {})

        # Handle different trace types
        if "modelInvocationInput" in event_data:
            self.span_manager.adjust_parent(trace_event)
            self.span_manager.create_chain_span(trace_id, trace_event)
            self._handle_model_invocation_input_trace(event_data, trace_event, trace_id)

        if "modelInvocationOutput" in event_data:
            self.span_manager.adjust_parent(trace_event)
            self.span_manager.create_chain_span(trace_id, trace_event)
            self._handle_model_invocation_output_trace(event_data, trace_event, trace_id)

        if "invocationInput" in event_data:
            self.span_manager.adjust_parent(trace_event)
            self.span_manager.create_chain_span(trace_id, trace_event)
            self._handle_invocation_input_trace(event_data, trace_event, trace_id)

        if "observation" in event_data:
            self.span_manager.adjust_parent(trace_event)
            self.span_manager.create_chain_span(trace_id, trace_event)
            self._handle_observation_trace(event_data, trace_event, trace_id)

        if "rationale" in event_data:
            self.span_manager.adjust_parent(trace_event)
            self.span_manager.create_chain_span(trace_id, trace_event)
            self._handle_rationale_trace(event_data, trace_event, trace_id)

    def _finish_tracing(self) -> None:
        """
        Finish tracing by ending all spans.
        """
        if self._is_finished:
            return

        # Use span manager to finish tracing
        self.span_manager.finish_tracing()
        self._is_finished = True


# Constants
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
