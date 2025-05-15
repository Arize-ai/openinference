"""Span manager module for managing OpenTelemetry spans in Bedrock instrumentation."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from openinference.instrumentation import (
    get_input_attributes,
    get_output_attributes,
    get_span_kind_attributes,
)
from openinference.instrumentation.bedrock.attribute_extractor import AttributeExtractor
from openinference.instrumentation.bedrock.utils import _finish
from openinference.semconv.trace import OpenInferenceSpanKindValues

logger = logging.getLogger(__name__)


class SpanManager:
    """
    Manages spans for trace events in Bedrock instrumentation.

    This class is responsible for creating, tracking, and managing the lifecycle of spans
    for different types of trace events. It maintains a hierarchy of spans including chain
    spans (parent spans) and child spans, and provides methods to interact with these spans.
    """

    def __init__(self, parent_span: Span, tracer: Tracer, request_parameters: Mapping[str, Any]):
        """
        Initialize the SpanManager.

        Args:
            parent_span (Span): The parent span.
            tracer (Tracer): The tracer instance.
            request_parameters (Mapping[str, Any]): The request parameters.
        """
        self.parent_span = parent_span
        self.tracer = tracer
        self.request_parameters = request_parameters

        # Current parent span reference - always a chain span
        self.current_parent_span = parent_span
        self.parent_trace_id = "default"

        # Enhanced data structure for both chain and child spans
        self.chain_spans: dict[str, dict[str, Any]] = {
            self.parent_trace_id: {
                "spanType": "CHAIN",
                "traceId": self.parent_trace_id,
                "parentSpan": None,
                "event": "",
                "parent_trace_id": self.parent_trace_id,
                "currentSpan": self.current_parent_span,
                "setInput": True,
                "setOutput": False,
            }
        }  # Chain spans (CHAIN type)
        self.child_spans: dict[str, dict[str, Any]] = {}  # Child spans (LLM, AGENT, TOOL etc.)
        self.raw_chain_spans: list[Span] = []  # Tracks span lifecycle state
        self.raw_child_spans: list[Span] = []
        self.is_finished: bool = False

    def create_chain_span(
        self,
        trace_id: str,
        trace_event: str,
        parent_span: Optional[Span] = None,
        trace_name: Optional[str] = None,
        span_kind: OpenInferenceSpanKindValues = OpenInferenceSpanKindValues.CHAIN,
    ) -> Any:
        """
        Initialize a chain span with enhanced metadata structure.

        Args:
            trace_id (str): The trace ID.
            trace_event (str): The trace event type.
            parent_span (Span, optional): The parent span. Defaults to self.parent_span.
            trace_name (str): The kind of span. Defaults to OpenInferenceSpanKindValues.CHAIN.
            span_kind (str): The kind of span. Defaults to OpenInferenceSpanKindValues.CHAIN.
        Returns:
            Span: The initialized chain span.
        """
        if trace_id not in self.chain_spans:
            # Use provided parent_span or default to the initial parent_span
            parent = parent_span if parent_span is not None else self.current_parent_span
            name = trace_name if trace_name else trace_event
            # Create the span
            span = self.tracer.start_span(
                name=name,
                context=trace_api.set_span_in_context(parent),
                attributes=get_span_kind_attributes(
                    span_kind
                    if isinstance(span_kind, OpenInferenceSpanKindValues)
                    else OpenInferenceSpanKindValues.CHAIN
                ),
            )
            self.raw_chain_spans.append(span)
            # Store with enhanced metadata
            self.chain_spans[trace_id] = {
                "spanType": "CHAIN",
                "event": trace_event,
                "traceId": trace_id,
                "parentSpan": parent,
                "parent_trace_id": self.parent_trace_id,
                "currentSpan": span,
                "setInput": False,
                "setOutput": False,
            }
            self.parent_trace_id = trace_id
            # Update current parent span reference
            self.current_parent_span = span
        else:
            self.current_parent_span = self.chain_spans[trace_id]["currentSpan"]
        return self.chain_spans[trace_id]["currentSpan"]

    def create_child_span(
        self, trace_id: str, name: str, span_kind: OpenInferenceSpanKindValues
    ) -> Any:
        """
        Create a child span with the specified parent.

        Args:
            trace_id (str): The trace ID.
            name (str): The name of the span.
            span_kind (str): The kind of span.

        Returns:
            Span: The created child span.
        """
        # Get the parent span from chain_spans
        parent_span = self.current_parent_span

        # Create the child span
        span = self.tracer.start_span(
            name=name,
            context=trace_api.set_span_in_context(parent_span),
            attributes=get_span_kind_attributes(
                span_kind
                if isinstance(span_kind, OpenInferenceSpanKindValues)
                else OpenInferenceSpanKindValues.TOOL
            ),
        )
        self.raw_child_spans.append(span)
        # Store with enhanced metadata (same structure as chain_spans)
        self.child_spans[trace_id] = {
            "spanType": span_kind,
            "event": name,
            "traceId": trace_id,
            "parentSpan": parent_span,
            "currentSpan": span,
            "parentId": self.parent_trace_id,  # Store parent ID for easier navigation
        }
        return self.child_spans[trace_id]["currentSpan"]

    def get_chain_span(self, trace_id: str) -> Optional[Span]:
        """
        Get a chain span by trace ID.

        Args:
            trace_id (str): The trace ID.

        Returns:
            Span: The chain span.
        """
        return self.chain_spans.get(trace_id, {}).get("currentSpan")

    def has_input_set_to_parent(self) -> Optional[bool]:
        """
        Check if input has been set to the parent span.

        Returns:
            bool: True if input has been set, False otherwise.
        """
        return self.chain_spans[self.parent_trace_id].get("setInput")

    def get_child_span(self, trace_id: str) -> Optional[Span]:
        """
        Get a child span by trace ID.

        Args:
            trace_id (str): The trace ID.

        Returns:
            Span: The child span.
        """
        return self.child_spans.get(trace_id, {}).get("currentSpan")

    def add_model_invocation_attributes_to_parent_span(
        self, trace_id: str, model_invocation_input: dict[str, Any]
    ) -> None:
        """
        Add model invocation attributes to the parent span.

        This method extracts user messages from the model invocation input and sets them
        as input attributes on the parent span. It only sets the attributes if they haven't
        been set already.

        Args:
            trace_id (str): The trace ID.
            model_invocation_input (dict[str, Any]): The model invocation input.
        """
        if self.has_input_set_to_parent():
            return
        try:
            text = model_invocation_input.get("text", "")
            for message in AttributeExtractor.get_messages_object(text):
                if message.get("role") == "user" and (input_value := message.get("content")):
                    self.current_parent_span.set_attributes(get_input_attributes(input_value))
                    self.chain_spans[self.parent_trace_id]["setInput"] = True
                    break
        except Exception:
            self.current_parent_span.set_attributes(
                get_input_attributes(model_invocation_input.get("text", ""))
            )

    def add_invocation_attributes_to_parent_span(
        self, trace_id: str, invocation_input: dict[str, Any]
    ) -> None:
        """
        Add invocation attributes to the parent span.

        This method extracts input attributes from the invocation input and sets them
        on the parent span. It only sets the attributes if they haven't been set already.

        Args:
            trace_id (str): The trace ID.
            invocation_input (dict[str, Any]): The invocation input.
        """
        if self.chain_spans[self.parent_trace_id]["setInput"] or not self.current_parent_span:
            return
        self.current_parent_span.set_attributes(
            AttributeExtractor.get_parent_input_attributes_from_invocation_input(invocation_input)
        )
        self.chain_spans[self.parent_trace_id]["setInput"] = True

    def set_parent_trace_output(self, trace_id: str, model_output: dict[str, Any]) -> None:
        """
        Set the output value for the parent trace.

        This method extracts output text from the model output and sets it as output
        attributes on the parent span. It handles both regular text output and rationale
        output from pre-processing traces.

        Args:
            trace_id (str): The trace ID.
            model_output (dict[str, Any]): The model output.
        """
        parent_span = self.get_chain_span(trace_id)
        if not parent_span:
            return

        parsed_response = model_output.get("parsedResponse", {})

        if output_text := parsed_response.get("text", ""):
            # This block will be executed for Post Processing trace
            parent_span.set_attributes(get_output_attributes(output_text))

        if output_text := parsed_response.get("rationale", ""):
            # This block will be executed for Pre Processing trace
            parent_span.set_attributes(get_output_attributes(output_text))

    def finish_tracing(self) -> None:
        """
        Finish tracing by ending all spans.
        """
        if self.is_finished:
            return

        # Finish all spans
        self.finish_chain_spans()
        self.finish_child_spans()
        _finish(self.parent_span, None, self.request_parameters)
        self.is_finished = True

    def finish_chain_spans(self) -> None:
        """
        Finish all chain spans.
        """
        for chain_span in self.raw_chain_spans[::-1]:
            chain_span.set_status(Status(StatusCode.OK))
            chain_span.end()

    def finish_child_spans(self) -> None:
        """
        Finish all child spans.
        """
        for child_span in self.raw_child_spans[::-1]:
            if child_span:
                child_span.set_status(Status(StatusCode.OK))
                child_span.end()

    def adjust_parent(self, trace_event: str) -> None:
        """
        Adjust the parent span based on the trace event.

        Args:
            trace_event (str): The trace event type.
        """
        if event := self.chain_spans[self.parent_trace_id]["event"]:
            if not event == trace_event:
                self.current_parent_span = self.chain_spans[self.parent_trace_id]["parentSpan"]
                self.parent_trace_id = self.chain_spans[self.parent_trace_id]["parent_trace_id"]

    def extract_trace_id(self, trace_data: dict[str, Any]) -> Any:
        """
        Extract a unique trace ID from trace data.

        This method attempts to find a trace ID in various locations within the trace data.
        It checks the main event data, model invocation input/output, invocation input,
        observation data, and rationale data. If no trace ID is found, it generates a unique
        ID based on the event type and current span counts.

        Args:
            trace_data (dict[str, Any]): The trace data containing trace information.

        Returns:
            str: A unique trace ID extracted from the data or generated if none exists.
        """
        trace_event = AttributeExtractor.get_event_type(trace_data)
        event_data = trace_data.get(trace_event, {})

        # Try to get trace ID from the trace data
        if "traceId" in event_data:
            return event_data["traceId"]

        # For model invocation traces
        if "modelInvocationInput" in event_data:
            model_input = event_data["modelInvocationInput"]
            if "traceId" in model_input:
                return model_input["traceId"]

        if "modelInvocationOutput" in event_data:
            model_output = event_data["modelInvocationOutput"]
            if "traceId" in model_output:
                return model_output["traceId"]

        # For invocation input traces
        if "invocationInput" in event_data:
            invocation_input = event_data["invocationInput"]
            if "traceId" in invocation_input:
                return invocation_input["traceId"]

        # For observation traces
        if "observation" in event_data:
            observation = event_data["observation"]
            if "traceId" in observation:
                return observation["traceId"]
        if "rationale" in event_data:
            rationale = event_data["rationale"]
            if "traceId" in rationale:
                return rationale["traceId"]

        # Generate a unique ID if none found
        return f"{trace_event}_{len(self.chain_spans) + len(self.child_spans)}"

    def handle_exception(self, exception: BaseException) -> None:
        """
        Handle an exception.

        Args:
            exception (BaseException): The exception to handle.
        """
        self.finish_chain_spans()
        self.finish_child_spans()
        self.parent_span.record_exception(exception)
        self.parent_span.set_status(Status(StatusCode.ERROR, str(exception)))
        self.parent_span.end()
