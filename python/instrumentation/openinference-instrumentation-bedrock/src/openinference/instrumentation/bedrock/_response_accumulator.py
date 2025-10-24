"""Response accumulator module for processing Bedrock service responses."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Mapping, Optional, TypeVar, Union

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from openinference.instrumentation import (
    get_input_attributes,
    get_output_attributes,
    get_span_kind_attributes,
)
from openinference.instrumentation.bedrock._attribute_extractor import AttributeExtractor
from openinference.instrumentation.bedrock._trace_collector import (
    TraceCollector,
    TraceNode,
    TraceSpan,
)
from openinference.instrumentation.bedrock.utils import _finish
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_AnyT = TypeVar("_AnyT")  # Type variable for generic return type

logger = logging.getLogger(__name__)


class _Attributes:
    """
    Container for span attributes and metadata.

    This class stores various attributes related to a span, including name,
    type, request and output attributes, metadata, and span kind.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        span_kind: OpenInferenceSpanKindValues = OpenInferenceSpanKindValues.LLM,
    ) -> None:
        """
        Initialize an _Attributes instance.

        Args:
            name: Optional name for the span
            span_kind: Kind of span (LLM, CHAIN, AGENT, etc.)
        """
        self.name: Optional[str] = name
        self.span_type: str = ""
        self.request_attributes: Dict[str, Any] = {}
        self.output_attributes: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.span_kind: OpenInferenceSpanKindValues = span_kind
        self.requires_agent_span: bool = False


class _ResponseAccumulator:
    """
    Accumulates and processes responses from BedrockAgent service.

    This class handles the processing of trace events, creating spans for different
    types of events, and managing the lifecycle of these spans. It acts as a callback
    for processing streaming responses from BedrockAgent and extracting telemetry data.
    """

    def __init__(
        self, span: Span, tracer: Tracer, request: Mapping[str, Any], idx: int = 0
    ) -> None:
        """
        Initialize the ResponseAccumulator.

        Args:
            span: The parent span for tracing
            tracer: The tracer instance for creating new spans
            request: The request parameters sent to Bedrock
            idx: Index for the accumulator, useful when handling multiple responses
        """
        self._span = span
        self._final_response: str = ""
        self._request_parameters = request
        self.tracer = tracer
        self._is_finished: bool = False
        self.trace_collector = TraceCollector()

        # Track model invocation inputs for matching with outputs
        self.model_inputs: Dict[str, Dict[str, Any]] = {}

    def __call__(self, obj: _AnyT) -> _AnyT:
        """
        Process an object received from the Bedrock service.

        This method is called for each chunk of data received from Bedrock.
        It handles different types of responses:
        - Dictionary responses with chunks (containing output text)
        - Dictionary responses with traces (containing telemetry data)
        - StopIteration/StopAsyncIteration signals (end of stream)
        - Exception objects (error handling)

        Args:
            obj: The object to process from the Bedrock service response

        Returns:
            The same object, allowing for pass-through processing

        Raises:
            Exception: Re-raises any exception after recording it in the span
        """
        try:
            if isinstance(obj, dict):
                if "chunk" in obj:
                    if "bytes" in obj["chunk"]:
                        output_text = obj["chunk"]["bytes"].decode("utf-8")
                        self._final_response += output_text
                        self._span.set_attributes(get_output_attributes(self._final_response))
                elif "trace" in obj:
                    self.trace_collector.collect(obj)
            elif isinstance(obj, (StopIteration, StopAsyncIteration)):
                self._process_traces()
                self._finish_tracing()

            elif isinstance(obj, BaseException):
                self._process_traces()
                self._handle_exception(obj)
        except Exception as e:
            logger.exception(e)
            self._span.record_exception(e)
            self._span.set_status(Status(StatusCode.ERROR))
            self._span.end()
            raise e
        return obj

    def _process_traces(self) -> None:
        """
        Process the collected traces and create spans.

        This method is called when all traces have been collected and need to be
        processed into OpenTelemetry spans. It starts the recursive span creation
        process using the initial trace node as the root.
        """
        self._create_spans(self._span, self.trace_collector.initial_node)

    def _create_chain_span(
        self,
        parent_span: Optional[Span],
        attributes: _Attributes,
        start_time: Optional[int] = None,
    ) -> Span:
        """
        Create a new span with the given attributes.

        This method creates a new OpenTelemetry span as a child of the parent span,
        with attributes derived from the provided _Attributes object. It handles
        setting span attributes, extracting and merging metadata, and setting the
        appropriate start time.

        Args:
            parent_span: The parent span for this new span
            attributes: The attributes object containing span properties
            start_time: Optional explicit start time for the span (nanoseconds)

        Returns:
            The newly created span
        """
        # Extract start time from attributes metadata if available
        start_time = (
            attributes.metadata.get("startTime", None) if attributes.metadata else start_time
        )

        # Create the span with appropriate context
        span = self.tracer.start_span(
            name=attributes.name or "LLM",
            context=trace_api.set_span_in_context(parent_span if parent_span else self._span),
            start_time=int(start_time) if start_time else None,
            attributes=get_span_kind_attributes(attributes.span_kind),
        )

        # Collect and merge metadata from various sources
        metadata = attributes.metadata or {}

        # Set request attributes and extract any metadata
        if attributes.request_attributes:
            if "metadata" in attributes.request_attributes:
                metadata.update(attributes.request_attributes.pop("metadata"))
            span.set_attributes(attributes.request_attributes)

        # Set output attributes and extract any metadata
        if attributes.output_attributes:
            if "metadata" in attributes.output_attributes:
                metadata.update(attributes.output_attributes.pop("metadata"))
            span.set_attributes(attributes.output_attributes)

        # Add collected metadata as a JSON string
        if metadata:
            span.set_attributes({"metadata": json.dumps(metadata)})

        return span

    def _fetch_span_time(
        self,
        attributes: _Attributes,
        trace_span: Union[TraceNode, TraceSpan],
        time_key: str,
        reverse: bool = False,
    ) -> Optional[int]:
        """
        Fetch span time (start or end) from attributes or trace span.

        This method recursively searches for timestamp information in the trace data.
        It first checks the attributes metadata, then searches through the trace span
        data structure, looking in both model invocation outputs and observations.

        Args:
            attributes: The attributes object that may contain timing metadata
            trace_span: The trace span to extract time from
            time_key: The key to look for ('startTime' or 'endTime')
            reverse: Whether to traverse spans and chunks in reverse order
                     (useful for finding end times which are typically at the end)

        Returns:
            The timestamp in nanoseconds if found, None otherwise
        """
        # First check if time is already in attributes metadata
        if attributes.metadata and (time_value := attributes.metadata.get(time_key)):
            return int(time_value)

        if not isinstance(trace_span, TraceNode):
            return None

        spans = trace_span.spans[::-1] if reverse else trace_span.spans

        for span in spans:
            chunks = span.chunks[::-1] if reverse else span.chunks

            for trace_data in chunks:
                trace_event = AttributeExtractor.get_event_type(trace_data)
                event_data = trace_data.get(trace_event, {})

                # Check model invocation output
                if "modelInvocationOutput" in event_data:
                    model_invocation_output = event_data.get("modelInvocationOutput", {})
                    metadata = AttributeExtractor.get_metadata_attributes(
                        model_invocation_output.get("metadata")
                    )
                    if time_value := metadata.get(time_key):
                        return int(time_value)

                # Check observation
                if "observation" in event_data:
                    observation = event_data.get("observation", {})
                    metadata = AttributeExtractor.get_observation_metadata_attributes(observation)
                    if time_value := metadata.get(time_key):
                        return int(time_value)

                # Check guardrail trace
                if "metadata" in event_data:
                    metadata = AttributeExtractor.get_metadata_attributes(
                        event_data.get("metadata")
                    )
                    if time_value := metadata.get(time_key):
                        return int(time_value)

            # Recursively check nested nodes
            if isinstance(span, TraceNode):
                if time_value := self._fetch_span_time(attributes, span, time_key, reverse):
                    return int(time_value)

        return None

    def _fetch_span_start_time(
        self, attributes: _Attributes, trace_span: Union[TraceSpan, TraceNode]
    ) -> Optional[int]:
        """
        Fetch the start time for a span.

        Args:
            attributes: The attributes object that may contain timing metadata
            trace_span: The trace span to extract time from

        Returns:
            The start timestamp in nanoseconds if found, None otherwise
        """
        return self._fetch_span_time(attributes, trace_span, "startTime", reverse=False)

    def _fetch_span_end_time(
        self, attributes: _Attributes, trace_span: Union[TraceSpan, TraceNode]
    ) -> Optional[int]:
        """
        Fetch the end time for a span.

        Args:
            attributes: The attributes object that may contain timing metadata
            trace_span: The trace span to extract time from

        Returns:
            The end timestamp in nanoseconds if found, None otherwise
        """
        return self._fetch_span_time(attributes, trace_span, "endTime", reverse=True)

    def _set_parent_span_input_attributes(
        self, attributes: _Attributes, trace_span: Union[TraceSpan, TraceNode]
    ) -> None:
        """
        Set input attributes for the parent span from trace data.

        Args:
            attributes (_Attributes): The attributes object to update
            trace_span (TraceSpan): The trace span to extract attributes from
        """
        input_attributes = dict()
        if not isinstance(trace_span, TraceNode):
            return

        # First check the node's own chunks (for agent-collaborator nodes)
        for trace_data in trace_span.chunks:
            trace_event = AttributeExtractor.get_event_type(trace_data)
            event_data = trace_data.get(trace_event, {})

            # Extract from invocation input in node chunks
            if "invocationInput" in event_data:
                invocation_input = event_data.get("invocationInput", {})
                # For agent-collaborator nodes, get full attributes including LLM messages
                if trace_span.node_type == "agent-collaborator":
                    attrs = AttributeExtractor.get_attributes_from_invocation_input(
                        invocation_input
                    )
                else:
                    attrs = AttributeExtractor.get_parent_input_attributes_from_invocation_input(
                        invocation_input
                    )
                if attrs:
                    input_attributes.update(attrs)
                    input_attributes.update(attributes.request_attributes)
                    attributes.request_attributes = input_attributes
                    return

        # Then check spans
        for span in trace_span.spans:
            for trace_data in span.chunks:
                trace_event = AttributeExtractor.get_event_type(trace_data)
                event_data = trace_data.get(trace_event, {})

                # Extract from model invocation input
                if "modelInvocationInput" in event_data:
                    model_invocation_input = event_data.get("modelInvocationInput", {})
                    text = model_invocation_input.get("text", "")
                    for message in AttributeExtractor.get_messages_object(text):
                        if message.get("role") == "user" and (
                            input_value := message.get("content")
                        ):
                            input_attributes.update(get_input_attributes(input_value))
                            input_attributes.update(attributes.request_attributes)
                            attributes.request_attributes = input_attributes
                            return

                # Extract from invocation input
                if "invocationInput" in event_data:
                    invocation_input = event_data.get("invocationInput", {})
                    attrs = AttributeExtractor.get_parent_input_attributes_from_invocation_input(
                        invocation_input
                    )
                    if attrs:
                        input_attributes.update(attrs)
                        input_attributes.update(attributes.request_attributes)
                        attributes.request_attributes = input_attributes
                        return

            # Recursively check nested nodes
            if isinstance(span, TraceNode):
                return self._set_parent_span_input_attributes(attributes, span)

    def _set_parent_span_output_attributes(
        self, attributes: _Attributes, trace_span: Union[TraceSpan, TraceNode]
    ) -> None:
        """
        Set output attributes for the parent span from trace data.

        Args:
            attributes (_Attributes): The attributes object to update
            trace_span (TraceSpan): The trace span to extract attributes from
        """
        if not isinstance(trace_span, TraceNode):
            return

        # First check the node's own chunks (for agent-collaborator nodes)
        for trace_data in trace_span.chunks[::-1]:
            trace_event = AttributeExtractor.get_event_type(trace_data)
            event_data = trace_data.get(trace_event, {})

            # Extract from observation in node chunks
            if "observation" in event_data:
                observation = event_data.get("observation", {})
                # For agent-collaborator nodes, get full output attributes including LLM messages
                if trace_span.node_type == "agent-collaborator":
                    attrs = AttributeExtractor.get_attributes_from_observation(observation)
                    if attrs:
                        attributes.request_attributes.update(attrs)
                    return
                elif final_response := observation.get("finalResponse"):
                    if text := final_response.get("text", ""):
                        attributes.request_attributes.update(get_output_attributes(text))
                    return

        # Then check spans
        for span in trace_span.spans[::-1]:
            for trace_data in span.chunks[::-1]:
                trace_event = AttributeExtractor.get_event_type(trace_data)
                event_data = trace_data.get(trace_event, {})

                # Extract from model invocation input
                if "modelInvocationOutput" in event_data:
                    model_invocation_output = event_data.get("modelInvocationOutput", {})
                    parsed_response = model_invocation_output.get("parsedResponse", {})
                    if output_text := parsed_response.get("text", ""):
                        # This block will be executed for Post Processing trace
                        return attributes.request_attributes.update(
                            get_output_attributes(output_text)
                        )
                    if output_text := parsed_response.get("rationale", ""):
                        # This block will be executed for Pre Processing trace
                        return attributes.request_attributes.update(
                            get_output_attributes(output_text)
                        )
                    # For Routing classifier events, the output is in rawResponse.content
                    if parsed_response == {}:
                        raw_response = model_invocation_output.get("rawResponse", {})
                        if raw_response_content := raw_response.get("content"):
                            try:
                                response_content_json = json.loads(raw_response_content)
                                if (
                                    output_content := response_content_json.get("output", {})
                                    .get("message", {})
                                    .get("content")
                                ):
                                    return attributes.request_attributes.update(
                                        get_output_attributes(output_content)
                                    )
                                # Return full parsed json if output isn't found
                                return attributes.request_attributes.update(
                                    get_output_attributes(response_content_json)
                                )
                            except Exception:
                                pass
                            # Fallback to raw response if content was not valid JSON
                            return attributes.request_attributes.update(
                                get_output_attributes(raw_response_content)
                            )

                # Extract from invocation input
                if "observation" in event_data:
                    observation = event_data.get("observation", {})
                    if final_response := observation.get("finalResponse"):
                        if text := final_response.get("text", ""):
                            attributes.request_attributes.update(get_output_attributes(text))
                        return

            # Recursively check nested nodes
            if isinstance(span, TraceNode):
                return self._set_parent_span_output_attributes(attributes, span)

    def _create_spans(self, parent_span: Span, trace_node: TraceNode) -> None:
        """
        Create spans recursively from trace nodes.

        This method processes each trace span in the trace node, creating OpenTelemetry
        spans for each one. It handles setting span attributes, determining span kind,
        and recursively processing child spans. It also manages span timing by fetching
        start and end times from the trace data.

        Args:
            parent_span: The parent span to attach new spans to
            trace_node: The trace node containing spans to process
        """
        for trace_span in trace_node.spans:
            attributes = self._prepare_span_attributes(trace_span)

            # Set span kind based on trace span type
            if isinstance(trace_span, TraceNode):
                attributes.span_kind = OpenInferenceSpanKindValues.CHAIN
                if trace_span.node_type == "agent-collaborator":
                    attributes.span_kind = OpenInferenceSpanKindValues.AGENT
                self._set_parent_span_input_attributes(attributes, trace_span)
                self._set_parent_span_output_attributes(attributes, trace_span)

            # Create span with appropriate timing
            start_time = self._fetch_span_start_time(attributes, trace_span)

            span = self._create_chain_span(parent_span, attributes, start_time)
            status_code = StatusCode.OK
            if attributes.span_kind == OpenInferenceSpanKindValues.GUARDRAIL:
                intervening_guardrails = attributes.metadata.get("intervening_guardrails", [])
                if AttributeExtractor.is_blocked_guardrail(intervening_guardrails):
                    status_code = StatusCode.ERROR
            span.set_status(Status(status_code))

            # Process child spans recursively
            if isinstance(trace_span, TraceNode):
                self._create_spans(span, trace_span)

            # End span with appropriate timing
            end_time = self._fetch_span_end_time(attributes, trace_span)
            span.end(end_time=int(end_time) if end_time else None)

    def _handle_exception(self, obj: BaseException) -> None:
        """
        Handle an exception object.

        Args:
            obj (BaseException): The exception to handle.
        """
        self._span.record_exception(obj)
        self._span.set_status(Status(StatusCode.ERROR, str(obj)))
        self._span.end()

    @classmethod
    def _prepare_span_attributes(cls, trace_span_data: Union[TraceSpan, TraceNode]) -> _Attributes:
        """
        Extract attributes from a TraceSpan.

        This method analyzes a TraceSpan and extracts relevant attributes for creating
        an OpenTelemetry span. It processes each chunk in the trace span and delegates
        to specialized processing methods based on the event type found in the chunk.

        Args:
            trace_span_data: The TraceSpan to extract attributes from

        Returns:
            An _Attributes object containing the extracted attributes
        """
        _attributes = _Attributes()

        # Set name from node type if it's a TraceNode
        if isinstance(trace_span_data, TraceNode):
            if trace_span_data.node_type == "guardrailTrace":
                pre_or_post = trace_span_data.node_trace_id.split("-")[-1]
                _attributes.name = pre_or_post + "GuardrailTrace"
            else:
                _attributes.name = trace_span_data.node_type
            cls._process_trace_node(trace_span_data, _attributes)
            return _attributes

        # Process each chunk in the trace span
        for trace_data in trace_span_data.chunks:
            trace_event = AttributeExtractor.get_event_type(trace_data)
            event_data = trace_data.get(trace_event, {})

            # Process model invocation input
            if "modelInvocationInput" in event_data:
                cls._process_model_invocation_input(event_data, _attributes)

            # Process model invocation output
            if "modelInvocationOutput" in event_data:
                cls._process_model_invocation_output(event_data, _attributes)

            # Process invocation input
            if "invocationInput" in event_data:
                cls._process_invocation_input(event_data, _attributes)

            # Process observation
            if "observation" in event_data:
                cls._process_observation(event_data, _attributes)

            # Process rationale
            if "rationale" in event_data:
                cls._process_rationale(event_data, _attributes)

            if trace_event == "guardrailTrace":
                cls._process_guardrail_trace(event_data, _attributes)

            if trace_event == "failureTrace":
                cls._process_failure_trace(event_data, _attributes)
        return _attributes

    @classmethod
    def _process_trace_node(cls, trace_node: TraceNode, attributes: _Attributes) -> None:
        """
        Process trace node data. This extracts metadata attributes and adds them
        to the trace node span. This includes routingClassifierTrace, orchestrationTrace,
        guardrailTrace, etc.
        """
        for trace_data in trace_node.chunks:
            trace_event = AttributeExtractor.get_event_type(trace_data)
            event_data = trace_data.get(trace_event, {})

            # Extract agent collaborator name for agent-collaborator nodes
            if trace_node.node_type == "agent-collaborator" and "invocationInput" in event_data:
                invocation_input = event_data.get("invocationInput", {})
                if "agentCollaboratorInvocationInput" in invocation_input:
                    agent_collaborator_name = invocation_input.get(
                        "agentCollaboratorInvocationInput", {}
                    ).get("agentCollaboratorName", "")
                    invocation_type = invocation_input.get("invocationType", "")
                    if agent_collaborator_name:
                        attributes.name = f"{invocation_type.lower()}[{agent_collaborator_name}]"

            # Extract child-level metadata first (will be overridden by trace-level metadata)
            if "modelInvocationOutput" in event_data:
                model_invocation_output = event_data.get("modelInvocationOutput", {})
                attributes.metadata.update(
                    AttributeExtractor.get_metadata_attributes(
                        model_invocation_output.get("metadata")
                    )
                )
            if observation := event_data.get("observation"):
                # For agent-collaborator nodes, extract metadata from the observation itself
                if trace_node.node_type == "agent-collaborator":
                    if observation_metadata := observation.get("metadata"):
                        attributes.metadata.update(
                            AttributeExtractor.get_observation_metadata_attributes(
                                observation_metadata
                            )
                        )
                # For other nodes, extract from finalResponse if present
                if final_response := observation.get("finalResponse"):
                    if final_response_metadata := final_response.get("metadata"):
                        attributes.metadata.update(
                            AttributeExtractor.get_metadata_attributes(final_response_metadata)
                        )

            # Extract trace-level metadata last so it takes precedence
            # (for orchestrationTrace, guardrailTrace, etc.)
            if metadata := event_data.get("metadata"):
                attributes.metadata.update(AttributeExtractor.get_metadata_attributes(metadata))

    @classmethod
    def _process_model_invocation_input(
        cls, event_data: Dict[str, Any], attributes: _Attributes
    ) -> None:
        """
        Process model invocation input data.

        Extracts attributes from model invocation input events and updates the
        attributes object with request attributes. Also sets the span name to "LLM"
        and the span kind to LLM.

        Args:
            event_data: The event data containing model invocation input
            attributes: The attributes object to update
        """
        model_invocation_input = event_data.get("modelInvocationInput", {})
        attributes.request_attributes.update(
            AttributeExtractor.get_attributes_from_model_invocation_input(model_invocation_input)
        )
        attributes.name = "LLM"
        attributes.span_kind = OpenInferenceSpanKindValues.LLM

    @classmethod
    def _process_model_invocation_output(
        cls, event_data: Dict[str, Any], attributes: _Attributes
    ) -> None:
        """
        Process model invocation output data.

        Extracts attributes from model invocation output events and updates the
        attributes object with output attributes and metadata.

        Args:
            event_data: The event data containing model invocation output
            attributes: The attributes object to update
        """
        model_invocation_output = event_data.get("modelInvocationOutput", {})
        attributes.output_attributes.update(
            AttributeExtractor.get_attributes_from_model_invocation_output(model_invocation_output)
        )
        attributes.metadata.update(
            AttributeExtractor.get_metadata_attributes(model_invocation_output.get("metadata"))
        )

    @classmethod
    def _process_invocation_input(cls, event_data: Dict[str, Any], attributes: _Attributes) -> None:
        """
        Process invocation input data.

        Extracts attributes from invocation input events and updates the attributes
        object with request attributes. Also sets the span name and kind based on
        the invocation type, handling special cases for agent collaborator invocations.

        Args:
            event_data: The event data containing invocation input
            attributes: The attributes object to update
        """
        invocation_input = event_data.get("invocationInput") or {}
        invocation_type = invocation_input.get("invocationType", "")

        if "agentCollaboratorInvocationInput" in invocation_input:
            agent_collaborator_name = invocation_input.get(
                "agentCollaboratorInvocationInput", {}
            ).get("agentCollaboratorName", "")
            attributes.name = f"{invocation_type.lower()}[{agent_collaborator_name}]"
            attributes.span_kind = OpenInferenceSpanKindValues.AGENT
            attributes.span_type = "agent_collaborator"
        else:
            attributes.name = invocation_type.lower()
            attributes.span_kind = OpenInferenceSpanKindValues.TOOL

        attributes.request_attributes.update(
            AttributeExtractor.get_attributes_from_invocation_input(invocation_input)
        )

    @classmethod
    def _process_observation(cls, event_data: Dict[str, Any], attributes: _Attributes) -> None:
        """
        Process observation data.

        Extracts attributes from observation events and updates the attributes
        object with output attributes and metadata.

        Args:
            event_data: The event data containing observation data
            attributes: The attributes object to update
        """
        observation = event_data.get("observation", {})
        attributes.output_attributes.update(
            AttributeExtractor.get_attributes_from_observation(observation)
        )
        attributes.metadata.update(AttributeExtractor.get_metadata_from_observation(observation))

    @classmethod
    def _process_guardrail_trace(cls, event_data: Dict[str, Any], attributes: _Attributes) -> None:
        """
        Process guardrail trace data.
        """
        attributes.span_kind = OpenInferenceSpanKindValues.GUARDRAIL
        attributes.name = "Guardrails"

        guardrail_attributes = AttributeExtractor.get_attributes_from_guardrail_trace(event_data)

        if "intervening_guardrails" not in attributes.metadata:
            attributes.metadata["intervening_guardrails"] = []

        if "non_intervening_guardrails" not in attributes.metadata:
            attributes.metadata["non_intervening_guardrails"] = []

        if guardrail_attributes.get("action") == "INTERVENED":
            attributes.metadata["intervening_guardrails"].append(guardrail_attributes)
        else:
            attributes.metadata["non_intervening_guardrails"].append(guardrail_attributes)

    @classmethod
    def _process_failure_trace(cls, event_data: Dict[str, Any], attributes: _Attributes) -> None:
        """
        Process failure trace data.

        Extracts attributes from failure trace  and updates the attributes
        object with output attributes and metadata.

        Args:
            trace_data: The trace data containing failure data
            attributes: The attributes object to update
        """
        attributes.output_attributes.update(
            AttributeExtractor.get_failure_trace_attributes(event_data)
        )
        attributes.metadata.update(
            AttributeExtractor.get_metadata_attributes(event_data.get("metadata", {}))
        )

    @classmethod
    def _process_rationale(cls, event_data: Dict[str, Any], attributes: _Attributes) -> None:
        """
        Process rationale data.

        Extracts rationale text from event data and updates the attributes
        object with output attributes.

        Args:
            event_data: The event data containing rationale data
            attributes: The attributes object to update
        """
        if rationale_text := event_data.get("rationale", {}).get("text", ""):
            attributes.output_attributes.update(get_output_attributes(rationale_text))

    def _finish_tracing(self) -> None:
        """
        Finish tracing by ending all spans.

        This method is called when all trace data has been collected and processed.
        It ensures that all spans are properly ended and prevents duplicate ending
        of spans by checking the _is_finished flag.
        """
        if self._is_finished:
            return

        # Use span manager to finish tracing
        # These attributes are removed as these are input values of the request.
        # these are not required to be added to the span.
        _finish(span=self._span, result=None, request_attributes={})
        self._is_finished = True


# Constants for document and retrieval attributes
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
