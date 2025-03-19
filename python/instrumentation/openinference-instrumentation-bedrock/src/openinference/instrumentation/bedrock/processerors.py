import logging
from typing import Any, Dict

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from openinference.instrumentation.bedrock.handlers import (
    handle_file_operations,
    handle_invocation_input,
    handle_model_invocation_input,
    handle_model_invocation_output,
    handle_observation,
    handle_rationale,
    initialize_process_trace_span,
)
from openinference.instrumentation.bedrock.utils import enhance_span_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_trace_info(data_part: dict[str, Any]) -> dict[str, Any]:
    for field in [
        "modelInvocationInput",
        "modelInvocationOutput",
        "invocationInput",
        "observation",
        "rationale",
    ]:
        if field in data_part and "traceId" in data_part[field]:
            return {"trace_id": data_part[field]["traceId"], "type": data_part[field].get("type")}
    return {}


def process_failure_trace(
    message_callback: Any, trace_data: dict[str, Any], parent_span: Span
) -> None:
    with message_callback.tracer.start_as_current_span(
        name="failure",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "failure.reason": trace_data["failureTrace"].get("failureReason", ""),
        },
    ) as failure_span:
        enhance_span_attributes(failure_span, trace_data["failureTrace"])
        failure_span.set_status(Status(StatusCode.ERROR))


def set_common_attributes(span: Span, attributes: Dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None and value != "":
            span.set_attribute(key, value)


def process_trace(
    message_callback: Any, trace_data: dict[str, Any], parent_span: Span, trace_type: str
) -> None:
    """Process orchestration trace with proper span hierarchy"""
    orch_trace = trace_data.get(trace_type) or {}
    trace_info = get_trace_info(orch_trace)
    if not trace_info:
        return
    trace_context_key = f"{trace_type}_{trace_info['trace_id']}"

    initialize_process_trace_span(message_callback, parent_span, trace_type, trace_info)
    current_trace_data = message_callback.trace_context.get(trace_context_key)
    trace_processing_data = message_callback.trace_processing_data[trace_type]
    current_span = trace_processing_data["span"]

    # Handle different trace types
    handle_model_invocation_input(
        message_callback, orch_trace, current_trace_data, trace_context_key
    )
    handle_model_invocation_output(
        message_callback,
        orch_trace,
        current_trace_data,
        current_span,
        trace_context_key,
        trace_type,
    )
    handle_rationale(orch_trace, current_span, message_callback)
    handle_invocation_input(orch_trace, current_trace_data, current_span, message_callback)

    # Handle file operations if present
    if "files" in trace_data:
        handle_file_operations(trace_data, current_span, message_callback.tracer)

    final_response = handle_observation(
        orch_trace, current_trace_data, current_span, message_callback
    )
    if final_response:
        # Before deleting trace data, set status to OK if not already set
        if current_span.status.status_code == StatusCode.UNSET:
            current_span.set_status(Status(StatusCode.OK))

        message_callback.trace_processing_data[trace_type] = {"span": None, "trace_id": None}
        message_callback.trace_context.delete(trace_context_key)


def process_trace_event(trace_data: dict[str, Any], root_span: Span, message_callback: Any) -> None:
    """Process different types of trace events with controlled ordering"""
    with trace.use_span(root_span, end_on_exit=False):
        if "preProcessingTrace" in trace_data:
            process_trace(message_callback, trace_data, root_span, "preProcessingTrace")
        elif "orchestrationTrace" in trace_data:
            process_trace(message_callback, trace_data, root_span, "orchestrationTrace")
        elif "postProcessingTrace" in trace_data:
            process_trace(message_callback, trace_data, root_span, "postProcessingTrace")
        elif "failureTrace" in trace_data:
            process_failure_trace(message_callback, trace_data, root_span)
