import logging
from typing import Dict, Any

from openinference.instrumentation.bedrock.handlers import (
    initialize_process_trace_span,
    handle_model_invocation_input,
    handle_model_invocation_output,
    handle_invocation_input,
    handle_rationale,
    handle_file_operations,
    handle_observation
)
from openinference.instrumentation.bedrock.utils import enhance_span_attributes
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes
)
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_trace_info(data_part):
    for field in ['modelInvocationInput', 'modelInvocationOutput', 'invocationInput', 'observation', 'rationale']:
        if field in data_part and 'traceId' in data_part[field]:
            return {
                'trace_id': data_part[field]['traceId'],
                'type': data_part[field].get('type')
            }
    return None


def process_failure_trace(message_callback, trace_data, parent_span):
    with message_callback.timing_metrics.measure("failure"):
        with message_callback.tracer.start_as_current_span(
                name="failure",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    "failure.reason": trace_data['failureTrace'].get('failureReason', '')
                }
        ) as failure_span:
            enhance_span_attributes(failure_span, trace_data['failureTrace'])
            failure_span.set_status(Status(StatusCode.ERROR))


def set_common_attributes(span, attributes: Dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None and value != "":
            span.set_attribute(key, value)


def process_trace(message_callback, trace_data, parent_span, trace_type):
    """Process orchestration trace with proper span hierarchy"""
    orch_trace = trace_data.get(trace_type)
    if not (trace_info := get_trace_info(orch_trace)):
        return
    trace_context_key = f"{trace_type}_{trace_info['trace_id']}"

    initialize_process_trace_span(message_callback, parent_span, trace_type, trace_info)
    current_trace_data = message_callback.trace_context.get(trace_context_key)
    trace_processing_data = message_callback.trace_processing_data[trace_type]
    current_span = trace_processing_data['span']

    # Handle different trace types
    handle_model_invocation_input(
        message_callback, orch_trace, current_trace_data, trace_context_key
    )
    handle_model_invocation_output(
        message_callback, orch_trace, current_trace_data, current_span,
        trace_context_key, trace_type
    )
    handle_rationale(orch_trace, current_span, message_callback)
    handle_invocation_input(orch_trace, current_trace_data, current_span, message_callback)

    # Handle file operations if present
    if "files" in trace_data:
        handle_file_operations(trace_data, current_span, message_callback.tracer)

    final_response = handle_observation(orch_trace, current_trace_data, current_span, message_callback)
    if final_response:
        # Before deleting trace data, set status to OK if not already set
        if current_span.status.status_code == StatusCode.UNSET:
            current_span.set_status(Status(StatusCode.OK))

        message_callback.trace_processing_data[trace_type] = {
            'span': None,
            'trace_id': None
        }
        message_callback.trace_context.delete(trace_context_key)


def process_trace_event(trace_data, root_span, message_callback):
    """Process different types of trace events with controlled ordering"""
    with trace.use_span(root_span, end_on_exit=False):
        if 'preProcessingTrace' in trace_data:
            process_trace(message_callback, trace_data, root_span, 'preProcessingTrace')
        elif 'orchestrationTrace' in trace_data:
            process_trace(message_callback, trace_data, root_span, 'orchestrationTrace')
        elif 'postProcessingTrace' in trace_data:
            process_trace(message_callback, trace_data, root_span, 'postProcessingTrace')
        elif 'failureTrace' in trace_data:
            process_failure_trace(message_callback, trace_data, root_span)
