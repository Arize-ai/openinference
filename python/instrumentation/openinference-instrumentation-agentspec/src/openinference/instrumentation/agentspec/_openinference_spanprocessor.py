from typing import Any, Dict, Optional

from opentelemetry.sdk.trace import Resource as OtelSdkResource  # type: ignore
from opentelemetry.sdk.trace.export import BatchSpanProcessor as OtelSdkBatchSpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor as OtelSdkSimpleSpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter as OtelSdkSpanExporter
from opentelemetry.sdk.trace.export import SpanProcessor as OtelSdkSpanProcessor  # type: ignore
from pyagentspec.property import _empty_default
from pyagentspec.tools import Tool as AgentSpecTool
from pyagentspec.tracing.events import Event as AgentSpecEvent
from pyagentspec.tracing.events import LlmGenerationRequest as AgentSpecLlmGenerationRequest
from pyagentspec.tracing.events import LlmGenerationResponse as AgentSpecLlmGenerationResponse
from pyagentspec.tracing.events import ToolExecutionRequest as AgentSpecToolExecutionRequest
from pyagentspec.tracing.spans import AgentExecutionSpan as AgentSpecAgentExecutionSpan
from pyagentspec.tracing.spans import LlmGenerationSpan as AgentSpecLlmGenerationSpan
from pyagentspec.tracing.spans import Span as AgentSpecSpan
from pyagentspec.tracing.spans import ToolExecutionSpan as AgentSpecToolExecutionSpan

from openinference.instrumentation.agentspec._opentelemetry_spanprocessor import (
    _OtelSpanProcessor,
    _try_json_serialization,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


def _get_event_of_given_type(
    span: AgentSpecSpan, event_type: type[AgentSpecEvent]
) -> Optional[AgentSpecEvent]:
    try:
        return next(event for event in span.events if isinstance(event, event_type))
    except StopIteration:
        return None


def _get_tool_json_schema(tool: AgentSpecTool) -> Dict[str, Any]:
    # Return tool's json schema as per OpenAI tools format
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {p.title: p.json_schema for p in tool.inputs or []},
                "required": [p.title for p in (tool.inputs or []) if p.default is _empty_default],
            },
        },
    }


def _to_open_inference_format_tracing_info(
    span: AgentSpecSpan, mask_sensitive_information: bool = True
) -> Dict[str, Any]:
    # See https://arize.com/docs/ax/concepts/tracing/semantic-conventions for naming conventions
    trace = span._trace
    tracing_info: Dict[str, Any] = span.model_dump(mask_sensitive_information=mask_sensitive_information)
    tracing_info[SpanAttributes.SESSION_ID] = trace.id if trace else None
    # The default value we use for span kind is CHAIN. We will overwrite if we find a better fit.
    tracing_info[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.CHAIN.value
    if isinstance(span, AgentSpecAgentExecutionSpan):
        tracing_info[SpanAttributes.OPENINFERENCE_SPAN_KIND] = (
            OpenInferenceSpanKindValues.AGENT.value
        )
        tracing_info[SpanAttributes.AGENT_NAME] = span.agent.name
        # Agent inputs and outputs are set at the end
    elif isinstance(span, AgentSpecLlmGenerationSpan):
        llm_info = tracing_info.pop("llm_config")
        tracing_info[SpanAttributes.OPENINFERENCE_SPAN_KIND] = OpenInferenceSpanKindValues.LLM.value
        tracing_info[SpanAttributes.LLM_MODEL_NAME] = llm_info.get("model_id", "name")
        tracing_info[SpanAttributes.LLM_INVOCATION_PARAMETERS] = _try_json_serialization(
            {
                k: v
                for k, v in (llm_info.get("default_generation_parameters") or {}).items()
                if v is not None
            }
        )
        # Start event for AgentSpecLlmGenerationSpan is a AgentSpecLlmGenerationRequest
        start_event = _get_event_of_given_type(span, AgentSpecLlmGenerationRequest)
        if start_event is not None:
            # If we find one, we update all the information related to the LLM requests
            # in the Span info
            event_info = start_event.model_dump(
                mask_sensitive_information=mask_sensitive_information
            )
            if event_info["llm_generation_config"]:
                # Update generation parameters with the specific ones for this call
                tracing_info[SpanAttributes.LLM_INVOCATION_PARAMETERS] = _try_json_serialization(
                    {
                        k: v
                        for k, v in (event_info.get("llm_generation_config") or {}).items()
                        if v is not None
                    }
                )
            # LLM tools seem to require the JSON schema of the tools in OpenAI format
            tracing_info[SpanAttributes.LLM_TOOLS] = [
                {
                    ToolAttributes.TOOL_JSON_SCHEMA: _try_json_serialization(
                        _get_tool_json_schema(tool)
                    )
                }
                for tool in start_event.tools
            ]
            tracing_info[SpanAttributes.LLM_INPUT_MESSAGES] = [
                # The prompt contains the list of messages dumped in dict format,
                # we can use that directly
                {"message": e}
                for e in event_info["prompt"]
            ]
        # End event for AgentSpecLlmGenerationSpan is a AgentSpecLlmGenerationResponse
        end_event = _get_event_of_given_type(span, AgentSpecLlmGenerationResponse)
        if end_event is not None:
            # If we find one, we update all the information related to the LLM response
            # in the Span info
            event_info = end_event.model_dump(mask_sensitive_information=mask_sensitive_information)
            tracing_info[SpanAttributes.LLM_OUTPUT_MESSAGES] = [
                {
                    "message": {
                        "content": event_info.pop("content"),
                        "tool_calls": [
                            {
                                "tool_call.id": tool_call["call_id"],
                                "tool_call.function.name": tool_call["tool_name"],
                                "tool_call.function.arguments": tool_call["arguments"],
                            }
                            for tool_call in event_info["tool_calls"]
                            if isinstance(tool_call, dict)
                        ],
                        "role": "assistant",
                    },
                }
            ]
            tracing_info[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = event_info.pop(
                "output_tokens"
            )
            tracing_info[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = event_info.pop("input_tokens")
    elif isinstance(span, AgentSpecToolExecutionSpan):
        tracing_info[SpanAttributes.OPENINFERENCE_SPAN_KIND] = (
            OpenInferenceSpanKindValues.TOOL.value
        )
        tool_info = tracing_info.pop("tool")
        tracing_info[SpanAttributes.TOOL_NAME] = tool_info["name"]
        tracing_info[SpanAttributes.TOOL_DESCRIPTION] = tool_info["description"]
        tracing_info[ToolAttributes.TOOL_JSON_SCHEMA] = _try_json_serialization(
            _get_tool_json_schema(span.tool)
        )
        # Start event for AgentSpecToolExecutionSpan is a AgentSpecToolExecutionRequest
        start_event = _get_event_of_given_type(span, AgentSpecToolExecutionRequest)
        if start_event is not None:
            # If we find one, we update all the information related to the tool execution
            # in the Span info
            event_info = start_event.model_dump(
                mask_sensitive_information=mask_sensitive_information
            )
            tracing_info[ToolCallAttributes.TOOL_CALL_ID] = event_info["request_id"]
            tracing_info[ToolCallAttributes.TOOL_CALL_FUNCTION_NAME] = tracing_info[
                SpanAttributes.TOOL_NAME
            ]
            tracing_info[SpanAttributes.TOOL_PARAMETERS] = event_info["inputs"]
            tracing_info[ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON] = (
                _try_json_serialization(event_info["inputs"])
            )

    if (
        tracing_info[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        != OpenInferenceSpanKindValues.LLM.value
    ):
        # Spans often have generic `inputs` and `outputs`. When available, we use start event
        # `inputs` attribute as input and end event `outputs` attribute as output of the span
        start_event = span.events[0] if len(span.events) > 0 else None
        if start_event is not None:
            event_info = start_event.model_dump(
                mask_sensitive_information=mask_sensitive_information
            )
            tracing_info[SpanAttributes.INPUT_VALUE] = _try_json_serialization(
                event_info.get("inputs", {})
            )
        end_event = span.events[-1] if len(span.events) > 1 else None
        if end_event is not None:
            event_info = end_event.model_dump(mask_sensitive_information=mask_sensitive_information)
            tracing_info[SpanAttributes.OUTPUT_VALUE] = _try_json_serialization(
                event_info.get("outputs", {})
            )

    return tracing_info


class OpenInferenceSpanProcessor(_OtelSpanProcessor):
    def __init__(
        self,
        resource: Optional[OtelSdkResource] = None,
        otel_span_processor: Optional[OtelSdkSpanProcessor] = None,
        span_exporter: Optional[OtelSdkSpanExporter] = None,
        use_batch_span_processor: bool = True,
        mask_sensitive_information: bool = True,
    ):
        """
        AgentSpec OpenInference SpanProcessor.

        This class forwards the calls to AgentSpec's span processors to an OpenInference one
        (based on OpenTelemetry).

        Parameters
        ----------
        resource:
            The OpenTelemetry Resource to use in Spans.
        otel_span_processor:
            The OpenTelemetry SpanProcessor to use to process spans.
            If None is given, a new instance is created based on ``span_exporter`` and
            ``use_batch_span_processor`` parameters.
        span_exporter:
            The OpenTelemetry SpanExporter to use to export spans.
            This parameter is ignored if an ``otel_span_processor`` is provided.
        use_batch_span_processor:
            If True, an instance of OpenTelemetry BatchSpanProcessor is used to process spans.
            If False, an instance of OpenTelemetry SimpleSpanProcessor is used to process spans.
            This parameter is ignored if an ``otel_span_processor`` is provided.
        mask_sensitive_information
            Whether to mask potentially sensitive information from the span and its events
        """
        self.use_batch_span_processor = use_batch_span_processor
        super().__init__(
            otel_span_processor=otel_span_processor,
            span_exporter=span_exporter,
            resource=resource,
            mask_sensitive_information=mask_sensitive_information,
            span_model_dump_func=_to_open_inference_format_tracing_info,
        )

    def _create_otel_span_processor(
        self, span_exporter: OtelSdkSpanExporter
    ) -> OtelSdkSpanProcessor:
        if self.use_batch_span_processor:
            return OtelSdkBatchSpanProcessor(span_exporter=span_exporter)
        else:
            return OtelSdkSimpleSpanProcessor(span_exporter=span_exporter)
