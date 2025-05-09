from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import StatusCode, set_span_in_context

from openinference.instrumentation.helpers import safe_json_dumps
from openinference.semconv.trace import (
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.tool_context import ToolContext

    from openinference.instrumentation import OITracer


logger = logging.getLogger(__name__)


AGENT = OpenInferenceSpanKindValues.AGENT.value
LLM = OpenInferenceSpanKindValues.LLM.value
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
TOOL = OpenInferenceSpanKindValues.TOOL.value


class GoogleADKTracingCallback:
    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer
        self._original_value = None
        self._current_spans: Dict[str, OtelSpan] = {}

    def before_agent_callback(
        self,
        callback_context: "CallbackContext",
    ) -> Optional[Any]:
        otel_span: OtelSpan = self._tracer.start_span(
            name=f"{callback_context.agent_name}-{AGENT}",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: AGENT,
                LLM_SYSTEM: OpenInferenceLLMSystemValues.VERTEXAI.value,
            },
        )
        self._current_spans[AGENT] = otel_span

        if callable(self._original_value):
            return self._original_value(callback_context)

        return None

    def before_model_callback(
        self,
        callback_context: "CallbackContext",
        llm_request: "LlmRequest",
    ) -> Optional[Any]:
        context = (
            set_span_in_context(self._current_spans[AGENT]) if self._current_spans[AGENT] else None
        )
        otel_span: OtelSpan = self._tracer.start_span(
            name=f"{callback_context.agent_name}-{LLM}",
            context=context,
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: LLM,
                LLM_SYSTEM: OpenInferenceLLMSystemValues.VERTEXAI.value,
            },
        )
        otel_span.set_attributes(
            {
                SpanAttributes.INPUT_VALUE: safe_json_dumps(
                    [content.model_dump(exclude_none=True) for content in llm_request.contents]
                ),
                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                SpanAttributes.LLM_INVOCATION_PARAMETERS: safe_json_dumps(
                    llm_request.config.model_dump(exclude_none=True)
                ),
                SpanAttributes.LLM_MODEL_NAME: llm_request.model,
                SpanAttributes.LLM_TOOLS: safe_json_dumps(
                    [
                        {
                            SpanAttributes.TOOL_NAME: tool.name,
                            SpanAttributes.TOOL_DESCRIPTION: tool.description,
                        }
                        for tool in llm_request.tools_dict.values()
                    ]
                ),
            }
        )
        self._current_spans[LLM] = otel_span

        if callable(self._original_value):
            return self._original_value(callback_context, llm_request)

        return None

    def before_tool_callback(
        self,
        tool: "BaseTool",
        args: Dict[str, Any],
        tool_context: "ToolContext",
    ) -> Optional[Any]:
        context = (
            set_span_in_context(self._current_spans[AGENT]) if self._current_spans[AGENT] else None
        )
        otel_span: OtelSpan = self._tracer.start_span(
            name=f"{tool_context.agent_name}-{TOOL}",
            context=context,
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: TOOL,
                LLM_SYSTEM: OpenInferenceLLMSystemValues.VERTEXAI.value,
            },
        )
        otel_span.set_attributes(
            {
                SpanAttributes.INPUT_VALUE: safe_json_dumps(args),
                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                SpanAttributes.TOOL_NAME: tool.name,
                SpanAttributes.TOOL_DESCRIPTION: tool.description,
            }
        )
        self._current_spans[TOOL] = otel_span

        if callable(self._original_value):
            return self._original_value(tool, args, tool_context)

        return None

    def after_agent_callback(
        self,
        callback_context: "CallbackContext",
    ) -> Optional[Any]:
        otel_span = self._current_spans[AGENT]
        otel_span.set_status(StatusCode.OK)
        otel_span.end()

        del self._current_spans[AGENT]

        if callable(self._original_value):
            return self._original_value(callback_context)

        return None

    def after_model_callback(
        self,
        callback_context: "CallbackContext",
        llm_response: "LlmResponse",
    ) -> Optional[Any]:
        otel_span = self._current_spans[LLM]

        if llm_response.content.parts[0].text:
            otel_span.set_attributes(
                {
                    SpanAttributes.OUTPUT_VALUE: llm_response.content.parts[0].text,
                    SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
                }
            )
        else:
            # Function calls
            otel_span.set_attributes(
                {
                    SpanAttributes.OUTPUT_VALUE: safe_json_dumps(
                        llm_response.content.model_dump(exclude_none=True),
                    ),
                    SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                }
            )

        otel_span.set_status(StatusCode.OK)
        otel_span.end()
        del self._current_spans[LLM]

        if callable(self._original_value):
            return self._original_value(callback_context, llm_response)

        return None

    def after_tool_callback(
        self,
        tool: "BaseTool",
        args: Dict[str, Any],
        tool_context: "ToolContext",
        tool_response: Dict[Any, Any],
    ) -> Optional[Any]:
        otel_span = self._current_spans[TOOL]

        otel_span.set_attributes(
            {
                SpanAttributes.OUTPUT_VALUE: safe_json_dumps(tool_response),
                SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
            }
        )
        otel_span.set_status(StatusCode.OK)
        otel_span.end()

        del self._current_spans[TOOL]

        if callable(self._original_value):
            return self._original_value(tool, args, tool_context, tool_response)

        return None
