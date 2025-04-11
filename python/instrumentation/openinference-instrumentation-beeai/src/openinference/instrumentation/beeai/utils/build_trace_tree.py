# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional, TypedDict

from opentelemetry import trace

from openinference.instrumentation import OITracer
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from .create_span import FrameworkSpan


class BuildTraceTreeData(TypedDict):
    prompt: Optional[str]
    history: list[dict[str, Any]]
    generatedMessage: dict[str, Any] | None
    spans: List[FrameworkSpan]
    traceId: str
    version: str
    startTime: int
    endTime: int
    source: Optional[str]
    run_error_span_key: str


def build_spans_for_parent(
    tracer: OITracer, spans: List[FrameworkSpan], trace_id: str, parent_id: str | None
) -> None:
    children = [s for s in spans if s.get("parent_id") == parent_id]

    for span in children:
        attributes = {
            "target": span["attributes"]["target"],
            "name": span.get("name"),
            "traceId": trace_id,
        }

        if span["attributes"]["metadata"]:
            attributes["metadata"] = str(span["attributes"]["metadata"])

        if span["attributes"]["data"]:
            attributes.update(span["attributes"]["data"])

        with tracer.start_as_current_span(
            name=span["context"]["span_id"],
            attributes={k: v for k, v in attributes.items() if v is not None},
            start_time=span.get("start_time"),
        ) as current_child:
            status = span.get("status")
            if status is not None:
                current_child.set_status(status["code"])
            build_spans_for_parent(
                tracer=tracer,
                spans=spans,
                trace_id=trace_id,
                parent_id=span["context"]["span_id"],
            )


def build_agent_main_span_data(data: BuildTraceTreeData) -> dict[str, Any]:
    payload = {}
    if data.get("prompt"):
        payload[SpanAttributes.INPUT_VALUE] = data.get("prompt")
    if data.get("generatedMessage"):
        payload[SpanAttributes.OUTPUT_VALUE] = str(data.get("generatedMessage"))
    if data.get("history"):
        payload["history"] = str(data.get("history"))
    return payload


def build_tool_main_span_data(data: BuildTraceTreeData) -> dict[str, Any]:
    start_span = next((s for s in data["spans"] if s["name"] == "start"), None)
    success_span = next((s for s in data["spans"] if s["name"] == "success"), None)

    payload = {}
    if start_span and start_span["attributes"]["data"] is not None:
        tool_params = start_span["attributes"]["data"].get(SpanAttributes.TOOL_PARAMETERS)
        payload[SpanAttributes.INPUT_VALUE] = tool_params
        payload[SpanAttributes.TOOL_PARAMETERS] = tool_params
    if success_span and success_span["attributes"]["data"] is not None:
        payload[SpanAttributes.OUTPUT_VALUE] = success_span["attributes"]["data"].get(
            SpanAttributes.OUTPUT_VALUE
        )
    return payload


def build_llm_main_span_data(data: BuildTraceTreeData) -> dict[str, Any]:
    start_span = next((s for s in data["spans"] if s["name"] == "start"), None)
    success_span = next((s for s in data["spans"] if s["name"] == "success"), None)
    provider = None
    model_name = None

    if start_span and start_span["attributes"]["data"] is not None:
        provider = start_span["attributes"]["data"].get(SpanAttributes.LLM_PROVIDER)
        model_name = start_span["attributes"]["data"].get(SpanAttributes.LLM_MODEL_NAME)

    if success_span and success_span["attributes"]["data"] is not None:
        provider = provider or success_span["attributes"]["data"].get(SpanAttributes.LLM_PROVIDER)
        model_name = model_name or success_span["attributes"]["data"].get(
            SpanAttributes.LLM_MODEL_NAME
        )

    payload = {}
    if start_span and start_span["attributes"]["data"] is not None:
        payload[SpanAttributes.INPUT_VALUE] = start_span["attributes"]["data"].get(
            SpanAttributes.INPUT_VALUE
        )
        payload[SpanAttributes.INPUT_MIME_TYPE] = start_span["attributes"]["data"].get(
            SpanAttributes.INPUT_MIME_TYPE
        )
    if success_span and success_span["attributes"]["data"] is not None:
        payload[SpanAttributes.OUTPUT_VALUE] = success_span["attributes"]["data"].get(
            SpanAttributes.OUTPUT_VALUE
        )
        payload[SpanAttributes.OUTPUT_MIME_TYPE] = success_span["attributes"]["data"].get(
            SpanAttributes.OUTPUT_MIME_TYPE
        )
    if provider:
        payload[SpanAttributes.LLM_PROVIDER] = provider
    if model_name:
        payload[SpanAttributes.LLM_MODEL_NAME] = model_name

    return payload


def build_trace_tree(tracer: OITracer, main_span_kind: str, data: BuildTraceTreeData) -> None:
    if main_span_kind == OpenInferenceSpanKindValues.AGENT.value:
        computed_data = build_agent_main_span_data(data)
    elif main_span_kind == OpenInferenceSpanKindValues.TOOL.value:
        computed_data = build_tool_main_span_data(data)
    elif main_span_kind == OpenInferenceSpanKindValues.LLM.value:
        computed_data = build_llm_main_span_data(data)
    else:
        computed_data = {}

    attributes = {
        "source": data["source"],
        "traceId": data["traceId"],
        SpanAttributes.OPENINFERENCE_SPAN_KIND: main_span_kind,
        "beeai.version": data["version"],
        **computed_data,
    }

    with tracer.start_as_current_span(
        name="beeai-framework-main",
        start_time=data["startTime"],
        attributes=attributes,
    ) as current_span:
        run_error_span = next(
            (s for s in data["spans"] if s["attributes"]["target"] == data["run_error_span_key"]),
            None,
        )
        if run_error_span is not None:
            current_span.set_status(trace.StatusCode.ERROR)
        else:
            current_span.set_status(trace.StatusCode.OK)

        build_spans_for_parent(tracer, data["spans"], data["traceId"], parent_id=None)
