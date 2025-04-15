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

import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Dict, Optional, cast

from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.react.events import ReActAgentSuccessEvent
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.agents.tool_calling.events import ToolCallingAgentSuccessEvent
from beeai_framework.backend import Role
from beeai_framework.context import RunContext
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError

from openinference.instrumentation import OITracer
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from .utils.build_trace_tree import build_trace_tree
from .utils.create_span import FrameworkSpan, create_span
from .utils.get_serialized_object_safe import get_serialized_object_safe
from .utils.id_name_manager import IdNameManager

try:
    Version = version("beeai-framework")
except PackageNotFoundError:
    Version = "unknown"

id_name_manager = IdNameManager()
active_traces_map: Dict[str, str] = {}


def create_telemetry_middleware(
    tracer: OITracer, main_span_kind: str
) -> Callable[[RunContext], None]:
    def middleware(context: RunContext) -> None:
        trace_obj = getattr(context.emitter, "trace", None)
        trace_id = getattr(trace_obj, "id", None)
        if not trace_id:
            raise FrameworkError("Fatal error. Missing traceId", context=context.__dict__)

        emitter = context.emitter
        run_params = context.run_params
        instance = context.instance

        if trace_id in active_traces_map:
            return
        active_traces_map[trace_id] = type(instance).__name__
        base_path = ".".join(emitter.namespace)
        prompt = None
        if isinstance(instance, BaseAgent):
            prompt = run_params.get("prompt") if isinstance(run_params, dict) else None

        spans_map: Dict[str, FrameworkSpan] = {}
        parent_ids_map: dict[str, int] = {}
        spans_to_delete_map: set[str] = set()
        events_iterations_map: dict[str, dict[str, str]] = {}

        def clean_span_sources(span_id: str) -> None:
            span = spans_map.get(span_id)
            parent_id = span.get("parent_id") if span else None
            if not parent_id:
                return

            span_count = parent_ids_map.get(parent_id)
            if not span_count:
                return

            if span_count > 1:
                parent_ids_map[parent_id] = span_count - 1
            elif span_count == 1:
                parent_ids_map.pop(parent_id)
                if parent_id in spans_to_delete_map:
                    spans_map.pop(parent_id, None)
                    spans_to_delete_map.discard(parent_id)

        history: list[dict[str, Any]] = []
        generated_message: Optional[dict[str, Any]] = None
        group_iterations = []

        start_time_perf = time.time_ns()

        def datetime_to_ns(dt: datetime) -> int:
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            return int((dt - epoch).total_seconds() * 1e9)

        def on_finish(_data: Any, _meta: EventMeta) -> None:
            nonlocal prompt
            try:
                if not prompt and isinstance(instance, BaseAgent):
                    prompt = next(
                        (m.text for m in reversed(instance.memory.messages) if m.role == Role.USER),
                        None,
                    )
                    if not prompt:
                        raise FrameworkError("The prompt must be defined", context=context.__dict__)

                ## This would call your tree-building logic
                build_trace_tree(
                    tracer=tracer,
                    main_span_kind=main_span_kind,
                    data={
                        "prompt": prompt,
                        "history": history,
                        "generatedMessage": generated_message,
                        "spans": list(spans_map.values()),
                        "traceId": trace_id,
                        "version": Version,
                        "startTime": start_time_perf,
                        "endTime": time.time_ns(),
                        "source": active_traces_map.get(trace_id),
                        "run_error_span_key": f"run.{base_path}.error",
                    },
                )
            except Exception as e:
                print("Instrumentation send data error", e)
            finally:
                del active_traces_map[trace_id]

        emitter.match(f"run.{base_path}.finish", on_finish)

        def on_any_event(data: Any, meta: EventMeta) -> None:
            nonlocal context
            if meta.path.startswith("run.") and meta.name != "run.error":
                return

            if meta.name == "new_token":
                return

            if not getattr(meta.trace, "run_id", None):
                raise FrameworkError(
                    f"Fatal error. Missing run_id for event: {meta.path}", context=context.__dict__
                )

            try:
                iteration_event_name = meta.group_id.strip("`") if meta.group_id else meta.group_id
                if (
                    iteration_event_name
                    and not getattr(meta.trace, "parent_run_id", None)
                    and iteration_event_name not in group_iterations
                ):
                    spans_map[iteration_event_name] = create_span(
                        id=iteration_event_name,
                        name=iteration_event_name,
                        target="groupId",
                        data={
                            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value  # noqa: E501
                        },
                        started_at=datetime_to_ns(meta.created_at),
                    )
                    group_iterations.append(iteration_event_name)

                ids = id_name_manager.get_ids(
                    path=meta.path,
                    id=meta.id,
                    run_id=meta.trace.run_id,  # type: ignore
                    parent_run_id=getattr(meta.trace, "parent_run_id", None),
                    group_id=iteration_event_name,
                )

                span_id = ids["spanId"]
                parent_span_id = ids["parentSpanId"]

                serialized_data = get_serialized_object_safe(data, meta)
                # Skip partialUpdate events with no data
                if meta.name == "partial_update" and not serialized_data:
                    return

                span = create_span(
                    id=span_id,
                    name=meta.name,
                    target=meta.path,
                    parent={"id": parent_span_id} if parent_span_id else None,
                    ctx=getattr(meta, "context", None),
                    data=serialized_data,
                    started_at=datetime_to_ns(meta.created_at),
                )
                spans_map[span["context"]["span_id"]] = span

                last_iteration = group_iterations[-1] if group_iterations else None

                # Clean up `partial_update` event if no nested spans
                last_iteration_event_span_id: Optional[str] = (
                    events_iterations_map.get(last_iteration, {}).get(meta.name)
                    if last_iteration is not None
                    else None
                )

                if (
                    last_iteration_event_span_id
                    and meta.name == "partial_update"
                    and last_iteration_event_span_id in spans_map
                ):
                    if span_id and span_id in parent_ids_map:
                        spans_to_delete_map.add(last_iteration_event_span_id)
                    else:
                        clean_span_sources(last_iteration_event_span_id)
                        spans_map.pop(last_iteration_event_span_id, None)

                # Create new span
                spans_map[span["context"]["span_id"]] = span

                # Update parent count
                if span.get("parent_id"):
                    parent_id = span["parent_id"]
                    if parent_id is not None:
                        parent_ids_map[parent_id] = parent_ids_map.get(parent_id, 0) + 1

                # Save the last event for each iteration
                if group_iterations:
                    if last_iteration in events_iterations_map:
                        events_iterations_map[last_iteration][meta.name] = span["context"][
                            "span_id"
                        ]
                    elif last_iteration is not None:
                        events_iterations_map[last_iteration] = {
                            meta.name: span["context"]["span_id"]
                        }

            except Exception as e:
                print("Instrumentation build data error", e)

        emitter.match("*.*", on_any_event)

        def is_success_event(event: Any) -> bool:
            return getattr(event, "name", None) == "success" and isinstance(
                getattr(event, "creator", None), BaseAgent
            )

        def on_success(data: Any, meta: EventMeta) -> None:
            nonlocal generated_message, history
            try:
                if isinstance(meta.creator, ToolCallingAgent):
                    tool_calling_typed_data = cast(ToolCallingAgentSuccessEvent, data)
                    history = [
                        {
                            "text": m.text,
                            "role": m.role.value if hasattr(m.role, "value") else m.role,
                        }
                        for m in tool_calling_typed_data.state.memory.messages
                    ]
                    if (
                        hasattr(tool_calling_typed_data.state, "result")
                        and tool_calling_typed_data.state.result is not None
                    ):
                        result_role = tool_calling_typed_data.state.result.role
                        generated_message = {
                            "role": result_role.value
                            if hasattr(result_role, "value")
                            else result_role,
                            "text": tool_calling_typed_data.state.result.text,
                        }
                if isinstance(meta.creator, ReActAgent):
                    react_agent_typed_data = cast(ReActAgentSuccessEvent, data)
                    tooling_result_role = react_agent_typed_data.data.role
                    generated_message = {
                        "role": tooling_result_role.value
                        if hasattr(tooling_result_role, "value")
                        else tooling_result_role,
                        "text": react_agent_typed_data.data.text,
                    }
                    history = [
                        {
                            "text": m.text,
                            "role": m.role.value if hasattr(m.role, "value") else m.role,
                        }
                        for m in react_agent_typed_data.memory.messages
                    ]
                print("2")
            except Exception as e:
                print("Instrumentation error: failed to extract success message", e)

        emitter.match(is_success_event, on_success)

    return middleware
