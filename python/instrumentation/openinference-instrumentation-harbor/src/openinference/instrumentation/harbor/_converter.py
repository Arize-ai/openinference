"""
Convert ATIF (Agent Trajectory Interchange Format) trajectory JSON to OpenInference/OTel spans.

Produces a 3-level span hierarchy:
  Root AGENT span → LLM spans (one per agent step) → TOOL spans (one per tool call)

Span IDs are deterministic (SHA-256 of session_id + context), so re-converting
the same trajectory always produces the same spans.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_SAMPLED = TraceFlags(TraceFlags.SAMPLED)
_OK = Status(StatusCode.OK)


def _step_role(step: dict[str, Any]) -> str:
    """Get the role from a step, handling both ATIF 'source' and legacy 'role' fields."""
    return str(step.get("source", step.get("role", "")))


def _trace_id(session_id: str) -> int:
    digest = hashlib.sha256(session_id.encode()).digest()
    return int.from_bytes(digest[:16], "big") & ((1 << 128) - 1)


def _span_id(session_id: str, *parts: str) -> int:
    key = "|".join([session_id, *parts])
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 64) - 1)


def _ts_ns(iso_str: str | None, fallback_ns: int) -> int:
    if not iso_str:
        return fallback_ns
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1_000_000_000)
    except (ValueError, AttributeError):
        return fallback_ns


def _make_span(
    name: str,
    trace_id: int,
    span_id: int,
    parent_span_id: int | None,
    attributes: dict[str, Any],
    start_time_ns: int,
    end_time_ns: int,
    resource: Resource,
) -> ReadableSpan:
    context = SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False, trace_flags=_SAMPLED)
    parent = (
        SpanContext(
            trace_id=trace_id, span_id=parent_span_id, is_remote=False, trace_flags=_SAMPLED
        )
        if parent_span_id is not None
        else None
    )
    return ReadableSpan(
        name=name,
        context=context,
        parent=parent,
        resource=resource,
        attributes={k: v for k, v in attributes.items() if v is not None},
        kind=SpanKind.INTERNAL,
        start_time=start_time_ns,
        end_time=end_time_ns,
        status=_OK,
    )


def _build_observation_map(steps: list[dict[str, Any]]) -> dict[str, str]:
    obs: dict[str, str] = {}
    for step in steps:
        if _step_role(step) != "observation":
            continue
        for result in step.get("results", []):
            source_id = result.get("source_call_id")
            if source_id:
                obs[source_id] = result.get("content", "")
    return obs


def convert_trajectory(
    trajectory: dict[str, Any],
    *,
    resource_attributes: dict[str, str] | None = None,
) -> list[ReadableSpan]:
    """Convert an ATIF trajectory dict to a list of OTel ReadableSpan objects."""
    session_id = trajectory.get("session_id", "unknown")
    agent_info = trajectory.get("agent", {})
    agent_name = agent_info.get("name", "unknown_agent")
    agent_model = agent_info.get("model_name", "")
    steps: list[dict[str, Any]] = trajectory.get("steps", [])
    final_metrics = trajectory.get("final_metrics", {})

    tid = _trace_id(session_id)
    now_ns = int(time.time() * 1_000_000_000)
    resource = Resource.create(resource_attributes or {})

    # Find first/last timestamps for root span bounds
    valid_ts = [s["timestamp"] for s in steps if s.get("timestamp")]
    root_start = _ts_ns(valid_ts[0] if valid_ts else None, now_ns)
    root_end = _ts_ns(valid_ts[-1] if valid_ts else None, now_ns + 1_000_000)

    # Extract first user message and last agent message
    first_user_msg = ""
    last_agent_msg = ""
    for step in steps:
        role = _step_role(step)
        content = step.get("message", step.get("content", ""))
        if role == "user" and not first_user_msg:
            first_user_msg = content
        if role in ("assistant", "agent"):
            last_agent_msg = content

    # Root AGENT span
    root_sid = _span_id(session_id, "root")
    root_attrs: dict[str, Any] = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
        SpanAttributes.SESSION_ID: session_id,
        SpanAttributes.AGENT_NAME: agent_name,
        SpanAttributes.INPUT_VALUE: first_user_msg,
        SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
        SpanAttributes.OUTPUT_VALUE: last_agent_msg,
        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
    }
    if agent_model and agent_model != "<synthetic>":
        root_attrs[SpanAttributes.LLM_MODEL_NAME] = agent_model
    prompt_total = final_metrics.get("total_prompt_tokens") or 0
    comp_total = final_metrics.get("total_completion_tokens") or 0
    if prompt_total:
        root_attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = prompt_total
    if comp_total:
        root_attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = comp_total
    if prompt_total + comp_total:
        root_attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = prompt_total + comp_total

    spans: list[ReadableSpan] = [
        _make_span(
            f"{agent_name} trajectory",
            tid,
            root_sid,
            None,
            root_attrs,
            root_start,
            root_end,
            resource,
        )
    ]

    # Each non-empty agent step is a separate LLM call.
    # Empty agent steps (no message, no tool calls) are skipped.
    # Conversation context (pending_msgs) accumulates — each LLM call sees
    # prior user messages AND prior agent responses, mirroring the real prompt.
    obs_map = _build_observation_map(steps)
    pending_msgs: list[dict[str, str]] = []
    llm_index = 0

    # Pre-collect timestamps of non-empty agent steps for end-time calculation.
    # A step's end time = next step's start time (shows real duration including tool waits).
    agent_step_timestamps: list[str | None] = []
    for step in steps:
        role = _step_role(step)
        if role not in ("assistant", "agent"):
            continue
        content = step.get("message", step.get("content", ""))
        if not content and not step.get("tool_calls"):
            continue
        agent_step_timestamps.append(step.get("timestamp"))

    for step in steps:
        role = _step_role(step)
        content = step.get("message", step.get("content", ""))

        if role in ("system", "user"):
            pending_msgs.append({"role": role, "content": content})
            continue

        if role not in ("assistant", "agent"):
            continue

        tool_calls = step.get("tool_calls", [])

        # Skip empty steps (no message and no tool calls)
        if not content and not tool_calls:
            continue

        llm_index += 1
        step_id = step.get("step_id", str(llm_index))
        step_start = _ts_ns(step.get("timestamp"), now_ns + llm_index * 1_000_000)

        # End time = next agent step's start, or start + 1ms if last step
        if llm_index < len(agent_step_timestamps):
            step_end = _ts_ns(agent_step_timestamps[llm_index], step_start + 1_000_000)
        else:
            step_end = _ts_ns(None, step_start + 1_000_000)
        if step_end <= step_start:
            step_end = step_start + 1_000_000

        llm_sid = _span_id(session_id, "llm", str(step_id))
        step_model = step.get("model_name") or agent_model

        input_messages = [
            {"message": {"role": m["role"], "content": m["content"]}} for m in pending_msgs
        ]
        output_messages = [{"message": {"role": "assistant", "content": content}}]
        input_text = "\n".join(f"{m['role']}: {m['content']}" for m in pending_msgs if m["content"])

        llm_attrs: dict[str, Any] = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.INPUT_VALUE: input_text,
            SpanAttributes.OUTPUT_VALUE: content,
            SpanAttributes.LLM_INPUT_MESSAGES: json.dumps(input_messages),
            SpanAttributes.LLM_OUTPUT_MESSAGES: json.dumps(output_messages),
            SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
            SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
        }
        if step_model and step_model != "<synthetic>":
            llm_attrs[SpanAttributes.LLM_MODEL_NAME] = step_model

        step_metrics = step.get("metrics", {})
        if step_metrics.get("prompt_tokens"):
            llm_attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = step_metrics["prompt_tokens"]
        if step_metrics.get("completion_tokens"):
            llm_attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = step_metrics["completion_tokens"]
        if step_metrics.get("cached_tokens"):
            llm_attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] = step_metrics[
                "cached_tokens"
            ]

        spans.append(
            _make_span(
                f"step {llm_index}",
                tid,
                llm_sid,
                root_sid,
                llm_attrs,
                step_start,
                step_end,
                resource,
            )
        )

        # Add this step's output to conversation context for subsequent steps
        if content:
            pending_msgs.append({"role": "assistant", "content": content})

        for tc_idx, tc in enumerate(tool_calls):
            func_name = tc.get("function_name", tc.get("name", "unknown_tool"))
            tc_id = tc.get("tool_call_id", f"{step_id}_tc_{tc_idx}")
            arguments = tc.get("arguments", {})
            args_json = json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
            observation = obs_map.get(tc_id, "")

            tool_sid = _span_id(session_id, "tool", str(step_id), str(tc_idx))
            tool_start = step_start + (tc_idx + 1) * 100_000
            tool_attrs: dict[str, Any] = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: (OpenInferenceSpanKindValues.TOOL.value),
                SpanAttributes.TOOL_NAME: func_name,
                SpanAttributes.TOOL_PARAMETERS: args_json,
                SpanAttributes.INPUT_VALUE: args_json,
                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                SpanAttributes.OUTPUT_VALUE: observation,
                SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
            }
            spans.append(
                _make_span(
                    func_name,
                    tid,
                    tool_sid,
                    llm_sid,
                    tool_attrs,
                    tool_start,
                    tool_start + 100_000,
                    resource,
                )
            )

            # Tool results feed into subsequent LLM calls too
            if observation:
                pending_msgs.append({"role": "tool", "content": f"{func_name}: {observation}"})

    return spans


def convert_trajectory_file(
    path: str | Path,
    *,
    resource_attributes: dict[str, str] | None = None,
) -> list[ReadableSpan]:
    """Read an ATIF trajectory JSON file and convert to OTel spans."""
    with open(path) as f:
        trajectory = json.load(f)
    return convert_trajectory(trajectory, resource_attributes=resource_attributes)


def convert_trajectory_dir(
    dir_path: str | Path,
    *,
    resource_attributes: dict[str, str] | None = None,
) -> list[ReadableSpan]:
    """Convert all .json trajectory files in a directory."""
    all_spans: list[ReadableSpan] = []
    for json_file in sorted(Path(dir_path).glob("*.json")):
        all_spans.extend(
            convert_trajectory_file(json_file, resource_attributes=resource_attributes)
        )
    return all_spans
