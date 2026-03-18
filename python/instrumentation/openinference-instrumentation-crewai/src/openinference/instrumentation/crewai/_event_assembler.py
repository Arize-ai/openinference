from __future__ import annotations

import logging
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import Context

from openinference.instrumentation import (
    get_attributes_from_context,
    get_output_attributes,
)
from openinference.instrumentation.crewai._wrappers import _flatten
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_FINISHED_CONTEXT_CACHE_SIZE = 1024


@dataclass(frozen=True)
class _SpanStartSpec:
    name: str
    span_kind: OpenInferenceSpanKindValues
    attributes: dict[str, Any] = field(default_factory=dict)
    remember_as_agent: bool = False


@dataclass(frozen=True)
class _SpanEndSpec:
    output: Any = None
    error: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _AgentHintKeys:
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_key: Optional[str] = None
    agent_role: Optional[str] = None


@dataclass(frozen=True)
class _ContextHint:
    event_id: str
    context: Context


@dataclass
class _SpanEntry:
    span: trace_api.Span
    context: Context
    context_attributes: dict[str, Any]
    agent_hint_keys: Optional[_AgentHintKeys] = None


@dataclass
class _DeferredEnd:
    output: Any
    error: Optional[str]
    attributes: dict[str, Any]
    end_time_ns: Optional[int]


class CrewAIEventAssembler:
    """Assembles CrewAI start/end events into OpenTelemetry spans."""

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer
        self._spans: dict[str, _SpanEntry] = {}
        self._transparent_contexts: dict[str, Context] = {}
        self._finished_contexts: "OrderedDict[str, Context]" = OrderedDict()
        self._task_contexts_by_task_id: "OrderedDict[str, _ContextHint]" = OrderedDict()
        self._task_scope_keys_by_event_id: dict[str, str] = {}
        self._agent_contexts_by_task_id: "OrderedDict[str, _ContextHint]" = OrderedDict()
        self._agent_contexts_by_agent_id: "OrderedDict[str, _ContextHint]" = OrderedDict()
        self._agent_contexts_by_agent_key: "OrderedDict[str, _ContextHint]" = OrderedDict()
        self._agent_contexts_by_agent_role: "OrderedDict[str, _ContextHint]" = OrderedDict()
        self._deferred_ends: "OrderedDict[str, _DeferredEnd]" = OrderedDict()
        self._pending_starts: "OrderedDict[str, list[Callable[[], None]]]" = OrderedDict()
        self._pending_start_event_ids: deque[str] = deque()
        self._pending_starts_draining = False
        self._closed_transparent_scopes: "OrderedDict[str, None]" = OrderedDict()
        self._lock = threading.RLock()

    def start_span(self, event: Any, spec: _SpanStartSpec) -> None:
        context_attributes = dict(get_attributes_from_context())

        def start(parent_context: Context) -> None:
            span_attributes = dict(
                _flatten({SpanAttributes.OPENINFERENCE_SPAN_KIND: spec.span_kind})
            )
            span_attributes.update(spec.attributes)

            span = self._tracer.start_span(
                name=spec.name,
                context=parent_context,
                attributes=span_attributes,
                start_time=self._get_start_time_ns(event),
                record_exception=False,
                set_status_on_exception=False,
            )
            span_context = trace_api.set_span_in_context(span, parent_context)

            event_id = getattr(event, "event_id", None)
            if not event_id:
                span.end()
                return

            with self._lock:
                agent_hint_keys = None
                if spec.remember_as_agent:
                    agent_hint_keys = self._remember_agent_context_locked(
                        event, span_context, event_id
                    )
                span_entry = _SpanEntry(
                    span=span,
                    context=span_context,
                    context_attributes=context_attributes,
                    agent_hint_keys=agent_hint_keys,
                )
                self._spans[event_id] = span_entry
                deferred_end = self._deferred_ends.pop(event_id, None)
                if deferred_end is not None:
                    self._spans.pop(event_id, None)
                    if span_entry.agent_hint_keys is not None:
                        self._forget_agent_context_locked(event_id, span_entry.agent_hint_keys)
                    self._remember_finished_context_locked(event_id, span_entry.context)
                else:
                    deferred_end = None

            if deferred_end is not None:
                self._finalize_span(
                    span_entry,
                    _SpanEndSpec(
                        output=deferred_end.output,
                        error=deferred_end.error,
                        attributes=deferred_end.attributes,
                    ),
                    deferred_end.end_time_ns,
                )

            self._drain_pending_starts(event_id)

        self._with_resolved_parent_context(event, start)

    def end_span(self, event: Any, spec: _SpanEndSpec) -> None:
        event_id = getattr(event, "started_event_id", None) or getattr(event, "event_id", None)
        if not event_id:
            return

        end_time_ns = self._get_end_time_ns(event)

        with self._lock:
            entry = self._spans.pop(event_id, None)
            if entry is None:
                self._remember_deferred_end_locked(
                    event_id,
                    _DeferredEnd(
                        output=spec.output,
                        error=spec.error,
                        attributes=dict(spec.attributes),
                        end_time_ns=end_time_ns,
                    ),
                )
                return
            if entry.agent_hint_keys is not None:
                self._forget_agent_context_locked(event_id, entry.agent_hint_keys)
            self._remember_finished_context_locked(event_id, entry.context)

        self._finalize_span(entry, spec, end_time_ns)

    def open_scope(self, event: Any) -> None:
        def open_scope(parent_context: Context) -> None:
            event_id = getattr(event, "event_id", None)
            if not event_id:
                return

            with self._lock:
                if event_id in self._closed_transparent_scopes:
                    self._closed_transparent_scopes.pop(event_id, None)
                    self._remember_finished_context_locked(event_id, parent_context)
                else:
                    self._transparent_contexts[event_id] = parent_context
                    if (task_key := self._get_task_hint_key(event)) is not None:
                        self._task_scope_keys_by_event_id[event_id] = task_key
                        self._remember_context_hint_locked(
                            self._task_contexts_by_task_id,
                            task_key,
                            parent_context,
                            event_id,
                        )

            self._drain_pending_starts(event_id)

        self._with_resolved_parent_context(event, open_scope)

    def close_scope(self, event: Any) -> None:
        event_id = getattr(event, "started_event_id", None) or getattr(event, "event_id", None)
        if not event_id:
            return

        with self._lock:
            context = self._transparent_contexts.pop(event_id, None)
            task_key = self._task_scope_keys_by_event_id.pop(event_id, None)
            if context is not None:
                if task_key is not None:
                    self._forget_context_hint_locked(
                        self._task_contexts_by_task_id, task_key, event_id
                    )
                self._remember_finished_context_locked(event_id, context)
                return
            if task_key is not None:
                self._forget_context_hint_locked(self._task_contexts_by_task_id, task_key, event_id)
            self._remember_closed_scope_locked(event_id)

    def shutdown(self) -> None:
        with self._lock:
            for entry in self._spans.values():
                try:
                    entry.span.set_status(trace_api.StatusCode.ERROR)
                    entry.span.end()
                except Exception:
                    logger.debug("Failed to end open span during shutdown", exc_info=True)

            self._spans.clear()
            self._transparent_contexts.clear()
            self._finished_contexts.clear()
            self._task_contexts_by_task_id.clear()
            self._task_scope_keys_by_event_id.clear()
            self._agent_contexts_by_task_id.clear()
            self._agent_contexts_by_agent_id.clear()
            self._agent_contexts_by_agent_key.clear()
            self._agent_contexts_by_agent_role.clear()
            self._deferred_ends.clear()
            self._pending_starts.clear()
            self._pending_start_event_ids.clear()
            self._pending_starts_draining = False
            self._closed_transparent_scopes.clear()

    @staticmethod
    def _to_time_ns(value: Any) -> Optional[int]:
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            return None
        value = value.astimezone(timezone.utc)
        return int(value.timestamp() * 1_000_000_000)

    def _get_start_time_ns(self, event: Any) -> Optional[int]:
        return self._to_time_ns(getattr(event, "timestamp", None))

    def _get_end_time_ns(self, event: Any) -> Optional[int]:
        for attr_name in ("finished_at", "timestamp"):
            if (time_ns := self._to_time_ns(getattr(event, attr_name, None))) is not None:
                return time_ns
        return None

    def _finalize_span(
        self,
        entry: _SpanEntry,
        spec: _SpanEndSpec,
        end_time_ns: Optional[int],
    ) -> None:
        span = entry.span
        try:
            if spec.error:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, spec.error))
            else:
                span.set_status(trace_api.StatusCode.OK)
                if spec.output is not None:
                    span.set_attributes(dict(get_output_attributes(spec.output)))
            if spec.attributes:
                span.set_attributes(spec.attributes)
            if entry.context_attributes:
                span.set_attributes(entry.context_attributes)
        finally:
            span.end(end_time_ns)

    def _remember_finished_context_locked(self, event_id: str, context: Context) -> None:
        self._finished_contexts.pop(event_id, None)
        self._finished_contexts[event_id] = context
        while len(self._finished_contexts) > _FINISHED_CONTEXT_CACHE_SIZE:
            self._finished_contexts.popitem(last=False)

    def _remember_deferred_end_locked(self, event_id: str, deferred_end: _DeferredEnd) -> None:
        self._deferred_ends.pop(event_id, None)
        self._deferred_ends[event_id] = deferred_end
        while len(self._deferred_ends) > _FINISHED_CONTEXT_CACHE_SIZE:
            self._deferred_ends.popitem(last=False)

    def _remember_closed_scope_locked(self, event_id: str) -> None:
        self._closed_transparent_scopes.pop(event_id, None)
        self._closed_transparent_scopes[event_id] = None
        while len(self._closed_transparent_scopes) > _FINISHED_CONTEXT_CACHE_SIZE:
            self._closed_transparent_scopes.popitem(last=False)

    @staticmethod
    def _normalize_hint_key(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _get_task_hint_key(self, event: Any) -> Optional[str]:
        task = getattr(event, "task", None)
        return self._normalize_hint_key(
            getattr(task, "id", None) or getattr(event, "task_id", None)
        )

    def _remember_context_hint_locked(
        self,
        cache: "OrderedDict[str, _ContextHint]",
        value: Any,
        context: Context,
        event_id: str,
    ) -> None:
        if not (key := self._normalize_hint_key(value)):
            return
        cache.pop(key, None)
        cache[key] = _ContextHint(event_id=event_id, context=context)
        while len(cache) > _FINISHED_CONTEXT_CACHE_SIZE:
            cache.popitem(last=False)

    def _forget_context_hint_locked(
        self,
        cache: "OrderedDict[str, _ContextHint]",
        value: Any,
        event_id: str,
    ) -> None:
        if not (key := self._normalize_hint_key(value)):
            return
        if (hint := cache.get(key)) is not None and hint.event_id == event_id:
            cache.pop(key, None)

    def _remember_agent_context_locked(
        self,
        event: Any,
        context: Context,
        event_id: str,
    ) -> _AgentHintKeys:
        task = getattr(event, "task", None)
        agent = getattr(event, "agent", None)
        agent_info = getattr(event, "agent_info", None) or {}

        hint_keys = _AgentHintKeys(
            task_id=self._normalize_hint_key(
                getattr(task, "id", None) or getattr(event, "task_id", None)
            ),
            agent_id=self._normalize_hint_key(
                getattr(agent, "id", None)
                or getattr(event, "agent_id", None)
                or agent_info.get("id")
            ),
            agent_key=self._normalize_hint_key(
                getattr(agent, "key", None)
                or getattr(event, "agent_key", None)
                or agent_info.get("key")
            ),
            agent_role=self._normalize_hint_key(
                getattr(agent, "role", None)
                or getattr(event, "agent_role", None)
                or agent_info.get("role")
            ),
        )

        self._remember_context_hint_locked(
            self._agent_contexts_by_task_id,
            hint_keys.task_id,
            context,
            event_id,
        )
        self._remember_context_hint_locked(
            self._agent_contexts_by_agent_id,
            hint_keys.agent_id,
            context,
            event_id,
        )
        self._remember_context_hint_locked(
            self._agent_contexts_by_agent_key,
            hint_keys.agent_key,
            context,
            event_id,
        )
        self._remember_context_hint_locked(
            self._agent_contexts_by_agent_role,
            hint_keys.agent_role,
            context,
            event_id,
        )
        return hint_keys

    def _forget_agent_context_locked(self, event_id: str, hint_keys: _AgentHintKeys) -> None:
        self._forget_context_hint_locked(
            self._agent_contexts_by_task_id, hint_keys.task_id, event_id
        )
        self._forget_context_hint_locked(
            self._agent_contexts_by_agent_id, hint_keys.agent_id, event_id
        )
        self._forget_context_hint_locked(
            self._agent_contexts_by_agent_key, hint_keys.agent_key, event_id
        )
        self._forget_context_hint_locked(
            self._agent_contexts_by_agent_role, hint_keys.agent_role, event_id
        )

    def _get_context_hint(
        self, cache: "OrderedDict[str, _ContextHint]", value: Any
    ) -> Optional[Context]:
        if not (key := self._normalize_hint_key(value)):
            return None
        with self._lock:
            hint = cache.pop(key, None)
            if hint is not None:
                cache[key] = hint
                return hint.context
            return None

    def _get_fallback_parent_context(self, event: Any) -> Optional[Context]:
        event_type = str(getattr(event, "type", "") or "")
        if not (event_type.startswith("tool_usage_") or event_type.startswith("llm_call_")):
            return None

        task_id = getattr(event, "task_id", None)
        for cache in (self._agent_contexts_by_task_id, self._task_contexts_by_task_id):
            context = self._get_context_hint(cache, task_id)
            if context is not None:
                return context

        for cache, value in (
            (self._agent_contexts_by_agent_id, getattr(event, "agent_id", None)),
            (self._agent_contexts_by_agent_key, getattr(event, "agent_key", None)),
            (self._agent_contexts_by_agent_role, getattr(event, "agent_role", None)),
        ):
            context = self._get_context_hint(cache, value)
            if context is not None:
                return context
        return None

    def _get_context_for_event(self, event_id: Optional[str]) -> Optional[Context]:
        if not event_id:
            return None
        with self._lock:
            if (entry := self._spans.get(event_id)) is not None:
                return entry.context
            if (context := self._transparent_contexts.get(event_id)) is not None:
                return context
            context = self._finished_contexts.pop(event_id, None)
            if context is not None:
                self._finished_contexts[event_id] = context
            return context

    def _queue_pending_start(self, parent_event_id: str, callback: Callable[[], None]) -> None:
        with self._lock:
            callbacks = self._pending_starts.setdefault(parent_event_id, [])
            callbacks.append(callback)
            self._pending_starts.move_to_end(parent_event_id)
            while len(self._pending_starts) > _FINISHED_CONTEXT_CACHE_SIZE:
                self._pending_starts.popitem(last=False)

    def _drain_pending_starts(self, event_id: str) -> None:
        with self._lock:
            self._pending_start_event_ids.append(event_id)
            if self._pending_starts_draining:
                return
            self._pending_starts_draining = True

        while True:
            with self._lock:
                if not self._pending_start_event_ids:
                    self._pending_starts_draining = False
                    return
                current_event_id = self._pending_start_event_ids.popleft()
                callbacks = self._pending_starts.pop(current_event_id, [])
            for callback in callbacks:
                try:
                    callback()
                except Exception:
                    logger.debug(
                        "Failed to start pending child span for parent %s",
                        current_event_id,
                        exc_info=True,
                    )

    def _with_resolved_parent_context(
        self, event: Any, callback: Callable[[Context], None]
    ) -> None:
        parent_event_id = getattr(event, "parent_event_id", None)
        if not parent_event_id:
            if (fallback_context := self._get_fallback_parent_context(event)) is not None:
                callback(fallback_context)
                return
            callback(context_api.get_current())
            return

        if (parent_context := self._get_context_for_event(parent_event_id)) is not None:
            callback(parent_context)
            return

        def retry() -> None:
            self._with_resolved_parent_context(event, callback)

        self._queue_pending_start(parent_event_id, retry)
