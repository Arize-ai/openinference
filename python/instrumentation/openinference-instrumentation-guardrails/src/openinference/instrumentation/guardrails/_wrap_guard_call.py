import json
from abc import ABC
from typing import Any, Callable, Iterator, List, Mapping, Tuple
from enum import Enum
import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from opentelemetry import context as otel_context

def _flatten(mapping: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, List) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value

class _WithTracer(ABC):
    """
    Base class for wrappers that need a tracer. Acts as a trait for the wrappers
    """

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

class _GuardCallWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        
        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)
        
        span_name = "guard"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        "guard.invocation.parameters": safe_json_dumps(invocation_parameters),
                    }
                )
            ),
        ) as span:
            print(f"GuardCallWrapper span: {span}")
            print(f"GuardCallWrapper spanContext: {span.get_span_context()}")
            span.set_attributes(dict(get_attributes_from_context()))
            span.set_attribute("openinference.span.kind", "guardrail")
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class _PromptCallableWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        
        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)
        
        span_name = "invoke_llm"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        "invoke_llm.invocation.parameters": safe_json_dumps(invocation_parameters),
                    }
                )
            ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            span.set_attribute("openinference.span.kind", "llm")
            print(f"LLM Invoke span: {span}")
            print(f"LLM Invoke TraceID: {span.get_span_context()}")
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class _ParseCallableWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        
        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)
        
        span_name = "guard_parse"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        "__parse.invocation.parameters": safe_json_dumps(invocation_parameters),
                    }
                )
            ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            span.set_attribute("openinference.span.kind", "guardrail")
            print(f"PromptCallableWrapper span: {span}")
            print(f"PromptCallableWrapper TraceID: {span.get_span_context()}")
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response

class _PostValidationWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        
        # Prepare invocation parameters by merging args and kwargs
        invocation_parameters = {}
        for arg in args:
            if arg and isinstance(arg, dict):
                invocation_parameters.update(arg)
        invocation_parameters.update(kwargs)
        
        span_name = "post_validation"
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        "post_validation.invocation.parameters": safe_json_dumps(invocation_parameters),
                    }
                )
            ),
        ) as span:
            span.set_attributes(dict(get_attributes_from_context()))
            span.set_attribute("openinference.span.kind", "guardrail")
            print(f"post validation span: {span}")
            print(f"post validation TraceID: {span.get_span_context()}")
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
        return response