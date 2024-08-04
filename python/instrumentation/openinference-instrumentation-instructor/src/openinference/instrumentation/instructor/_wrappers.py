import json
from enum import Enum
from inspect import signature
from instructor.utils import is_async
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue


class SafeJSONEncoder(json.JSONEncoder):
    """
    Safely encodes non-JSON-serializable objects.
    """

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            if hasattr(o, "dict") and callable(o.dict):  # pydantic v1 models, e.g., from Cohere
                return o.dict()
            return repr(o)


def _flatten(mapping: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    if not mapping:
        return
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


def _get_input_value(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """
    Parses a method call's inputs into a JSON string. Ensures a consistent
    output regardless of whether the those inputs are passed as positional or
    keyword arguments.
    """

    # For typical class methods, the corresponding instance of inspect.Signature
    # does not include the self parameter. However, the inspect.Signature
    # instance for __call__ does include the self parameter.
    method_signature = signature(method)
    first_parameter_name = next(iter(method_signature.parameters), None)
    signature_contains_self_parameter = first_parameter_name in ["self"]
    bound_arguments = method_signature.bind(
        *(
            [None]  # the value bound to the method's self argument is discarded below, so pass None
            if signature_contains_self_parameter
            else []  # no self parameter, so no need to pass a value
        ),
        *args,
        **kwargs,
    )
    return safe_json_dumps(
        {
            **{
                argument_name: argument_value
                for argument_name, argument_value in bound_arguments.arguments.items()
                if argument_name not in ["self", "kwargs"]
            },
            **bound_arguments.arguments.get("kwargs", {}),
        },
        cls=SafeJSONEncoder,
    )


class _PatchWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        create = kwargs.get("create")
        client = kwargs.get("client")

        if create is not None:
            func = create
        elif client is not None:
            func = client.chat.completions.create
        else:
            raise ValueError("Either client or create must be provided")
        new_func = wrapped(*args, **kwargs)
        func_is_async = is_async(func)

        def patched_new_func(*args, **kwargs):
            span_name = "instructor.patch"
            with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL,
                            INPUT_VALUE_MIME_TYPE: "application/json",
                            # TODO(harrison): figure out why i cant use args with _get_input_value
                            INPUT_VALUE: kwargs,
                        }
                    )
                ),
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                resp = new_func(*args, **kwargs)
                if resp is not None and hasattr(resp, "dict"):
                    span.set_attribute(OUTPUT_VALUE, json.dumps(resp.dict()))
                    span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
                return resp

        async def async_patched_new_func(*args, **kwargs):
            span_name = "instructor.async_patch"
            with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL,
                            INPUT_VALUE_MIME_TYPE: "application/json",
                            # TODO(harrison): figure out why i cant use args with _get_input_value
                            INPUT_VALUE: kwargs,
                        }
                    )
                ),
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                resp = await new_func(*args, **kwargs)
                if resp is not None and hasattr(resp, "dict"):
                    span.set_attribute(OUTPUT_VALUE, json.dumps(resp.dict()))
                    span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
                return resp

        new_create = async_patched_new_func if func_is_async else patched_new_func

        if client is not None:
            client.chat.completions.create = new_create
            return client
        else:
            return new_create


class _HandleResponseWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if instance:
            span_name = f"{instance.__class__.__name__}.{wrapped.__name__}"
        else:
            span_name = wrapped.__name__
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL,
                        INPUT_VALUE_MIME_TYPE: "application/json",
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
                response_model = response[0]
                response_model_name = (
                    response_model.__name__
                    if response_model is not None and hasattr(response_model, "__name__")
                    else "Unknown"
                )
                if response_model is not None and hasattr(response_model, "model_json_schema"):
                    span.set_attribute(OUTPUT_VALUE, json.dumps(response_model.model_json_schema()))
                    span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(OUTPUT_VALUE, response)
        return response


INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_VALUE_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
