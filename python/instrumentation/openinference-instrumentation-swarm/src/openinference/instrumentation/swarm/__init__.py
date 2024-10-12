from enum import Enum
from inspect import signature
from logging import getLogger
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import AttributeValue
from wrapt import FunctionWrapper, wrap_object

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_attributes_from_context,
    safe_json_dumps,
)
from openinference.instrumentation.swarm.package import _instruments
from openinference.instrumentation.swarm.version import __version__
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = getLogger(__name__)


_SWARM_METHOD_NAMES = (
    "get_chat_completion",
    "handle_function_result",
    "handle_tool_calls",
    "run_and_stream",
)


class SwarmInstrumentor(BaseInstrumentor):  # type: ignore
    """
    OpenInference Instrumentor for Swarm
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )
        for method_name in _SWARM_METHOD_NAMES:
            wrap_object(
                module="swarm",
                name=f"Swarm.{method_name}",
                factory=FunctionWrapper,
                args=(_Wrapper(self._tracer),),
            )
        for function_name in ("run_demo_loop", "run_demo_loop_iteration"):
            wrap_object(
                module="swarm.repl.repl",
                name=function_name,
                factory=FunctionWrapper,
                args=(_Wrapper(self._tracer, ignore_errors=(KeyboardInterrupt,)),),
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        from swarm import Swarm

        for method_name in _SWARM_METHOD_NAMES:
            if (method := getattr(Swarm, method_name, None)) is not None and (
                wrapped_method := getattr(method, "__wrapped__", None)
            ) is not None:
                setattr(Swarm, method_name, wrapped_method)


class _Wrapper:
    """
    Wrapper for functions and class methods that will be considered generic chain spans.
    """

    def __init__(
        self,
        tracer: trace_api.Tracer,
        ignore_errors: Optional[Sequence[Type[Exception]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer
        self._is_method = False
        self._ignore_errors = ignore_errors or tuple([])

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        span_name = (
            f"{instance.__class__.__name__}.{wrapped.__name__}"
            if instance is not None
            else wrapped.__name__
        )
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        **dict(_input_value_and_mime_type(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except self._ignore_errors:
                return
            span.set_attributes(
                dict(
                    _flatten(
                        {
                            **dict(_output_value_and_mime_type(response)),
                        }
                    )
                )
            )
        return response


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


def _input_value_and_mime_type(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _output_value_and_mime_type(response: Any) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_VALUE, safe_json_dumps(response)
    yield OUTPUT_MIME_TYPE, JSON


def _bind_arguments(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    sig = signature(function)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


JSON = OpenInferenceMimeTypeValues.JSON.value
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
