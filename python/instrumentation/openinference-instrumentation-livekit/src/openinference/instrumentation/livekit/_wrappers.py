import logging
from typing import Any, Callable, Mapping, Tuple

from openinference.semconv.trace import OpenInferenceMimeTypeValues
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, get_value
from opentelemetry.trace import Status, StatusCode

from openinference.instrumentation import (
    OITracer,
    get_input_attributes,
    get_output_attributes,
    get_span_kind_attributes,
)
from openinference.instrumentation.livekit._targets import LiveKitWrapTarget

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _make_async_wrapper(tracer: OITracer, target: LiveKitWrapTarget) -> Callable[..., Any]:
    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        span = tracer.start_span(target.span_name)
        try:
            with trace_api.use_span(
                span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            ):
                try:
                    attributes = get_span_kind_attributes(target.span_kind)
                    if target.input_extractor is not None:
                        input_value = target.input_extractor(args, kwargs)
                        if input_value is not None:
                            attributes.update(
                                get_input_attributes(
                                    input_value,
                                    mime_type=OpenInferenceMimeTypeValues.JSON,
                                )
                            )
                    span.set_attributes(attributes)
                except Exception:
                    logger.exception("Failed to set span attributes for %s", target.span_name)

            result = await wrapped(*args, **kwargs)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            if target.capture_output:
                try:
                    span.set_attributes(get_output_attributes(result))
                except Exception:
                    logger.exception("Failed to set output attributes for %s", target.span_name)
            return result
        finally:
            span.end()

    return wrapper
