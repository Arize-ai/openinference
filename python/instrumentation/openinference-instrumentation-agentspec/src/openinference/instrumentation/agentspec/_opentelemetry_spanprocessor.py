import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from opentelemetry import context as context_api
from opentelemetry.sdk.trace import Event as OtelSdkEvent
from opentelemetry.sdk.trace import Resource as OtelSdkResource  # type: ignore
from opentelemetry.sdk.trace import Span as OtelSdkSpan
from opentelemetry.sdk.trace import SpanContext as OtelSdkSpanContext  # type: ignore
from opentelemetry.sdk.trace import sampling as otel_sdk_sampling
from opentelemetry.sdk.trace.export import BatchSpanProcessor as OtelSdkBatchSpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor as OtelSdkSimpleSpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter as OtelSdkSpanExporter
from opentelemetry.sdk.trace.export import SpanProcessor as OtelSdkSpanProcessor  # type: ignore
from opentelemetry.trace import TraceFlags as OtelSdkTraceFlags
from opentelemetry.util.types import Attributes
from pyagentspec.tracing.events.event import Event as AgentSpecEvent
from pyagentspec.tracing.spanprocessor import SpanProcessor as AgentSpecSpanProcessor
from pyagentspec.tracing.spans.span import Span as AgentSpecSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


_MAX_UNSIGNED_LONG_LONG = 18446744073709551615


def _try_id_to_int_conversion(id_: Optional[str]) -> int:
    """
    Convert the string ID into an integer ID as per OpenTelemetry requirement

    https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/trace/api.md#spancontext
    """
    if id_ is not None:
        try:
            # First we try to get the original UUID, if the id is a compliant string
            id_ = uuid.UUID(id_)  # type: ignore
        except ValueError:
            # If it is not, we try to convert the id as it is
            pass
    try:
        # We try the conversion to int modulo 2**64 (unsigned long long, due to otel limitations),
        # if it is not None. Note also the 0 is not a valid ID,
        # so we reduce the modulo by 1 (2**64-1) and add 1 to the result
        return (int(id_) % _MAX_UNSIGNED_LONG_LONG) + 1 if id_ else id(id_)
    except Exception:
        # If the conversion fails, we fall back to the id, we cannot do better
        return id(id_)


def _try_json_serialization(value: Any) -> str:
    """
    Serialize the given object into the corresponding JSON string.
    If it is not JSON serializable, we simply stringify it
    """
    if isinstance(value, str):
        # Avoid quoting strings with json dump
        return str(value)
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def _flatten_attribute_dict(attribute: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the complex types using OpenTelemetry conventions"""
    flattened_attribute: Dict[str, Any] = {}
    for key, value in attribute.items():
        if isinstance(value, (tuple, set, list)):
            # List-like values have to become dictionaries with the index as the key
            # This will be translated according to dict rules below
            # (e.g., attribute_name.0, attribute_name.1, ...)
            flattened_value = dict()
            for i, inner_value in enumerate(value):
                flattened_value[i] = inner_value
            value = flattened_value
        if isinstance(value, dict):
            # Dictionary attributes are flattened by adding each dict entry
            # as a separate entry with name `attribute_name.key`
            for inner_key, inner_value in _flatten_attribute_dict(value).items():
                flattened_attribute[f"{key}.{inner_key}"] = inner_value
        else:
            flattened_attribute[key] = value
    return flattened_attribute


def _serialize_attribute_values(attributes: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize the attributes dictionary using types that OpenTelemetry supports"""
    # Note that this function performs shallow serialization, i.e., it does not apply recursively
    allowed_types = (str, int, float, bool, bytes)
    return {
        key: value if isinstance(value, allowed_types) else _try_json_serialization(value)
        for key, value in _flatten_attribute_dict(attributes).items()
    }


class _OtelSdkEvent(OtelSdkEvent):
    """Protected implementation of `opentelemetry.trace.Event`.

    Contains an ID that uniquely identifies the event.
    """

    def __init__(
        self,
        id: str,
        name: str,
        attributes: Attributes = None,
        timestamp: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, attributes=attributes, timestamp=timestamp)
        self.id = id


class _OtelSdkSpan(OtelSdkSpan):
    """Protected implementation of `opentelemetry.trace.Span`.

    As Spans cannot be directly instantiated in the opentelemetry SDK,
    we create a materialization of the interface that we use internally.
    This is the same solution that opentelemetry uses to prevent anyone
    from creating a span without a tracer provider, but we need more flexibility.
    """

    def update_events(self, events: list[AgentSpecEvent]) -> None:
        # We update the events list by adding the missing ones
        # These events should all be _OtelSdkEvent that contain an ID
        # If they are not, we can ignore them because they don't come from us
        current_event_ids = {event.id for event in self.events if hasattr(event, "id")}
        for event in events:
            if event.id not in current_event_ids:
                self._add_event(convert_agentspec_event_into_otel_event(event))


def convert_agentspec_event_into_otel_event(
    event: AgentSpecEvent,
    mask_sensitive_information: bool = True,
) -> _OtelSdkEvent:
    """
    Convert an AgentSpec Event into the equivalent OpenTelemetry Event object.

    Parameters
    ----------
    event:
        the event to convert
    mask_sensitive_information
        Whether to mask potentially sensitive information from the event attributes

    Returns
    -------
    opentelemetry.sdk.trace.Event
        The converted event
    """
    try:
        event_dump = event.model_dump(mask_sensitive_information=mask_sensitive_information)
    except Exception as e:
        # If anything goes wrong, we log it, and we go on without event attributes
        logger.warning(f"Error during the dump of an event: {e}")
        event_dump = {}
    return _OtelSdkEvent(
        id=event.id,
        name=event.name or event.__class__.__name__,
        timestamp=event.timestamp,
        attributes=_serialize_attribute_values(event_dump),
    )


def convert_agentspec_span_into_otel_span(
    span: AgentSpecSpan,
    resource: Optional[OtelSdkResource] = None,
    mask_sensitive_information: bool = True,
    span_model_dump: Optional[Callable[[AgentSpecSpan, bool], Dict[str, Any]]] = None,
    span_to_update: Optional[_OtelSdkSpan] = None,
) -> _OtelSdkSpan:
    """
    Convert a AgentSpec Span into the equivalent OpenTelemetry Span object.

    Parameters
    ----------
    span:
        the span to convert
    resource:
        the OpenTelemetry Resource object to use
    mask_sensitive_information
        Whether to mask potentially sensitive information from the span and its events
    span_model_dump
        Function to dump the span model to a dictionary.
        If None, the pydantic model_dump function will be used.
    span_to_update
        OpenTelemetry Span object to be updated with new information from Agent Spec Span.
        If None, a new OpenTelemetry Span object will be created.

    Returns
    -------
    _OtelSdkSpan
        The converted span
    """
    try:
        if span_model_dump:
            span_attributes = span_model_dump(span, mask_sensitive_information)
        else:
            span_attributes = span.model_dump(mask_sensitive_information=mask_sensitive_information)
    except Exception as e:
        # If anything goes wrong, we log it, and we go on without span attributes
        logger.warning(f"Error during the dump of a span: {e}")
        span_attributes = {}
    # We remove the tracing information we don't want to appear in the attributes,
    # as they will be handled separately
    for attribute_to_pop in (
        "events",
        "id",
        "type",
        "name",
        "start_time",
        "end_time",
    ):
        span_attributes.pop(attribute_to_pop, None)
    # We create the objects required by OpenTelemetry with the expected information
    sampling_result = otel_sdk_sampling.SamplingResult(
        otel_sdk_sampling.Decision.RECORD_AND_SAMPLE,
        span_attributes,
    )
    trace_flags = (
        OtelSdkTraceFlags(OtelSdkTraceFlags.SAMPLED)
        if sampling_result.decision.is_sampled()  # type: ignore
        else OtelSdkTraceFlags(OtelSdkTraceFlags.DEFAULT)
    )
    # The IDs in otel are required to be integers, so we try to transform them
    trace_id = _try_id_to_int_conversion(span._trace.id if span._trace else None)
    span_id = _try_id_to_int_conversion(span.id)
    span_name = span.name or span.__class__.__name__
    span_attributes = _serialize_attribute_values(span_attributes)
    if span_to_update:
        span_to_update.set_attributes(span_attributes)
        span_to_update.update_events(span.events)
        return span_to_update
    else:
        return _OtelSdkSpan(
            name=span_name,
            context=OtelSdkSpanContext(
                trace_id=trace_id,
                span_id=span_id,
                is_remote=False,
                trace_flags=trace_flags,
                trace_state=sampling_result.trace_state,
            ),
            parent=(
                OtelSdkSpanContext(
                    trace_id=trace_id,
                    span_id=_try_id_to_int_conversion(span._parent_span.id),
                    is_remote=False,
                )
                if span._parent_span
                else None
            ),
            resource=resource,
            attributes=span_attributes,
            events=[convert_agentspec_event_into_otel_event(event) for event in span.events],
        )


class _OtelSpanProcessor(AgentSpecSpanProcessor, ABC):
    """AgentSpec wrapper for the OpenTelemetry SpanProcessor"""

    def __init__(
        self,
        otel_span_processor: Optional[OtelSdkSpanProcessor] = None,
        span_exporter: Optional[OtelSdkSpanExporter] = None,
        resource: Optional[OtelSdkResource] = None,
        mask_sensitive_information: bool = True,
        span_model_dump_func: Optional[Callable[[AgentSpecSpan, bool], Dict[str, Any]]] = None,
    ) -> None:
        """
        AgentSpec wrapper for the OpenTelemetry SpanProcessor.

        This class forwards the calls to AgentSpec's span processors to an OpenTelemetry one.

        Parameters
        ----------
        otel_span_processor:
            The OpenTelemetry SpanProcessor to use to process spans.
            If None is given, a new instance is created.
        span_exporter:
            The OpenTelemetry SpanExporter to use to export spans.
            This parameter is ignored if an ``otel_span_processor`` is provided.
        resource:
            The OpenTelemetry Resource to use in Spans.
        mask_sensitive_information
            Whether to mask potentially sensitive information from the span and its events
        span_model_dump_func
            Function to dump the span model to a dictionary, used to get attributes in the
            transformation to OTel Span. The function accepts two parameters: the Agent Spec Span,
            and the mask_sensitive_information boolean flag.
            If None, the pydantic model_dump function on the Agent Spec Span will be used.
        """
        super().__init__(mask_sensitive_information=mask_sensitive_information)
        self.span_exporter = span_exporter
        self.resource = resource
        self.span_model_dump_func = span_model_dump_func
        if otel_span_processor is not None:
            self.span_processor = otel_span_processor
        elif span_exporter is not None:
            self.span_processor = self._create_otel_span_processor(span_exporter=span_exporter)
        else:
            raise ValueError("Span exporter must be provided when no span processor is given")
        # We keep a registry of spans that we already created
        # The idea is to reuse and update the same span at start and end
        self._span_registry: Dict[str, _OtelSdkSpan] = {}

    @abstractmethod
    def _create_otel_span_processor(
        self, span_exporter: OtelSdkSpanExporter
    ) -> OtelSdkSpanProcessor:
        pass

    def _create_otel_span_from_agentspec_span(self, span: AgentSpecSpan) -> OtelSdkSpan:
        # We retrieve the span from the registry, and we update it with the new information.
        # If there's no otel span in the registry with this ID, it is created
        span_from_registry = self._span_registry.get(span.id, None)
        otel_span = convert_agentspec_span_into_otel_span(
            span=span,
            resource=self.resource,
            mask_sensitive_information=self.mask_sensitive_information,
            span_model_dump=self.span_model_dump_func,
            span_to_update=span_from_registry,
        )
        # Record the updated span in the registry and return it
        self._span_registry[span.id] = otel_span
        return otel_span

    def on_start(self, span: AgentSpecSpan) -> None:
        """
        Method called at the start of a Span.

        Parameters
        ----------
        span:
            The AgentSpec span that is starting
        """
        try:
            otel_span = self._create_otel_span_from_agentspec_span(span=span)
            # If instrumentation is suppressed, we still convert the span,
            # but we don't emit anything
            if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
                return
            otel_span.start(start_time=span.start_time)
            self.span_processor.on_start(span=otel_span)
        except Exception as e:
            # Whatever happens we do not crash the execution of the assistant,
            # but we warn the user
            logger.warning(f"Exception raised during SpanProcessor `on_start`: {e}")

    def on_end(self, span: AgentSpecSpan) -> None:
        """
        Method called at the end of a Span.

        Parameters
        ----------
        span:
            The AgentSpec span that is ending
        """
        try:
            # We re-create the span because we might need to re-format
            # the changes happened in the span
            otel_span = self._create_otel_span_from_agentspec_span(span=span)
            # If instrumentation is suppressed, we manage the converted spans statuses,
            # but we don't emit anything
            if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
                return
            if otel_span.start_time is None:
                # If the span was not started for some reason, we start it manually here
                otel_span.start(start_time=span.start_time)
            otel_span.end(end_time=span.end_time)
            self.span_processor.on_end(span=otel_span)
        except Exception as e:
            # Whatever happens we do not crash the execution of the assistant,
            # but we warn the user
            logger.warning(f"Exception raised during SpanProcessor `on_end`: {e}")

    def on_event(self, event: AgentSpecEvent, span: AgentSpecSpan) -> None:
        # No need to do anything on events occurring, everything is handled at span closure
        pass

    def startup(self) -> None:
        pass

    def shutdown(self) -> None:
        try:
            self.span_processor.shutdown()
        except Exception as e:
            # Whatever happens we do not crash the execution of the assistant,
            # but we warn the user
            logger.warning(f"Exception raised during SpanProcessor `shutdown`: {e}")

    async def on_start_async(self, span: AgentSpecSpan) -> None:
        raise NotImplementedError(f"Async stack not implemented for {self.__class__.__name__} yet")

    async def on_end_async(self, span: AgentSpecSpan) -> None:
        raise NotImplementedError(f"Async stack not implemented for {self.__class__.__name__} yet")

    async def on_event_async(self, event: AgentSpecEvent, span: AgentSpecSpan) -> None:
        raise NotImplementedError(f"Async stack not implemented for {self.__class__.__name__} yet")

    async def startup_async(self) -> None:
        raise NotImplementedError(f"Async stack not implemented for {self.__class__.__name__} yet")

    async def shutdown_async(self) -> None:
        raise NotImplementedError(f"Async stack not implemented for {self.__class__.__name__} yet")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Export all ended spans to the configured Exporter that have not yet been exported.

        Forwards the call to the internal OpenTelemetry span processor.
        """
        try:
            return self.span_processor.force_flush(timeout_millis=timeout_millis)
        except Exception as e:
            # Whatever happens we do not crash the execution of the assistant, but we warn the user
            logger.warning(f"Exception raised during SpanProcessor `force_flush`: {e}")
            return True


class OtelSimpleSpanProcessor(_OtelSpanProcessor):
    """AgentSpec wrapper for the OpenTelemetry SimpleSpanProcessor"""

    def _create_otel_span_processor(
        self, span_exporter: OtelSdkSpanExporter
    ) -> OtelSdkSpanProcessor:
        return OtelSdkSimpleSpanProcessor(span_exporter=span_exporter)


class OtelBatchSpanProcessor(_OtelSpanProcessor):
    """AgentSpec wrapper for the OpenTelemetry BatchSpanProcessor"""

    def _create_otel_span_processor(
        self, span_exporter: OtelSdkSpanExporter
    ) -> OtelSdkSpanProcessor:
        return OtelSdkBatchSpanProcessor(span_exporter=span_exporter)
