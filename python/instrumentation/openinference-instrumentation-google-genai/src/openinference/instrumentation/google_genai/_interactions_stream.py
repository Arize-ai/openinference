import json
import logging
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Mapping,
    Optional,
)

from opentelemetry import trace as trace_api
from wrapt import ObjectProxy

from openinference.instrumentation.google_genai._utils import get_attribute
from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.instrumentation.google_genai.interactions_attributes import (
    get_attributes_from_response,
)

if TYPE_CHECKING:
    from google.genai._interactions.types import InteractionSSEEvent

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _InteractionsStream(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    __slots__ = (
        "_response_accumulator",
        "_with_span",
        "_is_finished",
        "request_parameters",
    )

    def __init__(
        self,
        stream: Iterator["InteractionSSEEvent"],
        with_span: _WithSpan,
        request_parameters: Mapping[str, Any],
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _InteractionAccumulator()
        self.request_parameters = request_parameters
        self._with_span = with_span

    def __iter__(self) -> Iterator["InteractionSSEEvent"]:
        try:
            for item in self.__wrapped__:
                self._response_accumulator.process_chunk(item)
                yield item
        except Exception as exception:
            status = trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
            self._with_span.record_exception(exception)
            self._finish_tracing(status=status)
            raise
        # completed without exception
        status = trace_api.Status(
            status_code=trace_api.StatusCode.OK,
        )
        self._finish_tracing(status=status)

    async def __aiter__(self) -> AsyncIterator["InteractionSSEEvent"]:
        try:
            async for item in self.__wrapped__:
                self._response_accumulator.process_chunk(item)
                yield item
        except Exception as exception:
            status = trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
            self._with_span.record_exception(exception)
            self._finish_tracing(status=status)
            raise
        # completed without exception
        status = trace_api.Status(
            status_code=trace_api.StatusCode.OK,
        )
        self._finish_tracing(status=status)

    def _finish_tracing(
        self,
        status: Optional[trace_api.Status] = None,
    ) -> None:
        self._with_span.set_attributes(
            get_attributes_from_response(
                self.request_parameters, self._response_accumulator.result()
            )
        )
        self._with_span.finish_tracing(status=status)


class _InteractionAccumulator:
    __slots__ = (
        "_is_null",
        "_interaction",
        "_steps",
    )

    def __init__(self) -> None:
        self._is_null = True
        self._interaction: Any = None
        self._steps: Any = []

    def process_chunk(self, event: Any) -> None:
        """Process a single streaming event and update the Interaction object."""
        self._is_null = False
        event_type = get_attribute(event, "event_type")

        if event_type == "interaction.created":
            self._interaction = event.interaction

        elif event_type == "interaction.status_update":
            if self._interaction:
                self._interaction.status = event.status

        elif event_type == "step.start":
            self._set_indexed_item(self._steps, event.index, event.step)

        elif event_type == "step.delta":
            self._process_step_delta(event.index, event.delta)

        elif event_type == "interaction.completed":
            if self._interaction:
                self._interaction.id = event.interaction.id
                self._interaction.status = event.interaction.status
                self._interaction.created = event.interaction.created
                self._interaction.updated = event.interaction.updated
                self._interaction.role = event.interaction.role
                self._interaction.usage = event.interaction.usage
                self._assign_accumulated_items()
            else:
                self._interaction = event.interaction
                self._assign_accumulated_items()

    def _set_indexed_item(self, items: list[Any], index: int, item: Any) -> None:
        while len(items) <= index:
            items.append(None)
        items[index] = item

    def _set_attribute(self, obj: Any, attr_name: str, value: Any) -> None:
        setattr(obj, attr_name, value)

    def _append_to_attribute(self, obj: Any, attr_name: str, value: Any) -> None:
        if obj is None or value is None:
            return
        existing_value = get_attribute(obj, attr_name)
        if not isinstance(existing_value, str):
            existing_value = ""
        self._set_attribute(obj, attr_name, existing_value + value)

    def _process_step_delta(self, index: int, delta: Any) -> None:
        if index >= len(self._steps) or self._steps[index] is None:
            self._set_indexed_item(self._steps, index, SimpleNamespace(type="model_output"))
        step = self._steps[index]
        delta_type = get_attribute(delta, "type")

        if delta_type == "thought_signature":
            self._set_attribute(step, "signature", get_attribute(delta, "signature"))
        elif delta_type == "thought_summary":
            summary = get_attribute(step, "summary") or []
            if content := get_attribute(delta, "content"):
                summary.append(content)
            self._set_attribute(step, "summary", summary)
        elif delta_type == "arguments_delta":
            self._append_to_attribute(
                step,
                "arguments",
                get_attribute(delta, "partial_arguments") or "",
            )
        elif delta_type in ("text", "image"):
            self._append_model_output_delta(step, delta)

    def _append_model_output_delta(self, step: Any, delta: Any) -> None:
        content = get_attribute(step, "content") or []
        if not content:
            content.append(SimpleNamespace(type=get_attribute(delta, "type")))
        item = content[-1]
        delta_type = get_attribute(delta, "type")
        if delta_type == "text":
            self._append_to_attribute(item, "text", get_attribute(delta, "text") or "")
        elif delta_type == "image":
            self._append_to_attribute(item, "data", get_attribute(delta, "data") or "")
            if (mime_type := get_attribute(delta, "mime_type")) is not None:
                self._set_attribute(item, "mime_type", mime_type)
            if (resolution := get_attribute(delta, "resolution")) is not None:
                self._set_attribute(item, "resolution", resolution)
            if (uri := get_attribute(delta, "uri")) is not None:
                self._set_attribute(item, "uri", uri)
        self._set_attribute(step, "content", content)

    def _assign_accumulated_items(self) -> None:
        if self._interaction is None:
            return
        if self._steps:
            self._interaction.steps = [step for step in self._steps if step is not None]
            self._finalize_function_call_steps(self._interaction.steps)

    def _finalize_function_call_steps(self, steps: list[Any]) -> None:
        for step in steps:
            if get_attribute(step, "type") != "function_call":
                continue
            arguments = get_attribute(step, "arguments")
            if not isinstance(arguments, str):
                continue
            try:
                self._set_attribute(step, "arguments", json.loads(arguments))
            except json.JSONDecodeError:
                pass

    def result(self) -> Any:
        """Return the accumulated Interaction object."""
        if self._is_null or self._interaction is None:
            return self._interaction
        self._assign_accumulated_items()
        return self._interaction
