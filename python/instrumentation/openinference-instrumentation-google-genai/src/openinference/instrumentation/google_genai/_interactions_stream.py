import logging
from datetime import datetime, timezone
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

from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.instrumentation.google_genai.interactions_attributes import (
    get_attributes_from_response,
)

if TYPE_CHECKING:
    from google.genai._interactions.types import Interaction
    from google.genai.types import GenerateContentResponse

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _InteractionsStream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_response_accumulator",
        "_with_span",
        "_is_finished",
        "request_parameters",
    )

    def __init__(
        self,
        stream: Iterator["GenerateContentResponse"],
        with_span: _WithSpan,
        request_parameters: Mapping[str, Any],
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _InteractionAccumulator()
        self.request_parameters = request_parameters
        self._with_span = with_span

    def __iter__(self) -> Iterator["GenerateContentResponse"]:
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

    async def __aiter__(self) -> AsyncIterator["GenerateContentResponse"]:
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
        self._interaction: Optional["Interaction"] = None
        self._steps: list[Any] = []

    def process_chunk(self, event: Any) -> None:
        """Process a single streaming event and update the Interaction object."""
        self._is_null = False
        event_type = getattr(event, "type", getattr(event, "event_type", None))
        from google.genai._interactions import types

        if event_type == "interaction.created":
            created_at = event.interaction.created or datetime.now(timezone.utc)
            self._interaction = types.Interaction(
                id=event.interaction.id,
                status=event.interaction.status,
                agent=None,
                created=created_at,
                model=event.interaction.model,
                steps=[],
                previous_interaction_id=None,
                role=None,
                updated=event.interaction.updated or created_at,
                usage=None,
            )

        elif event_type == "step.start":
            while len(self._steps) <= event.index:
                self._steps.append(None)

            # Initialize based on step type (text, thought, function_call)
            if event.step.type == "thought":
                self._steps[event.index] = types.Thought(type="thought", signature=None)
            elif event.step.type == "text":
                self._steps[event.index] = types.TextContent(type="text", text="")

        elif event_type == "step.delta":
            delta = event.delta
            idx = event.index

            if not self._steps[idx]:
                return

            if delta.type == "thought_signature":
                if isinstance(self._steps[idx], types.Thought):
                    self._steps[idx].signature = delta.signature
            elif delta.type == "text":
                if isinstance(self._steps[idx], types.TextContent):
                    current_text = self._steps[idx].text or ""
                    self._steps[idx].text = current_text + (delta.text or "")

        elif event_type == "interaction.completed":
            if self._interaction:
                self._interaction.status = event.interaction.status
                self._interaction.usage = event.interaction.usage
                self._interaction.steps = [s for s in self._steps if s is not None]

    def result(self) -> Optional["Interaction"]:
        """Return the accumulated Interaction object."""
        if self._is_null or self._interaction is None:
            return self._interaction
        if not self._interaction.steps and self._steps:
            self._interaction.steps = [s for s in self._steps if s is not None]
        return self._interaction
