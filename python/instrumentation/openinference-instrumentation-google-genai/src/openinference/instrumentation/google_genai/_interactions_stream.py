import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Optional,
)

from opentelemetry import trace as trace_api

from openinference.instrumentation.google_genai._stream import _ResponseAccumulator, _Stream
from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.instrumentation.google_genai.interactions_attributes import (
    get_attributes_from_response,
)

if TYPE_CHECKING:
    from google.genai._interactions.types import Interaction
    from google.genai.types import GenerateContentResponse

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _InteractionsStream(_Stream):
    __slots__ = (
        "_response_accumulator",
        "_with_span",
        "_is_finished",
    )

    def __init__(
        self,
        stream: Iterator["GenerateContentResponse"],
        with_span: _WithSpan,
    ) -> None:
        super().__init__(stream, with_span)
        self._response_accumulator = _InteractionAccumulator()

    def _finish_tracing(
        self,
        status: Optional[trace_api.Status] = None,
    ) -> None:
        self._with_span.set_attributes(
            get_attributes_from_response(self._response_accumulator.result())
        )
        self._with_span.finish_tracing(status=status)


class _InteractionAccumulator(_ResponseAccumulator):
    __slots__ = (
        "_is_null",
        "_interaction",
        "_outputs",
    )

    def __init__(self) -> None:
        self._is_null = True
        self._interaction: Optional["Interaction"] = None
        self._outputs: Any = []

    def process_chunk(self, event: Any) -> None:
        """Process a single streaming event and update the Interaction object."""
        self._is_null = False
        event_type = event.event_type
        from google.genai._interactions import types

        if event_type == "interaction.start":
            # Initialize the Interaction object
            self._interaction = types.Interaction(
                id=event.interaction.id,
                status=event.interaction.status,
                agent=None,
                created=event.interaction.created,
                model=event.interaction.model,
                outputs=[],
                previous_interaction_id=None,
                role=None,
                updated=None,
                usage=None,
            )

        elif event_type == "interaction.status_update":
            if self._interaction:
                self._interaction.status = event.status

        elif event_type == "content.start":
            # Ensure outputs list is large enough
            while len(self._outputs) <= event.index:
                self._outputs.append(None)

            # Initialize the appropriate content type
            if event.content.type == "thought":
                self._outputs[event.index] = types.ThoughtContent(
                    type="thought",
                    signature=None,
                    summary=None,
                )
            elif event.content.type == "text":
                self._outputs[event.index] = types.TextContent(
                    type="text",
                    annotations=None,
                    text="",
                )
            elif event.content.type == "image":
                self._outputs[event.index] = types.ImageContent(
                    type="image", data=None, mime_type=None, resolution=None
                )

        elif event_type == "content.delta":
            delta = event.delta
            idx = event.index

            if delta.type == "thought_signature":
                # Update thought signature
                if self._outputs[idx] and isinstance(self._outputs[idx], types.ThoughtContent):
                    self._outputs[idx].signature = delta.signature

            elif delta.type == "text":
                # Append text delta
                if self._outputs[idx] and isinstance(self._outputs[idx], types.TextContent):
                    if self._outputs[idx].text is None:
                        self._outputs[idx].text = delta.text or ""
                    else:
                        self._outputs[idx].text += delta.text or ""
            else:
                if self._outputs[idx] and isinstance(self._outputs[idx], types.ImageContent):
                    if self._outputs[idx].data is None:
                        self._outputs[idx].data = delta.data or ""
                    else:
                        self._outputs[idx].data += delta.data or ""
                    if delta.mime_type is not None:
                        self._outputs[idx].mime_type = delta.mime_type
                    if delta.resolution is not None:
                        self._outputs[idx].resolution = delta.resolution
        elif event_type == "interaction.complete":
            # Update final metadata
            if self._interaction:
                self._interaction.id = event.interaction.id
                self._interaction.status = event.interaction.status
                self._interaction.created = event.interaction.created
                self._interaction.updated = event.interaction.updated
                self._interaction.role = event.interaction.role
                self._interaction.usage = event.interaction.usage
                # Assign the accumulated outputs
                self._interaction.outputs = [out for out in self._outputs if out is not None]

    def result(self) -> Optional["Interaction"]:
        """Return the accumulated Interaction object."""
        if self._is_null or self._interaction is None:
            return self._interaction
        # Ensure outputs are assigned (in case interaction.complete wasn't received)
        if not self._interaction.outputs and self._outputs:
            self._interaction.outputs = [out for out in self._outputs if out is not None]
        return self._interaction
