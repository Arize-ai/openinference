from typing import Any

from beeai_framework.agents.experimental.events import (
    RequirementAgentStartEvent,
    RequirementAgentSuccessEvent,
)
from beeai_framework.agents.experimental.types import RequirementAgentRunStateStep
from beeai_framework.context import RunContextStartEvent
from beeai_framework.emitter import EventMeta
from typing_extensions import override

from openinference.instrumentation.beeai._utils import _unpack_object
from openinference.instrumentation.beeai.processors.agents.base import AgentProcessor
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes


class RequirementAgentProcessor(AgentProcessor):
    def __init__(self, event: "RunContextStartEvent", meta: "EventMeta"):
        super().__init__(event, meta)
        self._steps: list[RequirementAgentRunStateStep] = []

    @override
    async def update(
        self,
        event: Any,
        meta: "EventMeta",
    ) -> None:
        await super().update(event, meta)
        self.span.add_event(f"{meta.name} ({meta.path})", timestamp=meta.created_at)

        match event:
            case RequirementAgentStartEvent():
                self._sync_steps(event.state.steps)
            case RequirementAgentSuccessEvent():
                self._sync_steps(event.state.steps)

                if event.state.answer is not None:
                    self.span.set_attributes(
                        {
                            SpanAttributes.OUTPUT_VALUE: event.state.answer.text,
                            SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
                        }
                    )
                if event.state.result is not None:
                    self.span.set_attributes(
                        _unpack_object(
                            event.state.result, prefix=f"{SpanAttributes.METADATA}.result"
                        )
                    )
            case _:
                self.span.child(meta.name, event=(event, meta))

    def _sync_steps(self, steps: list["RequirementAgentRunStateStep"]) -> None:
        new_items = steps[len(self._steps) :]
        self._steps.extend(new_items)

        for idx, (new_step, old_step) in enumerate(zip(steps, self._steps)):
            if new_step.id != old_step.id:
                # TODO: cleanup old keys
                self._steps.clear()
                self._sync_steps(steps)
                return

            if new_step in new_items or new_step != old_step:
                self._steps[idx] = new_step
                self.span.set_attributes(
                    _unpack_object(
                        new_step,
                        prefix=f"{SpanAttributes.METADATA}.steps.{idx}",
                    )
                )
