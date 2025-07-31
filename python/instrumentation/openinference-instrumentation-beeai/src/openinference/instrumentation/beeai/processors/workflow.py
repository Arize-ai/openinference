from typing import Any, ClassVar

from beeai_framework.context import RunContext, RunContextStartEvent
from beeai_framework.emitter import EventMeta
from beeai_framework.workflows import (
    Workflow,
    WorkflowErrorEvent,
    WorkflowStartEvent,
    WorkflowSuccessEvent,
)
from beeai_framework.workflows.agent.agent import Schema as AgentWorkflowSchema
from pydantic import BaseModel
from typing_extensions import override

from openinference.instrumentation.beeai._utils import (
    _unpack_object,
    safe_dump_model_schema,
    stringify,
)
from openinference.instrumentation.beeai.processors.base import Processor
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class WorkflowProcessor(Processor):
    kind: ClassVar[OpenInferenceSpanKindValues] = OpenInferenceSpanKindValues.CHAIN

    def __init__(self, event: "RunContextStartEvent", meta: "EventMeta"):
        super().__init__(event, meta)

        assert isinstance(meta.creator, RunContext)
        assert isinstance(meta.creator.instance, Workflow)

        self._last_step = 0

        workflow = meta.creator.instance
        self.span.name = workflow.name or self.span.name
        self.span.set_attributes(
            {
                f"{SpanAttributes.METADATA}.name": workflow.name,
                f"{SpanAttributes.METADATA}.all_steps": workflow.step_names,
                f"{SpanAttributes.METADATA}.start_step": workflow.start_step,
                **_unpack_object(
                    safe_dump_model_schema(workflow.schema),
                    prefix=f"{SpanAttributes.METADATA}.schema",
                ),
            }
        )

    @override
    async def update(
        self,
        event: Any,
        meta: "EventMeta",
    ) -> None:
        await super().update(event, meta)

        self.span.add_event(f"{meta.name} ({meta.path})", timestamp=meta.created_at)

        match event:
            case WorkflowStartEvent() | WorkflowSuccessEvent() | WorkflowErrorEvent():
                self._update_state(event)
            case _:
                self.span.child(meta.name, event=(event, meta))

    def _update_state(
        self, event: "WorkflowStartEvent[Any] | WorkflowSuccessEvent[Any] | WorkflowErrorEvent[Any]"
    ) -> None:
        self.span.set_attribute(f"{SpanAttributes.METADATA}.current_step", event.step)
        self.span.set_attributes(
            _unpack_object(
                _serialize_state(event.run.state), prefix=f"{SpanAttributes.METADATA}.state"
            )
        )

        # update steps
        for idx, step in enumerate(event.run.steps):
            if idx < self._last_step:
                continue

            self.span.set_attributes(
                _unpack_object(
                    {
                        "name": step.name,
                        "state": _serialize_state(step.state),
                    },
                    prefix=f"{SpanAttributes.METADATA}.steps.{idx}",
                )
            )
            self._last_step = idx

        if isinstance(event, WorkflowSuccessEvent):
            if event.next == Workflow.END or event.run.result is not None:
                result = event.run.result if event.run.result is not None else event.state
                self.span.attributes.update(
                    {
                        SpanAttributes.OUTPUT_VALUE: stringify(_serialize_state(result)),
                        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                    }
                )


def _serialize_state(result: BaseModel) -> dict[str, Any]:
    exclude = {"new_messages", "inputs"} if isinstance(result, AgentWorkflowSchema) else set()
    return result.model_dump(exclude=exclude, exclude_none=True)
