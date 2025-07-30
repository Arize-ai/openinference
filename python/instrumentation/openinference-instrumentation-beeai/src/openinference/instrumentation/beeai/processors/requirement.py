from typing import Any, ClassVar

from beeai_framework.agents.experimental.requirements.ask_permission import AskPermissionRequirement
from beeai_framework.agents.experimental.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.experimental.requirements.events import RequirementInitEvent
from beeai_framework.agents.experimental.requirements.requirement import Requirement
from beeai_framework.context import RunContext, RunContextStartEvent
from beeai_framework.emitter import EventMeta
from typing_extensions import override

from openinference.instrumentation.beeai._utils import _unpack_object
from openinference.instrumentation.beeai.processors.base import Processor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


class RequirementProcessor(Processor):
    kind: ClassVar[OpenInferenceSpanKindValues] = OpenInferenceSpanKindValues.UNKNOWN

    def __init__(self, event: "RunContextStartEvent", meta: "EventMeta") -> None:
        super().__init__(event, meta)

        assert isinstance(meta.creator, RunContext)
        assert isinstance(meta.creator.instance, Requirement)

        requirement = meta.creator.instance
        self.span.name = requirement.name or self.span.name
        self._sync_state(meta.creator.instance)

    def _sync_state(self, instance: "Requirement[Any]") -> None:
        attributes = {
            "name": instance.name,
            "enabled": instance.enabled,
            "priority": instance.priority,
            "state": instance.state,
            "middlewares": [str(m) for m in instance.middlewares],
        }

        if isinstance(instance, ConditionalRequirement):
            attributes["target_name"] = str(instance.source)
        elif isinstance(instance, AskPermissionRequirement):
            attributes["includes"] = [str(t) for t in instance._include]
            attributes["excludes"] = [str(t) for t in instance._exclude]
            attributes["remember_choices"] = instance._remember_choices
            attributes["_always_allow"] = instance._always_allow

        self.span.set_attributes(
            _unpack_object(
                attributes,
                prefix=f"{SpanAttributes.METADATA}",
            )
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
            case RequirementInitEvent():
                pass
            case _:
                self.span.child(meta.name, event=(event, meta))
