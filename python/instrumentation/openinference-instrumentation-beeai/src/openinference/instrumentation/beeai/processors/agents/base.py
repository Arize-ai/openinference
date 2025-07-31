from typing import ClassVar

from beeai_framework.agents import BaseAgent
from beeai_framework.context import RunContext, RunContextStartEvent
from beeai_framework.emitter import EventMeta

from openinference.instrumentation.beeai._utils import _unpack_object
from openinference.instrumentation.beeai.processors.base import Processor
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class AgentProcessor(Processor):
    kind: ClassVar[OpenInferenceSpanKindValues] = OpenInferenceSpanKindValues.AGENT

    def __init__(self, event: RunContextStartEvent, meta: "EventMeta") -> None:
        super().__init__(event, meta)

        assert isinstance(meta.creator, RunContext)
        assert isinstance(meta.creator.instance, BaseAgent)

        agent = meta.creator.instance
        self.span.name = agent.meta.name or self.span.name
        self.span.set_attributes(
            {
                SpanAttributes.AGENT_NAME: agent.meta.name,
                f"{SpanAttributes.METADATA}.agent_description": agent.meta.description,
            }
        )
        self.span.set_attributes(
            _unpack_object(agent.memory.to_json_safe(), prefix=f"{SpanAttributes.METADATA}.memory")
        )
