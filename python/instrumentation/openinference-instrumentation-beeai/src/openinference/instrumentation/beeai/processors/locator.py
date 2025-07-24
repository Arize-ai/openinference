from typing import Any

from beeai_framework.agents import BaseAgent
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements import Requirement
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import ChatModel, EmbeddingModel
from beeai_framework.context import RunContext, RunContextStartEvent
from beeai_framework.emitter import EventMeta
from beeai_framework.tools import Tool
from beeai_framework.workflows import Workflow

from instrumentation.beeai.processors.agents.base import AgentProcessor
from instrumentation.beeai.processors.agents.react import ReActAgentProcessor
from instrumentation.beeai.processors.agents.requirement_agent import RequirementAgentProcessor
from instrumentation.beeai.processors.agents.tool_calling import ToolCallingAgentProcessor
from instrumentation.beeai.processors.base import Processor
from instrumentation.beeai.processors.chat import ChatModelProcessor
from instrumentation.beeai.processors.embedding import EmbeddingModelProcessor
from instrumentation.beeai.processors.others.requirement import RequirementProcessor
from instrumentation.beeai.processors.tool import ToolProcessor
from instrumentation.beeai.processors.workflow import WorkflowProcessor

processor_registry: dict[type, type[Processor]] = {
    ReActAgent: ReActAgentProcessor,
    ToolCallingAgent: ToolCallingAgentProcessor,
    RequirementAgent: RequirementAgentProcessor,
    Requirement: RequirementProcessor,
    BaseAgent: AgentProcessor,
    Tool: ToolProcessor,
    ChatModel: ChatModelProcessor,
    EmbeddingModel: EmbeddingModelProcessor,
    Workflow: WorkflowProcessor,
}


def init_processor(data: Any, event: EventMeta) -> Processor:
    assert isinstance(data, RunContextStartEvent)
    assert isinstance(event.creator, RunContext)

    instance_cls = type(event.creator.instance)

    if instance_cls in processor_registry:
        cls_processor = processor_registry[instance_cls]
    else:
        for cls, cls_processor in processor_registry.items():
            if isinstance(event.creator.instance, cls):
                break
        else:
            cls_processor = Processor

    return cls_processor(data, event)
