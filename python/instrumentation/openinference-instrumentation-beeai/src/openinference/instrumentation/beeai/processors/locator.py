import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from beeai_framework.emitter import EventMeta

from openinference.instrumentation.beeai.processors.base import Processor


class ProcessorLocator:
    entries: dict[type, type] = {}
    _loaded: bool = False

    @staticmethod
    def _load() -> None:
        if ProcessorLocator._loaded:
            return

        ProcessorLocator._loaded = True

        with contextlib.suppress(ImportError):
            from beeai_framework.backend.chat import ChatModel

            from .chat import ChatModelProcessor

            ProcessorLocator.entries[ChatModel] = ChatModelProcessor

        with contextlib.suppress(ImportError):
            from beeai_framework.backend.embedding import EmbeddingModel

            from .embedding import EmbeddingModelProcessor

            ProcessorLocator.entries[EmbeddingModel] = EmbeddingModelProcessor

        with contextlib.suppress(ImportError):
            from beeai_framework.agents.react.agent import ReActAgent

            from .agents.react import ReActAgentProcessor

            ProcessorLocator.entries[ReActAgent] = ReActAgentProcessor

        with contextlib.suppress(ImportError):
            from beeai_framework.agents.tool_calling.agent import ToolCallingAgent

            from .agents.tool_calling import ToolCallingAgentProcessor

            ProcessorLocator.entries[ToolCallingAgent] = ToolCallingAgentProcessor

        with contextlib.suppress(ImportError):
            from beeai_framework.agents.experimental.agent import RequirementAgent

            from .agents.requirement_agent import RequirementAgentProcessor

            ProcessorLocator.entries[RequirementAgent] = RequirementAgentProcessor

        with contextlib.suppress(ImportError):
            from beeai_framework.agents.experimental.requirements.requirement import Requirement

            from openinference.instrumentation.beeai.processors.requirement import (
                RequirementProcessor,
            )

            ProcessorLocator.entries[Requirement] = RequirementProcessor

        with contextlib.suppress(ImportError):
            from beeai_framework.agents.base import BaseAgent

            from openinference.instrumentation.beeai.processors.agents.base import AgentProcessor

            ProcessorLocator.entries[BaseAgent] = AgentProcessor

        with contextlib.suppress(ImportError):
            from beeai_framework.tools.tool import Tool

            from openinference.instrumentation.beeai.processors.tool import ToolProcessor

            ProcessorLocator.entries[Tool] = ToolProcessor

        with contextlib.suppress(ImportError):
            from beeai_framework.workflows.workflow import Workflow

            from .workflow import WorkflowProcessor

            ProcessorLocator.entries[Workflow] = WorkflowProcessor

    @staticmethod
    def locate(data: Any, event: "EventMeta") -> Processor:
        ProcessorLocator._load()

        from beeai_framework.context import RunContext, RunContextStartEvent

        assert isinstance(data, RunContextStartEvent)
        assert isinstance(event.creator, RunContext)

        instance_cls = type(event.creator.instance)

        if instance_cls in ProcessorLocator.entries:
            cls_processor = ProcessorLocator.entries[instance_cls]
        else:
            for cls, cls_processor in ProcessorLocator.entries.items():
                if isinstance(event.creator.instance, cls):
                    break
            else:
                cls_processor = Processor

        return cls_processor(data, event)  # type: ignore
