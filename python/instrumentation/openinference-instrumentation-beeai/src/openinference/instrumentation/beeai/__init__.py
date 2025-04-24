# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Collection, TypeVar

import wrapt
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues

from .middleware import create_telemetry_middleware

_instruments = ("beeai-framework >= 0.1.10, < 0.1.19",)
try:
    __version__ = version("beeai-framework")
except PackageNotFoundError:
    __version__ = "unknown"


class BeeAIInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_original_react_agent_run",
        "_original_tool_calling_agent_run",
        "_original_chat_model_create",
        "_original_chat_model_create_structure",
        "_original_tool_run",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
            if not (tracer_provider := kwargs.get("tracer_provider")):
                tracer_provider = trace_api.get_tracer_provider()
            if not (config := kwargs.get("config")):
                config = TraceConfig()
            else:
                assert isinstance(config, TraceConfig)

            from beeai_framework.agents.base import BaseAgent
            from beeai_framework.agents.react.agent import ReActAgent
            from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
            from beeai_framework.backend.chat import ChatModel
            from beeai_framework.tools import Tool

            self._tracer = OITracer(
                trace_api.get_tracer(__name__, __version__, tracer_provider),
                config=config,
            )

            F = TypeVar("F", bound=Callable[..., Any])

            @wrapt.decorator  # type: ignore
            def run_wrapper(
                wrapped: F, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> Any:
                result = wrapped(*args, **kwargs)

                span_kind = OpenInferenceSpanKindValues.UNKNOWN
                if isinstance(instance, ChatModel):
                    span_kind = OpenInferenceSpanKindValues.LLM
                if isinstance(instance, BaseAgent):
                    span_kind = OpenInferenceSpanKindValues.AGENT
                if isinstance(instance, Tool):
                    span_kind = OpenInferenceSpanKindValues.TOOL

                if result.middleware:
                    result.middleware(create_telemetry_middleware(self._tracer, span_kind.value))

                return result

            ## Agent support
            self._original_react_agent_run = getattr(
                import_module("beeai_framework.agents.react.agent"), "run", None
            )
            setattr(ReActAgent, "run", run_wrapper(ReActAgent.run))
            self._original_tool_calling_agent_run = getattr(
                import_module("beeai_framework.agents.tool_calling.agent"), "run", None
            )
            setattr(ToolCallingAgent, "run", run_wrapper(ToolCallingAgent.run))
            ## LLM support
            self._original_chat_model_create = getattr(
                import_module("beeai_framework.backend.chat"), "create", None
            )
            setattr(ChatModel, "create", run_wrapper(ChatModel.create))
            self._original_chat_model_create_structure = getattr(
                import_module("beeai_framework.backend.chat"), "create_structure", None
            )
            setattr(ChatModel, "create_structure", run_wrapper(ChatModel.create_structure))

            ## Tool support
            self._original_tool_run = getattr(import_module("beeai_framework.tools"), "run", None)
            setattr(Tool, "run", run_wrapper(Tool.run))

        except ImportError as e:
            print("ImportError during instrumentation:", e)
        except Exception as e:
            print("Instrumentation error:", e)

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._original_react_agent_run is not None:
            from beeai_framework.agents.react.agent import ReActAgent

            setattr(ReActAgent, "run", self._original_react_agent_run)
            self._original_react_agent_run = None
        if self._original_tool_calling_agent_run is not None:
            from beeai_framework.agents.tool_calling.agent import ToolCallingAgent

            setattr(ToolCallingAgent, "run", self._original_tool_calling_agent_run)
            self._original_tool_calling_agent_run = None
        if self._original_chat_model_create is not None:
            from beeai_framework.backend.chat import ChatModel

            setattr(ChatModel, "create", self._original_chat_model_create)
            self._original_chat_model_create = None
        if self._original_chat_model_create_structure is not None:
            from beeai_framework.backend.chat import ChatModel

            setattr(ChatModel, "create_structure", self._original_chat_model_create_structure)
            self._original_chat_model_create_structure = None
        if self._original_tool_run is not None:
            from beeai_framework.tools import Tool

            setattr(Tool, "run", self._original_tool_run)
            self._original_tool_run = None
