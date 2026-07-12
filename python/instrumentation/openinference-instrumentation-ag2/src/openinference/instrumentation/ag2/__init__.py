from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.ag2._wrappers import (
    _ChatWrapper,
    _ReplyWrapper,
    _ToolWrapper,
)
from openinference.instrumentation.ag2.version import __version__

_instruments = ("ag2 >= 0.14.0, < 1.0.0",)


class AG2Instrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = ("_original_methods", "_tracer")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from autogen import ConversableAgent

        tracer_provider = kwargs.get("tracer_provider") or trace_api.get_tracer_provider()
        config = kwargs.get("config") or TraceConfig()
        if not isinstance(config, TraceConfig):
            raise TypeError("config must be an instance of TraceConfig")
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        chat_wrapper = _ChatWrapper(self._tracer)  # type: ignore[arg-type]
        reply_wrapper = _ReplyWrapper(self._tracer)  # type: ignore[arg-type]
        tool_wrapper = _ToolWrapper(self._tracer)  # type: ignore[arg-type]
        wrappers: dict[str, Callable[..., Any]] = {
            "initiate_chat": chat_wrapper,
            "a_initiate_chat": chat_wrapper.async_call,
            "generate_reply": reply_wrapper,
            "a_generate_reply": reply_wrapper.async_call,
            "execute_function": tool_wrapper,
            "a_execute_function": tool_wrapper.async_call,
        }
        self._original_methods = {name: getattr(ConversableAgent, name) for name in wrappers}
        for name, wrapper in wrappers.items():
            wrap_function_wrapper(ConversableAgent, name, wrapper)

    def _uninstrument(self, **kwargs: Any) -> None:
        from autogen import ConversableAgent

        for name, method in getattr(self, "_original_methods", {}).items():
            setattr(ConversableAgent, name, method)
        self._original_methods = {}


Ag2Instrumentor = AG2Instrumentor

__all__ = ["AG2Instrumentor", "Ag2Instrumentor"]
