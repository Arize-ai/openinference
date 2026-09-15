"""OpenInference instrumentation for Qwen-Agent.

Usage
-----
.. code:: python

    from openinference.instrumentation.qwen_agent import QwenAgentInstrumentor
    from qwen_agent.agents import Assistant

    QwenAgentInstrumentor().instrument()

    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="my-assistant",
        system_message="You are a helpful assistant.",
    )
    for _ in bot.run([{"role": "user", "content": "Hello!"}]):
        pass
"""

import logging
from contextvars import copy_context
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Collection, Optional

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.qwen_agent._wrappers import (
    _ChatWrapper,
    _RunWrapper,
    _ToolCallWrapper,
)
from openinference.instrumentation.qwen_agent.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("qwen-agent >= 0.0.20, < 1",)

_AGENT_MODULE = "qwen_agent.agent"
_LLM_MODULE = "qwen_agent.llm.base"
# `parallel_exec` fans member agents out across a ThreadPoolExecutor without
# copying the caller's context, which would leave their spans orphaned in a
# separate trace. Used by ParallelDocQA.
_PARALLEL_MODULE = "qwen_agent.utils.parallel_executor"


class QwenAgentInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """OpenInference instrumentor for the Qwen-Agent framework.

    Three methods are wrapped:

    - ``Agent.run`` -> AGENT span (CHAIN for ``Memory``, which is an ``Agent``
      subclass used for file management and RAG). ``Agent.run_nonstream`` is not
      wrapped separately because it calls ``run`` internally.
    - ``BaseChatModel.chat`` -> LLM span. Every backend routes through this one
      method, so DashScope, the OpenAI-compatible backends, Azure,
      ``transformers`` and OpenVINO are all covered without duplication.
    - ``Agent._call_tool`` -> TOOL span. ``FnCallAgent`` is the only subclass
      that overrides it and it delegates to ``super()``, so wrapping the base
      method catches every tool call exactly once.

    The ``ThreadPoolExecutor`` used by ``qwen_agent.utils.parallel_executor`` is
    also swapped for a context-preserving one, so that agents fanned out across
    threads (``ParallelDocQA``) produce child spans rather than orphaned roots
    in a separate trace.

    Token counts are recorded only when Qwen-Agent itself exposes them, which
    the DashScope backends do via ``Message.extra["model_service_info"]``. On an
    OpenAI-compatible backend Qwen-Agent discards the usage block, so enable
    ``openinference-instrumentation-openai`` alongside this instrumentor to
    capture token counts from the nested OpenAI-SDK span. Counting the same
    tokens on both spans would inflate trace-level totals, which are summed
    across every span in a trace.
    """

    __slots__ = ("_tracer", "_original_executor")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        wrap_function_wrapper(
            _AGENT_MODULE,
            "Agent.run",
            _RunWrapper(tracer=self._tracer),  # type: ignore[arg-type]
        )
        wrap_function_wrapper(
            _AGENT_MODULE,
            "Agent._call_tool",
            _ToolCallWrapper(tracer=self._tracer),  # type: ignore[arg-type]
        )
        wrap_function_wrapper(
            _LLM_MODULE,
            "BaseChatModel.chat",
            _ChatWrapper(tracer=self._tracer),  # type: ignore[arg-type]
        )
        self._instrument_parallel_executor()

    def _instrument_parallel_executor(self) -> None:
        """Preserve the OTel context across `parallel_exec`'s worker threads."""
        self._original_executor = None
        try:
            module: ModuleType = import_module(_PARALLEL_MODULE)
            original_executor: type = getattr(module, "ThreadPoolExecutor")
        except Exception:
            logger.debug("could not patch %s", _PARALLEL_MODULE, exc_info=True)
            return

        self._original_executor = original_executor

        def _context_preserving_executor(*args: Any, **kwargs: Any) -> Any:
            executor = original_executor(*args, **kwargs)
            original_submit = executor.submit

            def submit(fn: Callable[..., Any], *fn_args: Any, **fn_kwargs: Any) -> Any:
                context = copy_context()
                return original_submit(lambda: context.run(fn, *fn_args, **fn_kwargs))

            executor.submit = submit
            return executor

        setattr(module, "ThreadPoolExecutor", _context_preserving_executor)

    def _uninstrument(self, **kwargs: Any) -> None:
        import qwen_agent.agent
        import qwen_agent.llm.base

        unwrap(qwen_agent.agent.Agent, "run")
        unwrap(qwen_agent.agent.Agent, "_call_tool")
        unwrap(qwen_agent.llm.base.BaseChatModel, "chat")

        if getattr(self, "_original_executor", None) is not None:
            try:
                setattr(
                    import_module(_PARALLEL_MODULE),
                    "ThreadPoolExecutor",
                    self._original_executor,
                )
            except Exception:
                logger.debug("could not restore %s", _PARALLEL_MODULE, exc_info=True)
            self._original_executor = None

    @property
    def tracer(self) -> Optional[OITracer]:
        return getattr(self, "_tracer", None)


__all__ = ["QwenAgentInstrumentor", "__version__"]
