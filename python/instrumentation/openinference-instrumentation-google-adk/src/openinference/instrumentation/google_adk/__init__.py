import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore[attr-defined]
from wrapt import wrap_object_attribute

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.google_adk.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("google-adk >= 0.3.0",)


class GoogleADKInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for google-adk
    """

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

        from openinference.instrumentation.google_adk._callback import (
            GoogleADKTracingCallback,
        )

        callback = GoogleADKTracingCallback(tracer=self._tracer)

        def callback_factory(value, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Honor any callback passed by the user
            kwargs["callback_parent"]._original_value = value
            return kwargs["callback_wrapper"]

        wrap_object_attribute(
            module="google.adk.agents.llm_agent",
            name="LlmAgent.before_agent_callback",
            factory=callback_factory,
            kwargs={
                "callback_parent": callback,
                "callback_wrapper": callback.before_agent_callback,
            },
        )

        wrap_object_attribute(
            module="google.adk.agents.llm_agent",
            name="LlmAgent.before_model_callback",
            factory=callback_factory,
            kwargs={
                "callback_parent": callback,
                "callback_wrapper": callback.before_model_callback,
            },
        )

        wrap_object_attribute(
            module="google.adk.agents.llm_agent",
            name="LlmAgent.before_tool_callback",
            factory=callback_factory,
            kwargs={"callback_parent": callback, "callback_wrapper": callback.before_tool_callback},
        )

        wrap_object_attribute(
            module="google.adk.agents.llm_agent",
            name="LlmAgent.after_agent_callback",
            factory=callback_factory,
            kwargs={"callback_parent": callback, "callback_wrapper": callback.after_agent_callback},
        )

        wrap_object_attribute(
            module="google.adk.agents.llm_agent",
            name="LlmAgent.after_model_callback",
            factory=callback_factory,
            kwargs={"callback_parent": callback, "callback_wrapper": callback.after_model_callback},
        )

        wrap_object_attribute(
            module="google.adk.agents.llm_agent",
            name="LlmAgent.after_tool_callback",
            factory=callback_factory,
            kwargs={"callback_parent": callback, "callback_wrapper": callback.after_tool_callback},
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        pass
