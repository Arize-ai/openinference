import logging
from typing import Any, Callable, Collection

from langchain_core.callbacks import BaseCallbackManager
from openinference.instrumentation.langchain._tracer import OpenInferenceTracer
from openinference.instrumentation.langchain.package import _instruments
from openinference.instrumentation.langchain.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LangChainInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for LangChain
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        wrap_function_wrapper(
            "langchain_core.callbacks",
            "BaseCallbackManager.__init__",
            _BaseCallbackManagerInit(tracer=tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        pass


class _BaseCallbackManagerInit:
    __slots__ = ("_tracer",)

    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., None],
        instance: BaseCallbackManager,
        args: Any,
        kwargs: Any,
    ) -> None:
        wrapped(*args, **kwargs)
        for handler in instance.inheritable_handlers:
            # Handlers may be copied when new managers are created, so we
            # don't want to keep adding. E.g. see the following location.
            # https://github.com/langchain-ai/langchain/blob/5c2538b9f7fb64afed2a918b621d9d8681c7ae32/libs/core/langchain_core/callbacks/manager.py#L1876  # noqa: E501
            if isinstance(handler, OpenInferenceTracer):
                break
        else:
            instance.add_handler(OpenInferenceTracer(tracer=self._tracer), True)
