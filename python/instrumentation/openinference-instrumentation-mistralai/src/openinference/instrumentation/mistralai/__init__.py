import logging
from typing import (
    Any,
    Collection,
)

from openinference.instrumentation.mistralai._sync_chat_wrapper import _SyncChatWrapper
from openinference.instrumentation.mistralai.package import _instruments
from openinference.instrumentation.mistralai.version import __version__
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MODULE = "mistralai"


class MistralAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for mistralai
    """

    __slots__ = ("_original_chat_method",)

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)

        try:
            from mistralai.client import MistralClient
        except ImportError as err:
            raise Exception(
                "Could not import mistralai. Please install with `pip install mistralai`."
            ) from err

        self._original_chat_method = MistralClient.chat
        wrap_function_wrapper(
            module=_MODULE,
            name="client.MistralClient.chat",
            wrapper=_SyncChatWrapper(tracer, MistralClient()),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        from mistralai.client import MistralClient

        MistralClient.chat = self._original_chat_method  # type: ignore
