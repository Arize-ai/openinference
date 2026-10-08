import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.cohere.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("cohere >= 5.13.0",)


class CohereInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Cohere Python client's v2 chat, embed, and rerank APIs."""

    __slots__ = ("_tracer",)

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from openinference.instrumentation.cohere._wrappers import (
            _AsyncChatStreamWrapper,
            _AsyncChatWrapper,
            _AsyncEmbedWrapper,
            _AsyncRerankWrapper,
            _ChatStreamWrapper,
            _ChatWrapper,
            _EmbedWrapper,
            _RerankWrapper,
        )

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

        # ``cohere.ClientV2`` / ``AsyncClientV2`` inherit these methods from the
        # base classes below, so wrapping the base methods covers both the sync
        # and async clients.
        for class_name, wrapper in (
            ("V2Client.chat", _ChatWrapper(tracer=self._tracer)),
            ("AsyncV2Client.chat", _AsyncChatWrapper(tracer=self._tracer)),
            ("V2Client.chat_stream", _ChatStreamWrapper(tracer=self._tracer)),
            ("AsyncV2Client.chat_stream", _AsyncChatStreamWrapper(tracer=self._tracer)),
            ("V2Client.embed", _EmbedWrapper(tracer=self._tracer)),
            ("AsyncV2Client.embed", _AsyncEmbedWrapper(tracer=self._tracer)),
            ("V2Client.rerank", _RerankWrapper(tracer=self._tracer)),
            ("AsyncV2Client.rerank", _AsyncRerankWrapper(tracer=self._tracer)),
        ):
            wrap_function_wrapper("cohere.v2.client", class_name, wrapper)

    def _uninstrument(self, **kwargs: Any) -> None:
        cohere_module = import_module("cohere.v2.client")
        for class_name, method_name in (
            ("V2Client", "chat"),
            ("AsyncV2Client", "chat"),
            ("V2Client", "chat_stream"),
            ("AsyncV2Client", "chat_stream"),
            ("V2Client", "embed"),
            ("AsyncV2Client", "embed"),
            ("V2Client", "rerank"),
            ("AsyncV2Client", "rerank"),
        ):
            # ``unwrap`` removes only this instrumentor's wrapper, leaving any
            # wrapper another layer installed on top of it intact.
            unwrap(getattr(cohere_module, class_name), method_name)
