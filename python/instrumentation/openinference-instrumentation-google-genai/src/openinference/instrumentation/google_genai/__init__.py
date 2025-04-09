import logging
from importlib import import_module
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import get_tracer, get_tracer_provider
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.google_genai._wrappers import (
    _AsyncGenerateContentWrapper,
    _SyncGenerateContent,
)
from openinference.instrumentation.google_genai.package import _instruments
from openinference.instrumentation.google_genai.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from google import genai


class GoogleGenAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for `google-genai`
    """

    __slots__ = ("_original_generate_content", "_original_async_generate_content")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        try:
            from google.genai.models import AsyncModels, Models
        except ImportError as err:
            raise Exception(
                "Could not import google-genai. Please install with `pip install google-genai`."
            ) from err

        self._original_generate_content = Models.generate_content
        wrap_function_wrapper(
            module="google.genai.models",
            name="Models.generate_content",
            wrapper=_SyncGenerateContent(tracer=self._tracer),
        )

        self._original_async_generate_content = AsyncModels.generate_content
        wrap_function_wrapper(
            module="google.genai.models",
            name="AsyncModels.generate_content",
            wrapper=_AsyncGenerateContentWrapper(tracer=self._tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        from google.genai.models import Models

        if self._original_generate_content is not None:
            Models.generate_content = self._original_generate_content
            self._original_generate_content = None

        if self._original_async_generate_content is not None:
            Models.async_generate_content = self._original_async_generate_content
            self._original_async_generate_content = None
