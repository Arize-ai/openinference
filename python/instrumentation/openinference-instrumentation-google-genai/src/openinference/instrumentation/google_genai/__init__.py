import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import get_tracer, get_tracer_provider
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.google_genai.package import _instruments
from openinference.instrumentation.google_genai.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GoogleGenAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for `google-genai`
    """

    __slots__ = (
        "_original_generate_content",
        "_original_async_generate_content",
        "_original_generate_content_stream",
        "_original_async_generate_content_stream",
        "_original_create_interactions_resource",
        "_original_async_create_interactions_resource",
        "_original_create_caches",
        "_original_async_create_caches",
        "_tracer",
    )

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
            from google.genai._interactions.resources import (
                AsyncInteractionsResource,
                InteractionsResource,
            )
            from google.genai.models import AsyncModels, Models
            from google.genai.caches import AsyncCaches, Caches
        except ImportError as err:
            raise Exception(
                "Could not import google-genai. Please install with `pip install google-genai`."
            ) from err
        from openinference.instrumentation.google_genai._wrappers import (
            _AsyncCreateInteractionWrapper,
            _AsyncGenerateContentStream,
            _AsyncGenerateContentWrapper,
            _SyncCreateInteractionWrapper,
            _SyncGenerateContent,
            _SyncCreateCachesWrapper,
            _SyncGenerateContentStream,
        )

        self._original_create_interactions_resource = InteractionsResource.create
        self._original_create_caches = Caches.create
        wrap_function_wrapper(
            module="google.genai._interactions.resources",
            name="InteractionsResource.create",
            wrapper=_SyncCreateInteractionWrapper(tracer=self._tracer),
        )

        wrap_function_wrapper(
            module="google.genai.caches",
            name="Caches.create",
            wrapper=_SyncCreateCachesWrapper(tracer=self._tracer),
        )

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

        self._original_generate_content_stream = Models.generate_content_stream
        wrap_function_wrapper(
            module="google.genai.models",
            name="Models.generate_content_stream",
            wrapper=_SyncGenerateContentStream(tracer=self._tracer),
        )

        self._original_async_generate_content_stream = AsyncModels.generate_content_stream
        wrap_function_wrapper(
            module="google.genai.models",
            name="AsyncModels.generate_content_stream",
            wrapper=_AsyncGenerateContentStream(tracer=self._tracer),
        )
        self._original_async_create_interactions_resource = AsyncInteractionsResource.create
        wrap_function_wrapper(
            module="google.genai._interactions.resources",
            name="AsyncInteractionsResource.create",
            wrapper=_AsyncCreateInteractionWrapper(tracer=self._tracer),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        from google.genai._interactions.resources import (
            AsyncInteractionsResource,
            InteractionsResource,
        )
        from google.genai.models import AsyncModels, Models

        if self._original_generate_content is not None:
            setattr(Models, "generate_content", self._original_generate_content)

        if self._original_async_generate_content is not None:
            setattr(AsyncModels, "generate_content", self._original_async_generate_content)

        if self._original_generate_content_stream is not None:
            setattr(Models, "generate_content_stream", self._original_generate_content_stream)

        if self._original_async_generate_content_stream is not None:
            setattr(
                AsyncModels, "generate_content_stream", self._original_async_generate_content_stream
            )

        if self._original_async_create_interactions_resource is not None:
            setattr(
                AsyncInteractionsResource,
                "create",
                self._original_async_create_interactions_resource,
            )

        if self._original_create_interactions_resource is not None:
            setattr(InteractionsResource, "create", self._original_create_interactions_resource)
