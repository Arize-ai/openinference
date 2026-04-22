import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import get_tracer, get_tracer_provider
from wrapt.patches import wrap_function_wrapper

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
        "_original_embed_content",
        "_original_async_embed_content",
        "_original_generate_content",
        "_original_async_generate_content",
        "_original_generate_content_stream",
        "_original_async_generate_content_stream",
        "_original_create_interactions_resource",
        "_original_async_create_interactions_resource",
        "_original_create_caches",
        "_original_async_create_caches",
        "_original_api_request",
        "_original_api_request_streamed",
        "_original_api_async_request",
        "_original_api_async_request_streamed",
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
            from google.genai.caches import AsyncCaches, Caches
            from google.genai.models import AsyncModels, Models
        except ImportError as err:
            raise Exception(
                "Could not import google-genai. Please install with `pip install google-genai`."
            ) from err
        from openinference.instrumentation.google_genai._wrappers import (
            _AsyncCreateCachesWrapper,
            _AsyncCreateInteractionWrapper,
            _AsyncEmbedContentWrapper,
            _AsyncGenerateContentStream,
            _AsyncGenerateContentWrapper,
            _SyncCreateCachesWrapper,
            _SyncCreateInteractionWrapper,
            _SyncEmbedContentWrapper,
            _SyncGenerateContent,
            _SyncGenerateContentStream,
        )

        self._original_create_interactions_resource = InteractionsResource.create
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai._interactions.resources",
            "InteractionsResource.create",
            _SyncCreateInteractionWrapper(tracer=self._tracer),
        )

        self._original_create_caches = Caches.create
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai.caches",
            "Caches.create",
            _SyncCreateCachesWrapper(tracer=self._tracer),
        )

        self._original_async_create_caches = AsyncCaches.create
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai.caches",
            "AsyncCaches.create",
            _AsyncCreateCachesWrapper(tracer=self._tracer),
        )

        self._original_embed_content = Models.embed_content
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai.models",
            "Models.embed_content",
            _SyncEmbedContentWrapper(tracer=self._tracer),
        )

        self._original_async_embed_content = AsyncModels.embed_content
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai.models",
            "AsyncModels.embed_content",
            _AsyncEmbedContentWrapper(tracer=self._tracer),
        )

        self._original_generate_content = Models.generate_content
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai.models",
            "Models.generate_content",
            _SyncGenerateContent(tracer=self._tracer),
        )

        self._original_async_generate_content = AsyncModels.generate_content
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai.models",
            "AsyncModels.generate_content",
            _AsyncGenerateContentWrapper(tracer=self._tracer),
        )

        self._original_generate_content_stream = Models.generate_content_stream
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai.models",
            "Models.generate_content_stream",
            _SyncGenerateContentStream(tracer=self._tracer),
        )

        self._original_async_generate_content_stream = AsyncModels.generate_content_stream
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai.models",
            "AsyncModels.generate_content_stream",
            _AsyncGenerateContentStream(tracer=self._tracer),
        )
        self._original_async_create_interactions_resource = AsyncInteractionsResource.create
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai._interactions.resources",
            "AsyncInteractionsResource.create",
            _AsyncCreateInteractionWrapper(tracer=self._tracer),
        )

        from google.genai._api_client import BaseApiClient

        from openinference.instrumentation.google_genai._context import (
            _CapturedRequestWrapper,
        )

        self._original_api_request = BaseApiClient.request
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai._api_client",
            "BaseApiClient.request",
            _CapturedRequestWrapper(),
        )

        self._original_api_request_streamed = BaseApiClient.request_streamed
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai._api_client",
            "BaseApiClient.request_streamed",
            _CapturedRequestWrapper(),
        )

        self._original_api_async_request = BaseApiClient.async_request
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai._api_client",
            "BaseApiClient.async_request",
            _CapturedRequestWrapper(),
        )

        self._original_api_async_request_streamed = BaseApiClient.async_request_streamed
        wrap_function_wrapper(  # type: ignore[no-untyped-call]
            "google.genai._api_client",
            "BaseApiClient.async_request_streamed",
            _CapturedRequestWrapper(),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        from google.genai._interactions.resources import (
            AsyncInteractionsResource,
            InteractionsResource,
        )
        from google.genai.caches import AsyncCaches, Caches
        from google.genai.models import AsyncModels, Models

        if self._original_embed_content is not None:
            setattr(Models, "embed_content", self._original_embed_content)

        if self._original_async_embed_content is not None:
            setattr(AsyncModels, "embed_content", self._original_async_embed_content)

        if self._original_generate_content is not None:
            setattr(Models, "generate_content", self._original_generate_content)

        if self._original_async_generate_content is not None:
            setattr(AsyncModels, "generate_content", self._original_async_generate_content)

        if self._original_generate_content_stream is not None:
            setattr(Models, "generate_content_stream", self._original_generate_content_stream)

        if self._original_create_caches is not None:
            setattr(Caches, "create", self._original_create_caches)

        if self._original_async_create_caches is not None:
            setattr(AsyncCaches, "create", self._original_async_create_caches)

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

        from google.genai._api_client import BaseApiClient

        if self._original_api_request is not None:
            setattr(BaseApiClient, "request", self._original_api_request)

        if self._original_api_request_streamed is not None:
            setattr(BaseApiClient, "request_streamed", self._original_api_request_streamed)

        if self._original_api_async_request is not None:
            setattr(BaseApiClient, "async_request", self._original_api_async_request)

        if self._original_api_async_request_streamed is not None:
            setattr(
                BaseApiClient, "async_request_streamed", self._original_api_async_request_streamed
            )
