import json
import logging
from contextlib import contextmanager
from inspect import Parameter, Signature, signature
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    Document,
    OITracer,
    get_attributes_from_context,
    get_input_attributes,
    get_metadata_attributes,
    get_output_attributes,
    get_retriever_attributes,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Identifiers that scope a mem0 operation. `add` and `delete_all` on `Memory` take them as
# keyword arguments, `search` and `get_all` take them inside `filters`, and the platform
# client accepts them in either place (or inside a pydantic `options` model).
_SCOPE_KEYS = ("user_id", "agent_id", "run_id", "app_id")

# Fields of a memory item that are mapped onto first-class document attributes rather than
# being repeated in the document metadata.
_DOCUMENT_FIELDS = ("id", "memory", "score")


def _to_jsonable(value: Any) -> Any:
    """Converts pydantic option models (used by ``MemoryClient``) to plain data."""
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return value.model_dump(exclude_unset=True)
        except Exception:
            return str(value)
    return value


def _call_arguments(
    sig: Signature, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Dict[str, Any]:
    """Returns the arguments explicitly passed by the caller, keyed by parameter name.

    Defaults are not applied, so the span only records what the caller supplied. A
    ``**kwargs`` catch-all and a ``MemoryClient`` ``options`` model are flattened into the
    top level, with explicit keyword arguments winning as they do inside mem0.
    """
    arguments = dict(sig.bind_partial(*args, **kwargs).arguments)
    for name, param in sig.parameters.items():
        if param.kind is Parameter.VAR_KEYWORD and name in arguments:
            arguments.update(arguments.pop(name))
    options = arguments.pop("options", None)
    if options is not None:
        options = _to_jsonable(options)
        if isinstance(options, Mapping):
            arguments = {**options, **arguments}
        else:
            arguments["options"] = options
    return {key: _to_jsonable(value) for key, value in arguments.items()}


def _scope_metadata(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Collects ``user_id`` / ``agent_id`` / ``run_id`` / ``app_id`` from the call."""
    scope: Dict[str, Any] = {}
    filters = arguments.get("filters")
    if isinstance(filters, Mapping):
        for key in _SCOPE_KEYS:
            if isinstance(value := filters.get(key), (str, int)):
                scope[key] = value
    for key in _SCOPE_KEYS:
        if isinstance(value := arguments.get(key), (str, int)):
            scope[key] = value
    return scope


def _context_metadata() -> Optional[Dict[str, Any]]:
    """Returns metadata set via ``using_metadata``, or None if it cannot be merged."""
    for key, value in get_attributes_from_context():
        if key != SpanAttributes.METADATA:
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return None
        return dict(value) if isinstance(value, Mapping) else None
    return {}


def _metadata_attributes(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    if not (scope := _scope_metadata(arguments)):
        return
    context_metadata = _context_metadata()
    if context_metadata is None:
        # Context metadata that is not a JSON object cannot be merged; leave it intact.
        return
    # Metadata set explicitly through `using_metadata` takes precedence.
    yield from get_metadata_attributes(metadata={**scope, **context_metadata}).items()


def _input_attributes(
    method_name: str, arguments: Mapping[str, Any]
) -> Iterator[Tuple[str, AttributeValue]]:
    query = arguments.get("query")
    if method_name == "search" and isinstance(query, str):
        yield from get_input_attributes(query, mime_type=OpenInferenceMimeTypeValues.TEXT).items()
    elif arguments:
        yield from get_input_attributes(
            dict(arguments), mime_type=OpenInferenceMimeTypeValues.JSON
        ).items()


def _memory_items(response: Any) -> List[Mapping[str, Any]]:
    results = response.get("results") if isinstance(response, Mapping) else response
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, Mapping)]


def _document(item: Mapping[str, Any]) -> Document:
    document = Document()
    if isinstance(memory_id := item.get("id"), (str, int)):
        document["id"] = memory_id
    if isinstance(memory := item.get("memory"), str):
        document["content"] = memory
    if isinstance(score := item.get("score"), (int, float)) and not isinstance(score, bool):
        document["score"] = float(score)
    metadata = {
        key: value
        for key, value in item.items()
        if key not in _DOCUMENT_FIELDS and value is not None
    }
    if metadata:
        document["metadata"] = metadata
    return document


def _output_attributes(
    span_kind: OpenInferenceSpanKindValues, response: Any
) -> Iterator[Tuple[str, AttributeValue]]:
    if span_kind is OpenInferenceSpanKindValues.RETRIEVER:
        documents = [_document(item) for item in _memory_items(response)]
        yield from get_retriever_attributes(documents=documents).items()
        return
    if response is None:
        return
    if isinstance(response, str):
        yield from get_output_attributes(
            response, mime_type=OpenInferenceMimeTypeValues.TEXT
        ).items()
    else:
        yield from get_output_attributes(
            response, mime_type=OpenInferenceMimeTypeValues.JSON
        ).items()


def _finish(span: Span, span_kind: OpenInferenceSpanKindValues, response: Any) -> None:
    """Records the response attributes and marks the span ``OK``."""
    # Sampled-out spans discard every attribute, so skip the serialization entirely.
    if span.is_recording():
        try:
            span.set_attributes(dict(_output_attributes(span_kind, response)))
        except Exception:
            logger.exception("Failed to extract mem0 response attributes")
    span.set_status(Status(StatusCode.OK))


def _fail(span: Span, exception: BaseException) -> None:
    """Records the exception and marks the span ``ERROR``."""
    span.record_exception(exception)
    span.set_status(Status(StatusCode.ERROR, f"{type(exception).__name__}: {exception}"))


class _WithTracer:
    """Base class for the sync and async memory operation wrappers."""

    def __init__(
        self,
        tracer: OITracer,
        span_name: str,
        method_name: str,
        span_kind: OpenInferenceSpanKindValues,
    ) -> None:
        self._tracer = tracer
        self._span_name = span_name
        self._method_name = method_name
        self._span_kind = span_kind
        self._signature: Optional[Signature] = None

    @contextmanager
    def _start_span(
        self,
        wrapped: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Iterator[Span]:
        """Starts the span for one memory operation and records the request attributes."""
        input_attributes, metadata_attributes = self._request_attributes(wrapped, args, kwargs)
        with self._tracer.start_as_current_span(
            self._span_name,
            attributes=input_attributes,
            openinference_span_kind=self._span_kind,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            # OITracer applies context attributes after the ones passed at start, so the
            # merged metadata is set once the span exists to keep the mem0 scope ids.
            if metadata_attributes:
                span.set_attributes(metadata_attributes)
            yield span

    def _request_attributes(
        self,
        wrapped: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Tuple[Dict[str, AttributeValue], Dict[str, AttributeValue]]:
        """Returns the input and metadata attributes; both are empty if extraction fails."""
        try:
            # Each wrapper instance wraps exactly one method, so the signature is computed
            # once. `wrapped` is the bound method, so `self` is not part of it.
            if self._signature is None:
                self._signature = signature(wrapped)
            arguments = _call_arguments(self._signature, args, kwargs)
            return (
                dict(_input_attributes(self._method_name, arguments)),
                dict(_metadata_attributes(arguments)),
            )
        except Exception:
            logger.exception("Failed to extract mem0 request attributes")
            return {}, {}


class _MemoryOperationWrapper(_WithTracer):
    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        with self._start_span(wrapped, args, kwargs) as span:
            try:
                response = wrapped(*args, **kwargs)
            except BaseException as exception:
                _fail(span, exception)
                raise
            _finish(span, self._span_kind, response)
            return response


class _AsyncMemoryOperationWrapper(_WithTracer):
    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        with self._start_span(wrapped, args, kwargs) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except BaseException as exception:
                _fail(span, exception)
                raise
            _finish(span, self._span_kind, response)
            return response
