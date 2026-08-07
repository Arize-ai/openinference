import logging
from abc import ABC
from collections.abc import AsyncIterator as AsyncIteratorABC
from collections.abc import Iterator as IteratorABC
from contextlib import contextmanager
from inspect import Signature, signature
from typing import Any, AsyncIterator, Callable, Dict, Iterable, Iterator, List, Mapping, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation.ollama._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.ollama._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.ollama._utils import _finish_tracing
from openinference.instrumentation.ollama._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithTracer(ABC):
    """Base class for wrappers that need a tracer."""

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        attributes: Iterable[Tuple[str, AttributeValue]],
        context_attributes: Iterable[Tuple[str, AttributeValue]],
        extra_attributes: Iterable[Tuple[str, AttributeValue]],
    ) -> Iterator[_WithSpan]:
        # Because OTEL has a default limit of 128 attributes, we split our
        # attributes into two tiers, where "extra_attributes" are added first to
        # ensure that the most important "attributes" are added last and are not
        # dropped.
        try:
            span = self._tracer.start_span(name=span_name, attributes=dict(extra_attributes))
        except Exception:
            span = INVALID_SPAN
        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield _WithSpan(
                span=span,
                context_attributes=dict(context_attributes),
                extra_attributes=dict(attributes),
            )


def _parse_args(
    signature: Signature,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    bound_signature = signature.bind(*args, **kwargs)
    bound_signature.apply_defaults()
    return {
        key: value
        for key, value in bound_signature.arguments.items()
        if value is not None and key != "self"
    }


def _merge_chat_chunks(chunks: List[Any]) -> Any:
    """Combines streamed ``ChatResponse`` chunks into a single response-like
    object. The final chunk carries the model and token counts, so it is used
    as the base, with the message content accumulated across all chunks."""
    if not chunks:
        return None
    last = chunks[-1]
    try:
        merged = last.model_copy(deep=True)
        merged.message.content = "".join(
            content
            for chunk in chunks
            if (content := getattr(getattr(chunk, "message", None), "content", None))
        )
        tool_calls = [
            tool_call
            for chunk in chunks
            for tool_call in (getattr(getattr(chunk, "message", None), "tool_calls", None) or ())
        ]
        if tool_calls:
            merged.message.tool_calls = tool_calls
        return merged
    except Exception:
        logger.exception("Failed to merge streamed chat chunks")
        return last


class _ChatWrapperBase(_WithTracer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_extractor = _RequestAttributesExtractor()
        self._response_extractor = _ResponseAttributesExtractor()

    def _record_exception(self, span: _WithSpan, exception: Exception) -> None:
        span.record_exception(exception)
        span.finish_tracing(
            status=trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

    def _finalize_response(self, span: _WithSpan, response: Any) -> None:
        if response is None:
            span.finish_tracing(status=trace_api.Status(status_code=trace_api.StatusCode.OK))
            return
        try:
            _finish_tracing(
                status=trace_api.Status(status_code=trace_api.StatusCode.OK),
                with_span=span,
                attributes=self._response_extractor.get_attributes(response=response),
                extra_attributes=self._response_extractor.get_extra_attributes(response=response),
            )
        except Exception:
            logger.exception(f"Failed to finalize response of type {type(response)}")
            span.finish_tracing()

    def _wrap_stream(self, stream: Iterator[Any], span: _WithSpan) -> Iterator[Any]:
        chunks: List[Any] = []
        try:
            for chunk in stream:
                chunks.append(chunk)
                yield chunk
        except GeneratorExit:
            # The caller abandoned the stream: finish the span with what arrived.
            self._finalize_response(span, _merge_chat_chunks(chunks))
            raise
        except Exception as exception:
            self._record_exception(span, exception)
            raise
        self._finalize_response(span, _merge_chat_chunks(chunks))

    async def _wrap_async_stream(
        self, stream: AsyncIterator[Any], span: _WithSpan
    ) -> AsyncIterator[Any]:
        chunks: List[Any] = []
        try:
            async for chunk in stream:
                chunks.append(chunk)
                yield chunk
        except GeneratorExit:
            self._finalize_response(span, _merge_chat_chunks(chunks))
            raise
        except Exception as exception:
            self._record_exception(span, exception)
            raise
        self._finalize_response(span, _merge_chat_chunks(chunks))


class _ChatWrapper(_ChatWrapperBase):
    """Wraps ``ollama.Client.chat`` to trace synchronous chat calls."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        with self._start_as_current_span(
            span_name="chat",
            attributes=self._request_extractor.get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                request_parameters
            ),
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                self._record_exception(span, exception)
                raise
            if isinstance(response, IteratorABC):
                # ``stream=True``: the span is finished when the stream is
                # exhausted (or fails), not when the call returns.
                return self._wrap_stream(response, span)
            self._finalize_response(span, response)
        return response


class _AsyncChatWrapper(_ChatWrapperBase):
    """Wraps ``ollama.AsyncClient.chat`` to trace asynchronous chat calls."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        request_parameters = _parse_args(signature(wrapped), *args, **kwargs)
        with self._start_as_current_span(
            span_name="async_chat",
            attributes=self._request_extractor.get_attributes_from_request(request_parameters),
            context_attributes=get_attributes_from_context(),
            extra_attributes=self._request_extractor.get_extra_attributes_from_request(
                request_parameters
            ),
        ) as span:
            try:
                response = await wrapped(*args, **kwargs)
            except Exception as exception:
                self._record_exception(span, exception)
                raise
            if isinstance(response, AsyncIteratorABC):
                # ``stream=True``: the span is finished when the stream is
                # exhausted (or fails), not when the call returns.
                return self._wrap_async_stream(response, span)
            self._finalize_response(span, response)
        return response
