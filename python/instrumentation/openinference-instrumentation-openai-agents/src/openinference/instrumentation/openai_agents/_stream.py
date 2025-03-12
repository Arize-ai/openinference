import logging
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    Union,
)

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openinference.instrumentation.openai._utils import _finish_tracing
from openinference.instrumentation.openai._with_span import _WithSpan

if TYPE_CHECKING:
    from openai import AsyncStream, Stream

__all__ = (
    "_Stream",
    "_ResponseAccumulator",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAccumulator(Protocol):
    def process_chunk(self, chunk: Any) -> None: ...

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...


class _Stream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_self_with_span",
        "_self_iteration_count",
        "_self_is_finished",
        "_self_response_accumulator",
    )

    def __init__(
        self,
        stream: Union["Stream[Any]", "AsyncStream[Any]"],
        with_span: _WithSpan,
        response_accumulator: Optional[_ResponseAccumulator] = None,
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_iteration_count = 0
        self._self_is_finished = with_span.is_finished
        self._self_response_accumulator = response_accumulator

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        # pass through mistaken calls
        if not hasattr(self.__wrapped__, "__next__"):
            self.__wrapped__.__next__()
        try:
            chunk: Any = self.__wrapped__.__next__()
        except Exception as exception:
            if not self._self_is_finished:
                if isinstance(exception, StopIteration):
                    status = trace_api.Status(status_code=trace_api.StatusCode.OK)
                else:
                    status = trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        # Follow the format in OTEL SDK for description, see:
                        # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                        description=f"{type(exception).__name__}: {exception}",
                    )
                    self._self_with_span.record_exception(exception)
                self._finish_tracing(status=status)
            raise
        else:
            self._process_chunk(chunk)
            return chunk

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        # pass through mistaken calls
        if not hasattr(self.__wrapped__, "__anext__"):
            self.__wrapped__.__anext__()
        try:
            chunk: Any = await self.__wrapped__.__anext__()
        except Exception as exception:
            if not self._self_is_finished:
                if isinstance(exception, StopAsyncIteration):
                    status = trace_api.Status(status_code=trace_api.StatusCode.OK)
                else:
                    status = trace_api.Status(
                        status_code=trace_api.StatusCode.ERROR,
                        # Follow the format in OTEL SDK for description, see:
                        # https://github.com/open-telemetry/opentelemetry-python/blob/2b9dcfc5d853d1c10176937a6bcaade54cda1a31/opentelemetry-api/src/opentelemetry/trace/__init__.py#L588  # noqa E501
                        description=f"{type(exception).__name__}: {exception}",
                    )
                    self._self_with_span.record_exception(exception)
                self._finish_tracing(status=status)
            raise
        else:
            self._process_chunk(chunk)
            return chunk

    def __enter__(self) -> Any:
        # Stream response can be used as a context manager. For example, see here
        # https://github.com/langchain-ai/langchain/blob/dc42279eb55fbb8ec5175d24c7b30fe7b502b6d1/libs/partners/openai/langchain_openai/chat_models/base.py#L513  # noqa E501
        # in LangChain. When that happens, the __enter__ method on the wrapped
        # object is called and the stream object escapes our wrapper. See here
        # https://github.com/openai/openai-python/blob/435a5805ccbd5939a68f7f359ab72e937ef86e59/src/openai/_streaming.py#L103-L104  # noqa E501
        # We override the __enter__ method so the wrapped object does not escape.
        obj = self.__wrapped__.__enter__()
        if obj is self.__wrapped__:
            return self
        return obj

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.__wrapped__.__exit__(*args, **kwargs)

    async def __aenter__(self) -> Any:
        obj = await self.__wrapped__.__aenter__()
        if obj is self.__wrapped__:
            return self
        return obj

    async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
        await self.__wrapped__.__aexit__(*args, **kwargs)

    def _process_chunk(self, chunk: Any) -> None:
        if not self._self_iteration_count:
            try:
                self._self_with_span.add_event("First Token Stream Event")
            except Exception:
                logger.exception("Failed to add event to span")
        self._self_iteration_count += 1
        if self._self_response_accumulator is not None:
            try:
                self._self_response_accumulator.process_chunk(chunk)
            except Exception:
                logger.exception("Failed to accumulate response")

    def _finish_tracing(
        self,
        status: Optional[trace_api.Status] = None,
    ) -> None:
        _finish_tracing(
            status=status,
            with_span=self._self_with_span,
            has_attributes=self,
        )
        self._self_is_finished = True

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if self._self_response_accumulator is not None:
            yield from self._self_response_accumulator.get_attributes()

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if self._self_response_accumulator is not None:
            yield from self._self_response_accumulator.get_extra_attributes()
