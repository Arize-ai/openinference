import logging
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    Union,
)

from openinference.instrumentation.openai._utils import _finish_tracing
from openinference.instrumentation.openai._with_span import _WithSpan
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openai import AsyncStream, Stream

__all__ = (
    "_Stream",
    "_ResponseAccumulator",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAccumulator(Protocol):
    def process_chunk(self, chunk: Any) -> None:
        ...

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        ...

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        ...


class _Stream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_self_with_span",
        "_self_iteration_count",
        "_self_is_finished",
        "_self_include_extra_attributes",
        "_self_response_accumulator",
    )

    def __init__(
        self,
        stream: Union[Stream[Any], AsyncStream[Any]],
        with_span: _WithSpan,
        response_accumulator: Optional[_ResponseAccumulator] = None,
        include_extra_attributes: bool = True,
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_iteration_count = 0
        self._self_is_finished = with_span.is_finished
        self._self_include_extra_attributes = include_extra_attributes
        self._self_response_accumulator = response_accumulator

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        # pass through mistaken calls
        if not hasattr(self.__wrapped__, "__next__"):
            self.__wrapped__.__next__()
        iteration_is_finished = False
        status_code: Optional[trace_api.StatusCode] = None
        try:
            chunk: Any = self.__wrapped__.__next__()
        except Exception as exception:
            iteration_is_finished = True
            if isinstance(exception, StopIteration):
                status_code = trace_api.StatusCode.OK
            else:
                status_code = trace_api.StatusCode.ERROR
                self._self_with_span.record_exception(exception)
            raise
        else:
            self._process_chunk(chunk)
            status_code = trace_api.StatusCode.OK
            return chunk
        finally:
            if iteration_is_finished and not self._self_is_finished:
                self._finish_tracing(status_code=status_code)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        # pass through mistaken calls
        if not hasattr(self.__wrapped__, "__anext__"):
            self.__wrapped__.__anext__()
        iteration_is_finished = False
        status_code: Optional[trace_api.StatusCode] = None
        try:
            chunk: Any = await self.__wrapped__.__anext__()
        except Exception as exception:
            iteration_is_finished = True
            if isinstance(exception, StopAsyncIteration):
                status_code = trace_api.StatusCode.OK
            else:
                status_code = trace_api.StatusCode.ERROR
                self._self_with_span.record_exception(exception)
            raise
        else:
            self._process_chunk(chunk)
            status_code = trace_api.StatusCode.OK
            return chunk
        finally:
            if iteration_is_finished and not self._self_is_finished:
                self._finish_tracing(status_code=status_code)

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
        status_code: Optional[trace_api.StatusCode] = None,
    ) -> None:
        _finish_tracing(
            status_code=status_code,
            with_span=self._self_with_span,
            has_attributes=self,
        )
        self._self_is_finished = True

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if self._self_response_accumulator is not None:
            yield from self._self_response_accumulator.get_attributes()

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if self._self_include_extra_attributes and self._self_response_accumulator is not None:
            yield from self._self_response_accumulator.get_extra_attributes()
