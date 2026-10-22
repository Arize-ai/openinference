import logging
from contextvars import Context, copy_context
from typing import Any, Awaitable, Callable, Generator, Optional

from opentelemetry import context as context_api
from wrapt import ObjectProxy

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ContextAwaitable:
    def __init__(self, context: Context, awaitable: Awaitable[Any]) -> None:
        self._context = context
        self._awaitable = awaitable

    def __await__(self) -> Generator[Any, Any, Any]:
        iterator = self._awaitable.__await__()
        try:
            value = self._context.run(iterator.send, None)
            while True:
                try:
                    received = yield value
                except BaseException as exception:
                    value = self._context.run(iterator.throw, exception)
                else:
                    value = self._context.run(iterator.send, received)
        except StopIteration as stop:
            return stop.value


class _ResponseStream(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    def __init__(
        self,
        stream: Any,
        context: context_api.Context,
        finish: Callable[[Optional[BaseException]], None],
        execution_context: Optional[Context] = None,
    ) -> None:
        super().__init__(stream)  # type: ignore[no-untyped-call]
        self._self_context = execution_context if execution_context is not None else copy_context()
        self._self_context.run(context_api.attach, context)
        self._self_finish = finish
        self._self_finished = False

    def _finish(self, exception: Optional[BaseException] = None) -> None:
        if self._self_finished:
            return
        self._self_finished = True
        try:
            self._self_finish(None if isinstance(exception, GeneratorExit) else exception)
        except Exception:
            logger.exception("Failed to finish streaming span")

    def __iter__(self) -> "_ResponseStream":
        return self

    def __next__(self) -> Any:
        return self._self_context.run(self._run, self.__wrapped__.__next__)

    def send(self, value: Any) -> Any:
        return self._self_context.run(self._run, self.__wrapped__.send, value)

    def throw(self, *args: Any) -> Any:
        return self._self_context.run(self._run, self.__wrapped__.throw, *args)

    def _run(self, operation: Callable[..., Any], *args: Any) -> Any:
        try:
            return operation(*args)
        except (StopIteration, StopAsyncIteration):
            self._finish()
            raise
        except BaseException as exception:
            self._finish(exception)
            raise

    def close(self) -> None:
        self._self_context.run(self._close)

    def _close(self) -> None:
        self._run(self.__wrapped__.close)
        self._finish()

    def __aiter__(self) -> "_ResponseStream":
        return self

    def __anext__(self) -> Awaitable[Any]:
        return _ContextAwaitable(self._self_context, self._arun(self.__wrapped__.__anext__))

    def asend(self, value: Any) -> Awaitable[Any]:
        return _ContextAwaitable(self._self_context, self._arun(self.__wrapped__.asend, value))

    def athrow(self, *args: Any) -> Awaitable[Any]:
        return _ContextAwaitable(self._self_context, self._arun(self.__wrapped__.athrow, *args))

    async def _arun(self, operation: Callable[..., Awaitable[Any]], *args: Any) -> Any:
        try:
            return await operation(*args)
        except StopAsyncIteration:
            self._finish()
            raise
        except BaseException as exception:
            self._finish(exception)
            raise

    def aclose(self) -> Awaitable[None]:
        return _ContextAwaitable(self._self_context, self._aclose())

    async def _aclose(self) -> None:
        await self._arun(self.__wrapped__.aclose)
        self._finish()
