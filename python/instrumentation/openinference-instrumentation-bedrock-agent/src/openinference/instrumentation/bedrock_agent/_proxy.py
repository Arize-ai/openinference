import inspect
import logging
from random import getrandbits
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    ContextManager,
    Generator,
    Generic,
    Iterable,
    Iterator,
    Optional,
    TypeVar,
    cast,
)

import wrapt
from typing_extensions import TypeAlias

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ("_proxy",)

_AnyT = TypeVar("_AnyT")
_CallbackT: TypeAlias = Callable[[_AnyT], _AnyT]
_WrappedT = TypeVar("_WrappedT")
_T_co = TypeVar("_T_co", covariant=True)
_YieldT_co = TypeVar("_YieldT_co", covariant=True)
_SendT_contra = TypeVar("_SendT_contra", contravariant=True)
_ReturnT_co = TypeVar("_ReturnT_co", covariant=True)


def _proxy(
    obj: _WrappedT,
    callback: Optional[_CallbackT[_AnyT]] = None,
    context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
) -> _WrappedT:
    if callback is None and context_manager_factory is None:
        return obj
    if getattr(obj, _SELF_IS_PROXY, False):
        return obj
    if isinstance(obj, Awaitable):
        return _Awaitable(obj, callback, context_manager_factory)
    if isinstance(obj, AsyncGenerator):
        return _AsyncGenerator(obj, callback, context_manager_factory)
    if isinstance(obj, Generator):
        return _Generator(obj, callback, context_manager_factory)
    if isinstance(obj, AsyncIterator):
        return _AsyncIterator(obj, callback, context_manager_factory)
    if isinstance(obj, Iterator):
        return _Iterator(obj, callback, context_manager_factory)
    if isinstance(obj, AsyncIterable):
        return _AsyncIterable(obj, callback, context_manager_factory)
    if isinstance(obj, Iterable):
        return _Iterable(obj, callback, context_manager_factory)
    return obj


class _Proxy(wrapt.ObjectProxy):  # type: ignore[misc]
    def __init__(
        self,
        wrapped: _WrappedT,
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
    ) -> None:
        super().__init__(wrapped)
        setattr(self, _SELF_IS_PROXY, True)
        self._self_callback: _CallbackT[_AnyT] = _no_err(callback)
        # The `use_span` context manager can't be entered more than once. It would err here:
        # https://github.com/open-telemetry/opentelemetry-python/blob/b1e99c1555721f818e578d7457587693e767e182/opentelemetry-api/src/opentelemetry/util/_decorator.py#L56  # noqa E501
        # So we need a factory.
        self._self_context_manager_factory: Callable[[], ContextManager[Any]] = _no_err(
            context_manager_factory
        )


class _Awaitable(_Proxy, Awaitable[_T_co], Generic[_T_co]):
    def __init__(
        self,
        wrapped: Awaitable[_T_co],
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
    ) -> None:
        super().__init__(wrapped, callback, context_manager_factory)

    def __await__(self) -> Generator[Any, None, _T_co]:
        callback = self._self_callback
        generator = cast(Generator[Any, None, _T_co], self.__wrapped__.__await__())
        while True:
            context_manager = _no_err(self._self_context_manager_factory())
            try:
                with context_manager:
                    obj = generator.send(None)
            except StopIteration as exc:
                v = cast(_T_co, exc.value)
                return callback(v)  # type: ignore[arg-type,return-value]
            except BaseException as exc:
                callback(exc)  # type: ignore[arg-type]
                raise
            try:
                yield obj
            except BaseException as exc:
                callback(exc)  # type: ignore[arg-type]
                generator.throw(exc)


class _Iterable(_Proxy, Iterable[_T_co], Generic[_T_co]):
    def __init__(
        self,
        wrapped: Iterable[_T_co],
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
    ) -> None:
        super().__init__(wrapped, callback, context_manager_factory)

    def __iter__(self) -> Iterator[_T_co]:
        ans = cast(Iterator[_T_co], _call(self))
        return _Iterator(
            wrapped=ans,
            callback=self._self_callback,
            context_manager_factory=self._self_context_manager_factory,
        )


class _Iterator(_Iterable[_T_co], Iterator[_T_co], Generic[_T_co]):
    def __init__(
        self,
        wrapped: Iterator[_T_co],
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
    ) -> None:
        super().__init__(wrapped, callback, context_manager_factory)

    def __next__(self) -> _T_co:
        return cast(_T_co, _call(self))


class _Generator(
    _Iterator[_YieldT_co],
    Generator[_YieldT_co, _SendT_contra, _ReturnT_co],
    Generic[_YieldT_co, _SendT_contra, _ReturnT_co],
):
    def __init__(
        self,
        wrapped: Generator[_YieldT_co, _SendT_contra, _ReturnT_co],
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
    ) -> None:
        super().__init__(wrapped, callback, context_manager_factory)

    def __iter__(self) -> Generator[_YieldT_co, _SendT_contra, _ReturnT_co]:
        ans = cast(Generator[_YieldT_co, _SendT_contra, _ReturnT_co], _call(self))
        return _Generator(
            wrapped=ans,
            callback=self._self_callback,
            context_manager_factory=self._self_context_manager_factory,
        )

    def send(self, v: _SendT_contra) -> _YieldT_co:
        return cast(_YieldT_co, _call(self, v))

    def throw(self, *args: Any, **kwargs: Any) -> _YieldT_co:
        return cast(_YieldT_co, _call(self, *args, **kwargs))


class _AsyncIterable(_Proxy, AsyncIterable[_T_co], Generic[_T_co]):
    def __init__(
        self,
        wrapped: AsyncIterable[_T_co],
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
    ) -> None:
        super().__init__(wrapped, callback, context_manager_factory)

    def __aiter__(self) -> AsyncIterator[_T_co]:
        ans = cast(AsyncIterator[_T_co], _call(self))
        return _AsyncIterator(
            wrapped=ans,
            callback=self._self_callback,
            context_manager_factory=self._self_context_manager_factory,
        )


class _AsyncIterator(_AsyncIterable[_T_co], AsyncIterator[_T_co], Generic[_T_co]):
    def __init__(
        self,
        wrapped: AsyncIterator[_T_co],
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
    ) -> None:
        super().__init__(wrapped, callback, context_manager_factory)

    def __anext__(self) -> Awaitable[_T_co]:
        ans = cast(Awaitable[_T_co], _call(self))
        return _Awaitable(
            wrapped=ans,
            callback=self._self_callback,
            context_manager_factory=self._self_context_manager_factory,
        )


class _AsyncGenerator(
    _AsyncIterator[_YieldT_co],
    AsyncGenerator[_YieldT_co, _SendT_contra],
    Generic[_YieldT_co, _SendT_contra],
):
    def __init__(
        self,
        wrapped: AsyncGenerator[_YieldT_co, _SendT_contra],
        callback: Optional[_CallbackT[_AnyT]] = None,
        context_manager_factory: Optional[Callable[[], ContextManager[Any]]] = None,
    ) -> None:
        super().__init__(wrapped, callback, context_manager_factory)

    def asend(self, v: _SendT_contra) -> Awaitable[_YieldT_co]:
        ans = cast(Awaitable[_YieldT_co], _call(self, v))
        return _Awaitable(
            wrapped=ans,
            callback=self._self_callback,
            context_manager_factory=self._self_context_manager_factory,
        )

    def athrow(self, *args: Any, **kwargs: Any) -> Awaitable[_YieldT_co]:
        ans = cast(Awaitable[_YieldT_co], _call(self, *args, **kwargs))
        return _Awaitable(
            wrapped=ans,
            callback=self._self_callback,
            context_manager_factory=self._self_context_manager_factory,
        )


class _NoErr(wrapt.ObjectProxy):  # type: ignore[misc]
    def __init__(self, wrapped: Any) -> None:
        super().__init__(wrapped)
        setattr(self, _SELF_IS_NO_ERR, True)
        self._self_is_entered = False

    def __enter__(self) -> Any:
        if (
            not (cm := self.__wrapped__)
            or not isinstance(cm, ContextManager)
            or self._self_is_entered
        ):
            return
        try:
            return cm.__enter__()
        except BaseException as exc:
            logger.exception(exc)
        finally:
            self._self_is_entered = True

    def __exit__(self, *args: Any, **kwargs: Any) -> Any:
        if (
            not (cm := self.__wrapped__)
            or not isinstance(cm, ContextManager)
            or not self._self_is_entered
        ):
            return
        try:
            return cm.__exit__(*args, **kwargs)
        except BaseException as exc:
            logger.exception(exc)
        finally:
            self._self_is_entered = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ans = args[0] if args else (next(iter(kwargs.values())) if kwargs else None)
        if (fn := self.__wrapped__) and callable(fn):
            try:
                ans = fn(*args, **kwargs)
            except BaseException as exc:
                logger.exception(exc)
        return ans


def _no_err(obj: Optional[_T_co]) -> _T_co:
    if getattr(obj, _SELF_IS_NO_ERR, False):
        return cast(_T_co, obj)
    return _NoErr(obj)


def _call(self: _Proxy, *args: Any, **kwargs: Any) -> Any:
    name = inspect.stack()[1].function
    fn = getattr(self.__wrapped__, name)
    callback = self._self_callback
    context_manager = _no_err(self._self_context_manager_factory())
    try:
        with context_manager:
            ans = fn(*args, **kwargs)
    except BaseException as exc:
        callback(exc)  # type: ignore[arg-type]
        raise
    return callback(ans)


_SUFFIX = f"_{getrandbits(64).to_bytes(8, 'big').hex()}"
_SELF_IS_NO_ERR = f"_self_is_no_err{_SUFFIX}"
_SELF_IS_PROXY = f"_self_is_proxy{_SUFFIX}"
