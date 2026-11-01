"""Runs each step of a streamed LLM call with the call's span current.

`stream_chat`, `stream_complete`, `astream_chat` and `astream_complete` return their stream
before any request is made: the request starts on the first step, after the dispatcher has
exited the span, and LlamaIndex runs no handler in that step. So these methods are wrapped on
every LLM class, including ones defined later, and each step of the returned stream runs with
the span current. The span is taken down before the chunk reaches the caller.
"""

import functools
import inspect
import logging
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
)

from opentelemetry import context as context_api
from opentelemetry.trace import Span, set_span_in_context

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_STREAM_METHODS = ("stream_chat", "stream_complete", "astream_chat", "astream_complete")
_ORIGINAL = "__openinference_original__"
_SUBCLASS_HOOK = "__pydantic_init_subclass__"

# Finds the span a stream was returned from, if any.
_find_span: Optional[Callable[[Any], Optional[Span]]] = None
_hooks: List[Tuple[Type[Any], Any]] = []


def _span_of(stream: Any) -> Optional[Span]:
    if (find := _find_span) is None:
        return None
    try:
        return find(stream)
    except Exception:
        logger.exception("Failed to find the span of an LLM stream")
        return None


def _attach(span: Span) -> object:
    # Built from the context of the step, so what the caller set around it (suppression,
    # baggage, context attributes) still holds; only the current span changes.
    return context_api.attach(set_span_in_context(span, context_api.get_current()))


# The streams of LLM methods are plain generators that LlamaIndex only iterates and closes,
# so a generator that does the same is enough here; it needs no proxy.
def _steps_under(span: Span, stream: Iterator[Any]) -> Generator[Any, None, None]:
    try:
        while True:
            token = _attach(span)
            try:
                item = next(stream)
            except StopIteration:
                return
            finally:
                context_api.detach(token)  # type: ignore[arg-type]
            yield item
    finally:
        if (close := getattr(stream, "close", None)) is not None:
            close()


async def _asteps_under(span: Span, stream: AsyncIterator[Any]) -> AsyncGenerator[Any, None]:
    try:
        while True:
            token = _attach(span)
            try:
                item = await stream.__anext__()
            except StopAsyncIteration:
                return
            finally:
                context_api.detach(token)  # type: ignore[arg-type]
            yield item
    finally:
        if (aclose := getattr(stream, "aclose", None)) is not None:
            await aclose()


def _wrap(original: Any) -> Any:
    @functools.wraps(original)
    def stream(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original.__get__(self, type(self))(*args, **kwargs)
        if (span := _span_of(result)) is None:
            return result
        return _steps_under(span, result)

    setattr(stream, _ORIGINAL, original)
    return stream


def _awrap(original: Any) -> Any:
    @functools.wraps(original)
    async def astream(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original.__get__(self, type(self))(*args, **kwargs)
        if (span := _span_of(result)) is None:
            return result
        return _asteps_under(span, result)

    setattr(astream, _ORIGINAL, original)
    return astream


def _wrapper_for(method: Any) -> Optional[Callable[[Any], Any]]:
    # By what the method is rather than by its name: an async generator function is called
    # without an await and is left alone.
    function = inspect.unwrap(method)
    if inspect.iscoroutinefunction(function):
        return _awrap
    if inspect.isfunction(function) and not inspect.isasyncgenfunction(function):
        return _wrap
    return None


def _wrap_class(cls: Type[Any]) -> None:
    for name in _STREAM_METHODS:
        if (method := cls.__dict__.get(name)) is None or hasattr(method, _ORIGINAL):
            continue
        if (wrap := _wrapper_for(method)) is not None:
            setattr(cls, name, wrap(method))


def _unwrap_class(cls: Type[Any]) -> None:
    for name in _STREAM_METHODS:
        if (method := cls.__dict__.get(name)) is not None and hasattr(method, _ORIGINAL):
            setattr(cls, name, getattr(method, _ORIGINAL))


def _with_subclasses(cls: Type[Any]) -> List[Type[Any]]:
    found = [cls]
    for each in found:
        found.extend(sub for sub in each.__subclasses__() if sub not in found)
    return found


def _subclass_hook(base: Type[Any]) -> Any:
    def hook(cls: Type[Any], /, **kwargs: Any) -> None:
        super(base, cls).__pydantic_init_subclass__(**kwargs)  # type: ignore[misc]
        try:
            _wrap_class(cls)
        except Exception:
            logger.exception(f"Failed to wrap the stream methods of {cls}")

    return classmethod(hook)


def instrument_llm_streams(
    bases: Tuple[Type[Any], ...],
    find_span: Callable[[Any], Optional[Span]],
) -> None:
    global _find_span
    _find_span = find_span
    for base in bases:
        for cls in _with_subclasses(base):
            _wrap_class(cls)
        # A hook already on the base class itself is left alone, along with the classes
        # defined after it.
        if base.__dict__.get(_SUBCLASS_HOOK) is None:
            hook = _subclass_hook(base)
            setattr(base, _SUBCLASS_HOOK, hook)
            _hooks.append((base, hook))


def uninstrument_llm_streams(bases: Tuple[Type[Any], ...]) -> None:
    global _find_span
    _find_span = None
    while _hooks:
        base, hook = _hooks.pop()
        if base.__dict__.get(_SUBCLASS_HOOK) is hook:
            delattr(base, _SUBCLASS_HOOK)
    for base in bases:
        for cls in _with_subclasses(base):
            _unwrap_class(cls)
