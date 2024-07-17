from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

import pytest
from openinference.instrumentation.vertexai._proxy import _proxy


class Err(BaseException): ...


T = TypeVar("T")
Item = Optional[Union[str, int, bool]]
ITEMS: List[Item] = ["a", "", None, False, 0, 1]
MSG = "MSG"


class ItemsIterable:
    def __iter__(self) -> Iterator[Item]:
        return iter(ITEMS)


class ItemsIteratorWithError:
    def __init__(self) -> None:
        self._it = iter(ITEMS)

    def __iter__(self) -> Iterator[Item]:
        return self

    def __next__(self) -> Item:
        try:
            return next(self._it)
        except StopIteration:
            raise Err(MSG)


class ItemsIterableWithError:
    def __iter__(self) -> Iterator[Item]:
        return ItemsIteratorWithError()


@dataclass
class SimpleCallback:
    def __call__(self, obj: T) -> T:
        self.result = obj
        return obj


@dataclass
class AppendCallback:
    result: List[Union[Item, Err]] = field(init=False, default_factory=list)

    def __call__(self, obj: T) -> T:
        if isinstance(obj, (str, int, bool, type(None), Err)):
            self.result.append(obj)
        return obj


@dataclass
class CallbackWithError:
    def __call__(self, _: T) -> T:
        raise RuntimeError


@pytest.mark.parametrize("items_factory", [ItemsIterable, ItemsIterableWithError])
@pytest.mark.parametrize("cb_factory", [SimpleCallback, CallbackWithError])
async def test_coroutine(
    items_factory: Callable[[], Iterable[Item]],
    cb_factory: Callable[[], Callable[[T], T]],
) -> None:
    items, cb = items_factory(), cb_factory()

    async def foo() -> List[Item]:
        return list(items)

    assert isinstance((x := foo()), Awaitable)
    assert isinstance((p := _proxy(x, cb)), Awaitable)
    assert isinstance(p, type(x))
    assert p.__class__ is type(x)

    if isinstance(items, ItemsIterable):
        assert (await p) == ITEMS
        if isinstance(cb, SimpleCallback):
            assert cb.result == ITEMS
    elif isinstance(items, ItemsIterableWithError):
        with pytest.raises(Err) as e:
            await p
        assert str(e.value) == MSG
        if isinstance(cb, SimpleCallback):
            assert cb.result == e.value


def by_comp(obj: Iterable[Item]) -> List[Item]:
    return [_ for _ in obj]


def by_iter(obj: Iterable[Item]) -> List[Item]:
    return list(iter(obj))


def by_next(obj: Iterator[Item]) -> List[Item]:
    ans = []
    while True:
        try:
            item = next(obj)
        except StopIteration:
            break
        ans.append(item)
    return ans


def by_send(obj: Generator[Item, None, Any]) -> List[Item]:
    ans = []
    while True:
        try:
            item = obj.send(None)
        except StopIteration:
            break
        ans.append(item)
    return ans


async def by_acomp(obj: AsyncIterable[Item]) -> List[Item]:
    return [_ async for _ in obj]


async def by_aiter(obj: AsyncIterable[Item]) -> List[Item]:
    return [_ async for _ in obj.__aiter__()]


async def by_anext(obj: AsyncIterator[Item]) -> List[Item]:
    ans = []
    while True:
        try:
            task = obj.__anext__()
            item = await task
        except StopAsyncIteration:
            break
        ans.append(item)
    return ans


async def by_asend(obj: AsyncGenerator[Item, None]) -> List[Item]:
    ans = []
    while True:
        try:
            task = obj.asend(None)
            item = await task
        except StopAsyncIteration:
            break
        ans.append(item)
    return ans


@pytest.mark.parametrize("items_factory", [ItemsIterable, ItemsIterableWithError])
@pytest.mark.parametrize("cb_factory", [AppendCallback, CallbackWithError])
@pytest.mark.parametrize("list_fn", [list, by_comp, by_iter, by_next, by_send])
def test_generator(
    items_factory: Callable[[], Iterable[Item]],
    cb_factory: Callable[[], Callable[[T], T]],
    list_fn: Callable[[Iterable[Item]], List[Item]],
) -> None:
    items, cb = items_factory(), cb_factory()

    assert isinstance((x := (_ for _ in items)), Generator)
    assert isinstance((p := _proxy(x, cb)), Generator)
    assert isinstance(p, type(x))
    assert p.__class__ is type(x)

    if isinstance(items, ItemsIterable):
        assert list_fn(p) == ITEMS
        if isinstance(cb, AppendCallback):
            assert cb.result == ITEMS
    elif isinstance(items, ItemsIterableWithError):
        with pytest.raises(Err) as e:
            list_fn(p)
        assert str(e.value) == MSG
        if isinstance(cb, AppendCallback):
            assert cb.result[:-1] == ITEMS
            assert cb.result[-1] is e.value


@pytest.mark.parametrize("items_factory", [ItemsIterable, ItemsIterableWithError])
@pytest.mark.parametrize("cb_factory", [AppendCallback, CallbackWithError])
@pytest.mark.parametrize("list_fn", [list, by_comp, by_iter, by_next])
def test_iterator(
    items_factory: Callable[[], Iterable[Item]],
    cb_factory: Callable[[], Callable[[T], T]],
    list_fn: Callable[[Iterable[Item]], List[Item]],
) -> None:
    items, cb = items_factory(), cb_factory()

    assert isinstance((x := iter(items)), Iterator) and not isinstance(x, Generator)
    assert isinstance((p := _proxy(x, cb)), Iterator) and not isinstance(p, Generator)
    assert isinstance(p, type(x))
    assert p.__class__ is type(x)

    if isinstance(items, ItemsIterable):
        assert list_fn(p) == ITEMS
        if isinstance(cb, AppendCallback):
            assert cb.result == ITEMS
    elif isinstance(items, ItemsIterableWithError):
        with pytest.raises(Err) as e:
            list_fn(p)
        assert str(e.value) == MSG
        if isinstance(cb, AppendCallback):
            assert cb.result[:-1] == ITEMS
            assert cb.result[-1] is e.value


@pytest.mark.parametrize("items_factory", [ItemsIterable, ItemsIterableWithError])
@pytest.mark.parametrize("cb_factory", [AppendCallback, CallbackWithError])
@pytest.mark.parametrize("list_fn", [list, by_comp, by_iter])
def test_iterable(
    items_factory: Callable[[], Iterable[Item]],
    cb_factory: Callable[[], Callable[[T], T]],
    list_fn: Callable[[Iterable[Item]], List[Item]],
) -> None:
    items, cb = items_factory(), cb_factory()

    assert isinstance((x := items), Iterable) and not isinstance(x, Iterator)
    assert isinstance((p := _proxy(x, cb)), Iterable) and not isinstance(p, Iterator)
    assert isinstance(p, type(x))
    assert p.__class__ is type(x)

    if isinstance(items, ItemsIterable):
        assert list_fn(p) == ITEMS
        if isinstance(cb, AppendCallback):
            assert cb.result == ITEMS
    elif isinstance(items, ItemsIterableWithError):
        with pytest.raises(Err) as e:
            list_fn(p)
        assert str(e.value) == MSG
        if isinstance(cb, AppendCallback):
            assert cb.result[:-1] == ITEMS
            assert cb.result[-1] is e.value


@pytest.mark.parametrize("items_factory", [ItemsIterable, ItemsIterableWithError])
@pytest.mark.parametrize("cb_factory", [AppendCallback, CallbackWithError])
@pytest.mark.parametrize("list_fn", [by_acomp, by_aiter, by_anext, by_asend])
async def test_async_generator(
    items_factory: Callable[[], Iterable[Item]],
    cb_factory: Callable[[], Callable[[T], T]],
    list_fn: Callable[[AsyncIterable[Item]], Awaitable[List[Item]]],
) -> None:
    items, cb = items_factory(), cb_factory()

    async def foo() -> AsyncGenerator[Item, None]:
        for _ in items:
            yield _

    assert isinstance((x := foo()), AsyncGenerator)
    assert isinstance((p := _proxy(x, cb)), AsyncGenerator)
    assert isinstance(p, type(x))
    assert p.__class__ is type(x)

    if isinstance(items, ItemsIterable):
        assert await list_fn(p) == ITEMS
        if isinstance(cb, AppendCallback):
            assert cb.result == ITEMS
    elif isinstance(items, ItemsIterableWithError):
        with pytest.raises(Err) as e:
            await list_fn(p)
        assert str(e.value) == MSG
        if isinstance(cb, AppendCallback):
            assert cb.result[:-1] == ITEMS
            assert cb.result[-1] is e.value


@pytest.mark.parametrize("items_factory", [ItemsIterable, ItemsIterableWithError])
@pytest.mark.parametrize("cb_factory", [AppendCallback, CallbackWithError])
@pytest.mark.parametrize("list_fn", [by_acomp, by_aiter, by_anext])
async def test_async_iterator(
    items_factory: Callable[[], Iterable[Item]],
    cb_factory: Callable[[], Callable[[T], T]],
    list_fn: Callable[[AsyncIterable[Item]], Awaitable[List[Item]]],
) -> None:
    items, cb = items_factory(), cb_factory()
    it = iter(items)

    async def _next() -> Item:
        try:
            return next(it)
        except StopIteration:
            raise StopAsyncIteration

    foo = type("", (), dict(__aiter__=lambda _: _, __anext__=lambda _: _next()))

    assert isinstance((x := foo()), AsyncIterator) and not isinstance(x, AsyncGenerator)
    assert isinstance((p := _proxy(x, cb)), AsyncIterator) and not isinstance(p, AsyncGenerator)
    assert isinstance(p, type(x))
    assert p.__class__ is type(x)

    if isinstance(items, ItemsIterable):
        assert await list_fn(p) == ITEMS
        if isinstance(cb, AppendCallback):
            assert cb.result == ITEMS
    elif isinstance(items, ItemsIterableWithError):
        with pytest.raises(Err) as e:
            await list_fn(p)
        assert str(e.value) == MSG
        if isinstance(cb, AppendCallback):
            assert cb.result[:-1] == ITEMS
            assert cb.result[-1] is e.value


@pytest.mark.parametrize("items_factory", [ItemsIterable, ItemsIterableWithError])
@pytest.mark.parametrize("cb_factory", [AppendCallback, CallbackWithError])
@pytest.mark.parametrize("list_fn", [by_acomp, by_aiter])
async def test_async_iterable(
    items_factory: Callable[[], Iterable[Item]],
    cb_factory: Callable[[], Callable[[T], T]],
    list_fn: Callable[[AsyncIterable[Item]], Awaitable[List[Item]]],
) -> None:
    items, cb = items_factory(), cb_factory()
    it = iter(items)

    async def _next() -> Item:
        try:
            return next(it)
        except StopIteration:
            raise StopAsyncIteration

    foo = type("", (), dict(__aiter__=lambda _: _, __anext__=lambda _: _next()))
    bar = type("", (), dict(__aiter__=lambda _: foo()))

    assert isinstance((x := bar()), AsyncIterable) and not isinstance(x, AsyncIterator)
    assert isinstance((p := _proxy(x, cb)), AsyncIterable) and not isinstance(p, AsyncIterator)
    assert isinstance(p, type(x))
    assert p.__class__ is type(x)

    if isinstance(items, ItemsIterable):
        assert await list_fn(p) == ITEMS
        if isinstance(cb, AppendCallback):
            assert cb.result == ITEMS
    elif isinstance(items, ItemsIterableWithError):
        with pytest.raises(Err) as e:
            await list_fn(p)
        assert str(e.value) == MSG
        if isinstance(cb, AppendCallback):
            assert cb.result[:-1] == ITEMS
            assert cb.result[-1] is e.value
