from collections import defaultdict
from copy import deepcopy
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

__all__ = (
    "_KeyValuesAccumulator",
    "_StringAccumulator",
    "_IndexedAccumulator",
)


class _KeyValuesAccumulator:
    __slots__ = ("_kv",)

    def __init__(self, **values: Any) -> None:
        self._kv: Dict[str, Any] = values

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for k, v in self._kv.items():
            if v is None:
                continue
            if isinstance(v, _KeyValuesAccumulator):
                if dict_value := dict(v):
                    yield k, dict_value
            elif isinstance(v, _IndexedAccumulator):
                if list_value := list(v):
                    yield k, list_value
            elif isinstance(v, _StringAccumulator):
                if str_value := str(v):
                    yield k, str_value
            else:
                yield k, v

    def __iadd__(self, values: Optional[Mapping[str, Any]]) -> "_KeyValuesAccumulator":
        if not values:
            return self
        for k in self._kv.keys():
            if (v := values.get(k)) is None:
                continue
            self_value = self._kv[k]
            if isinstance(self_value, _KeyValuesAccumulator):
                if isinstance(v, Mapping):
                    self_value += v
            elif isinstance(self_value, _StringAccumulator):
                if isinstance(v, str):
                    self_value += v
            elif isinstance(self_value, _IndexedAccumulator):
                if isinstance(v, Iterable):
                    for v in v:
                        self_value += v
                else:
                    self_value += v
            elif isinstance(self_value, List) and isinstance(v, Iterable):
                self_value.extend(v)
            else:
                self._kv[k] = v  # replacement
        for k in values.keys():
            if k in self._kv or (v := values[k]) is None:
                continue
            v = deepcopy(v)
            if isinstance(v, Mapping):
                v = _KeyValuesAccumulator(**v)
            self._kv[k] = v  # new entry
        return self


class _StringAccumulator:
    __slots__ = ("_fragments",)

    def __init__(self) -> None:
        self._fragments: List[str] = []

    def __str__(self) -> str:
        return "".join(self._fragments)

    def __iadd__(self, value: Optional[str]) -> "_StringAccumulator":
        if not value:
            return self
        self._fragments.append(value)
        return self


class _IndexedAccumulator:
    __slots__ = ("_indexed",)

    def __init__(self, kv_factory: Callable[[], _KeyValuesAccumulator]) -> None:
        self._indexed: DefaultDict[int, _KeyValuesAccumulator] = defaultdict(kv_factory)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for _, values in sorted(self._indexed.items()):
            yield dict(values)

    def __iadd__(self, values: Optional[Mapping[str, Any]]) -> "_IndexedAccumulator":
        if not values or not hasattr(values, "get"):
            return self
        index = values.get("index") or 0
        self._indexed[index] += values
        return self
