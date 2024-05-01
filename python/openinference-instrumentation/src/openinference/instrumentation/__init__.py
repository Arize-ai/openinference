from contextlib import AbstractContextManager, AbstractAsyncContextManager
from typing import Dict, Type, Sequence, List, Any
from openinference.semconv.trace import SpanAttributes
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    attach,
    detach,
    set_value,
    get_value,
    get_current,
)
from opentelemetry.util.types import AttributeValue
from typing import (
    Iterator,
    Tuple,
)

CONTEXT_ATTRIBUTES = (
    SpanAttributes.SESSION_ID,
    SpanAttributes.USER_ID,
    SpanAttributes.METADATA,
    SpanAttributes.TAG_TAGS,
)


class suppress_tracing(AbstractContextManager):
    slots = ["_token"]
    """
    Context manager to pause OpenTelemetry instrumentation.

    Examples:
        with suppress_tracing():
            # No tracing will occur within this block
            ...
    """

    def __enter__(self) -> "suppress_tracing":
        self._token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Raise any exception triggered within the runtime context."""
        detach(self._token)
        return None


class UsingAttributes:
    slots = [
        "_token",
        "_session_id",
        "_user_id",
        "_metadata",
        "_tags",
    ]

    def __init__(
        self,
        session_id: str = "",
        user_id: str = "",
        metadata: Dict[str, str] = {},
        tags: List[str] = [],
    ) -> None:
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string")
        if not isinstance(user_id, str):
            raise TypeError("user_id must be a string")
        if not is_dict_of(metadata, key_allowed_types=(str), value_allowed_types=(str)):
            raise TypeError("metadata must be a dictionary with string keys and values")
        if not is_list_of(tags, str):
            raise TypeError("tags must be a list of strings")

        self._session_id = session_id
        self._user_id = user_id
        self._metadata = metadata
        self._tags = tags

    def attach_context(self) -> None:
        ctx = get_current()
        if self._session_id:
            ctx = set_value(SpanAttributes.SESSION_ID, self._session_id, ctx)
        if self._user_id:
            ctx = set_value(SpanAttributes.USER_ID, self._user_id, ctx)
        if self._metadata:
            ctx = set_value(SpanAttributes.METADATA, self._metadata, ctx)
        if self._tags:
            ctx = set_value(SpanAttributes.TAG_TAGS, self._tags, ctx)

        self._token = attach(ctx)
        return

    def detach_context(self) -> None:
        detach(self._token)
        return


class using_attributes(AbstractContextManager, UsingAttributes):
    """
    TBD
    """

    slots = [
        "_token",
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __enter__(self) -> "using_attributes":
        self.attach_context()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.detach_context()
        return


class async_using_attributes(AbstractAsyncContextManager, UsingAttributes):
    """
    TBD
    """

    slots = [
        "_token",
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def __aenter__(self) -> "async_using_attributes":
        self.attach_context()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.detach_context()
        return


def get_attributes_from_context() -> Iterator[Tuple[str, AttributeValue]]:
    for ctx_attr in CONTEXT_ATTRIBUTES:
        yield ctx_attr, get_value(ctx_attr)


def is_dict_of(
    d: Dict[object, object],
    key_allowed_types: (Type),
    value_allowed_types: (Type) = (),
    value_list_allowed_types: (Type) = (),
) -> bool:
    """
    Method to check types are valid for dictionary.

    Arguments:
    ----------
        d (Dict[object, object]): dictionary itself
        key_allowed_types (T): all allowed types for keys of dictionary
        value_allowed_types (T): all allowed types for values of dictionary
        value_list_allowed_types (T): if value is a list, these are the allowed types for value list

    Returns:
    --------
        True if the data types of dictionary match the types specified by the arguments, false otherwise
    """
    if value_list_allowed_types and not isinstance(value_list_allowed_types, tuple):
        value_list_allowed_types = (value_list_allowed_types,)

    return (
        isinstance(d, dict)
        and all(isinstance(k, key_allowed_types) for k in d.keys())
        and all(
            isinstance(v, value_allowed_types)
            or any(is_list_of(v, t) for t in value_list_allowed_types)
            for v in d.values()
            if value_allowed_types or value_list_allowed_types
        )
    )


def is_list_of(lst: Sequence[object], tp: Type) -> bool:
    return isinstance(lst, list) and all(isinstance(x, tp) for x in lst)
