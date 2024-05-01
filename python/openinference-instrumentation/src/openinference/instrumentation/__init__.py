from typing import Dict, List, Any, Optional
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


class suppress_tracing:
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
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
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

    def __enter__(self) -> "UsingAttributes":
        self.attach_context()
        return self

    async def __aenter__(self) -> "UsingAttributes":
        self.attach_context()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.detach_context()
        return

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.detach_context()
        return


class using_session(UsingAttributes):
    """
    TBD
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id=session_id)

    def __enter__(self) -> "using_session":
        super().__enter__()
        return self

    async def __aenter__(self) -> "using_session":
        super().__aenter__()
        return self


class using_user(UsingAttributes):
    """
    TBD
    """

    def __init__(self, user_id: str) -> None:
        super().__init__(user_id=user_id)

    def __enter__(self) -> "using_user":
        super().__enter__()
        return self

    async def __aenter__(self) -> "using_user":
        super().__aenter__()
        return self


class using_metadata(UsingAttributes):
    """
    TBD
    """

    def __init__(self, metadata: Dict[str, str]) -> None:
        super().__init__(metadata=metadata)

    def __enter__(self) -> "using_metadata":
        super().__enter__()
        return self

    async def __aenter__(self) -> "using_metadata":
        super().__aenter__()
        return self


class using_tags(UsingAttributes):
    """
    TBD
    """

    def __init__(self, tags: List[str]) -> None:
        super().__init__(tags=tags)

    def __enter__(self) -> "using_tags":
        super().__enter__()
        return self

    async def __aenter__(self) -> "using_tags":
        super().__aenter__()
        return self


class using_attributes(UsingAttributes):
    """
    TBD
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def __enter__(self) -> "using_attributes":
        super().__enter__()
        return self

    async def __aenter__(self) -> "using_attributes":
        await super().__aenter__()
        return self


def get_attributes_from_context() -> Iterator[Tuple[str, AttributeValue]]:
    for ctx_attr in CONTEXT_ATTRIBUTES:
        yield ctx_attr, get_value(ctx_attr)
