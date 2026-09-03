import contextvars
import inspect
from functools import partial, wraps
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from opentelemetry.context import (
    Context,
    attach,
    detach,
    get_current,
    get_value,
    set_value,
)
from opentelemetry.util.types import AttributeValue
from typing_extensions import Self

from openinference.semconv.trace import SpanAttributes

from .helpers import safe_json_dumps

DecoratedCallable = TypeVar("DecoratedCallable", bound=Callable[..., Any])

CONTEXT_ATTRIBUTES = (
    SpanAttributes.SESSION_ID,
    SpanAttributes.USER_ID,
    SpanAttributes.METADATA,
    SpanAttributes.TAG_TAGS,
    SpanAttributes.LLM_PROMPT_TEMPLATE,
    SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION,
    SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
)


class _UsingAttributesContextManager:
    """
    Base class for the ``using_*`` helpers.

    Attaches OpenInference context attributes (session id, user id, metadata, tags, prompt
    template) to the current OpenTelemetry context. ``OITracer`` copies whatever is attached
    onto every span it starts, so instrumentors and manual spans pick the attributes up
    without being told about them.

    An instance can be used in three ways:

    * ``with using_session(...)`` / ``async with using_session(...)``: attributes are attached
      on entry and detached on exit.
    * As a decorator on a plain function or ``async def`` coroutine function: attributes are
      attached when the call starts running and detached when it returns or raises.
    * As a decorator on a generator or async generator function: the generator body runs in
      its own copy of the ``contextvars`` context carrying the attributes, so they are visible
      to the body across every ``yield`` but never to the consumer of the generator.

    The decorator protocol is implemented here rather than inherited from
    ``contextlib.ContextDecorator`` because the stdlib version only wraps the synchronous
    call, which for ``async def`` and generator functions ends before the body runs.
    """

    def __init__(
        self,
        *,
        session_id: str = "",
        user_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        prompt_template: str = "",
        prompt_template_version: str = "",
        prompt_template_variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            session_id: Value for ``session.id``.
            user_id: Value for ``user.id``.
            metadata: Value for ``metadata``; JSON-serialized with ``safe_json_dumps``.
            tags: Value for ``tag.tags``.
            prompt_template: Value for ``llm.prompt_template.template``.
            prompt_template_version: Value for ``llm.prompt_template.version``.
            prompt_template_variables: Value for ``llm.prompt_template.variables``;
                JSON-serialized with ``safe_json_dumps``.

        Empty values are skipped, so a helper that sets only some attributes inherits, rather
        than clears, the others from an enclosing block.
        """
        self._session_id = session_id
        self._user_id = user_id
        self._metadata = metadata
        self._tags = tags
        self._prompt_template = prompt_template
        self._prompt_template_version = prompt_template_version
        self._prompt_template_variables = prompt_template_variables

    def _context_with_attributes(self) -> Context:
        """
        Return the current OpenTelemetry context with this instance's non-empty attributes
        added. Nothing is attached; callers decide how to make the returned context current.
        """
        ctx = get_current()
        if self._session_id:
            ctx = set_value(SpanAttributes.SESSION_ID, self._session_id, ctx)
        if self._user_id:
            ctx = set_value(SpanAttributes.USER_ID, self._user_id, ctx)
        if self._metadata:
            ctx = set_value(SpanAttributes.METADATA, safe_json_dumps(self._metadata), ctx)
        if self._tags:
            ctx = set_value(SpanAttributes.TAG_TAGS, self._tags, ctx)
        if self._prompt_template:
            ctx = set_value(SpanAttributes.LLM_PROMPT_TEMPLATE, self._prompt_template, ctx)
        if self._prompt_template_version:
            ctx = set_value(
                SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION, self._prompt_template_version, ctx
            )
        if self._prompt_template_variables:
            ctx = set_value(
                SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
                safe_json_dumps(self._prompt_template_variables),
                ctx,
            )
        return ctx

    def attach_context(self) -> None:
        """
        Attach this instance's attributes to the current OpenTelemetry context and keep the
        token for ``__exit__`` / ``__aexit__``.

        Only the ``with`` / ``async with`` protocol stores the token on ``self``. The decorator
        wrappers keep theirs in a local variable because one decorator instance serves every
        call of the function it wraps, including concurrent and re-entrant calls.
        """
        self._token = attach(self._context_with_attributes())

    def __enter__(self) -> Self:
        self.attach_context()
        return self

    async def __aenter__(self) -> Self:
        self.attach_context()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        detach(self._token)

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        detach(self._token)

    def _copy_of_current_contextvars_with_attributes(self) -> contextvars.Context:
        """
        Return a copy of the current ``contextvars`` context whose OpenTelemetry context
        carries this instance's attributes.

        Generator bodies run inside this copy. Anything the body makes current, including a
        span it holds open across a ``yield``, stays current for the body only. Nothing needs
        detaching afterwards; the copy is simply dropped when the generator is done.
        """
        contextvars_context = contextvars.copy_context()
        contextvars_context.run(attach, self._context_with_attributes())
        return contextvars_context

    def __call__(self, decorated_function: DecoratedCallable) -> DecoratedCallable:
        """
        Use this instance as a decorator; see the class docstring for the three supported
        kinds of function.

        ``functools.wraps`` replaces each wrapper's docstring with the decorated function's,
        so the wrappers below are documented with comments.
        """
        if inspect.isasyncgenfunction(decorated_function):

            @wraps(decorated_function)
            async def async_generator_wrapper(
                *args: Any, **kwargs: Any
            ) -> AsyncGenerator[Any, Any]:
                # Async generators have no ``yield from``, so the forwarding is spelled out:
                # items flow out, ``asend`` values and ``athrow`` exceptions flow in, and
                # ``aclose`` closes the wrapped generator. Every step runs in the copied context.
                await_in_context = partial(
                    _AwaitInContextvarsContext,
                    self._copy_of_current_contextvars_with_attributes(),
                )
                generator: AsyncGenerator[Any, Any] = decorated_function(*args, **kwargs)
                try:
                    item = await await_in_context(generator.asend(None))
                    while True:
                        try:
                            received = yield item
                        except GeneratorExit:
                            await await_in_context(generator.aclose())
                            raise
                        except BaseException as exception:
                            item = await await_in_context(generator.athrow(exception))
                        else:
                            item = await await_in_context(generator.asend(received))
                except StopAsyncIteration:
                    return

            return cast(DecoratedCallable, async_generator_wrapper)

        if inspect.isgeneratorfunction(decorated_function):

            @wraps(decorated_function)
            def generator_wrapper(*args: Any, **kwargs: Any) -> Generator[Any, Any, Any]:
                # ``yield from`` semantics with every step run in the copied context.
                return_value = yield from _drive_in_contextvars_context(
                    self._copy_of_current_contextvars_with_attributes(),
                    decorated_function(*args, **kwargs),
                )
                return return_value

            return cast(DecoratedCallable, generator_wrapper)

        if inspect.iscoroutinefunction(decorated_function):

            @wraps(decorated_function)
            async def coroutine_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Attaching inside the coroutine means it happens when the body starts running,
                # in the awaiting task's context, not when the coroutine object is created.
                token = attach(self._context_with_attributes())
                try:
                    return await decorated_function(*args, **kwargs)
                finally:
                    detach(token)

            return cast(DecoratedCallable, coroutine_wrapper)

        @wraps(decorated_function)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            token = attach(self._context_with_attributes())
            try:
                return decorated_function(*args, **kwargs)
            finally:
                detach(token)

        return cast(DecoratedCallable, sync_wrapper)


def _drive_in_contextvars_context(
    contextvars_context: contextvars.Context,
    steps: Generator[Any, Any, Any],
) -> Generator[Any, Any, Any]:
    """
    Equivalent of ``yield from steps``, except that every step of ``steps`` runs inside
    ``contextvars_context``.

    ``steps`` is a user generator or the ``__await__`` of a coroutine. Whatever it yields
    (items, or the futures a coroutine suspends on) passes through untouched; values sent
    into the returned generator are forwarded with ``send``, exceptions with ``throw``, and
    closing it closes ``steps``. The return value of ``steps`` becomes the return value here.
    """
    run = contextvars_context.run
    try:
        yielded = run(steps.send, None)
        while True:
            try:
                received = yield yielded
            except GeneratorExit:
                run(steps.close)
                raise
            except BaseException as exception:
                yielded = run(steps.throw, exception)
            else:
                yielded = run(steps.send, received)
    except StopIteration as stop:
        return stop.value


class _AwaitInContextvarsContext:
    """
    Awaitable that runs every step of ``awaitable`` inside ``contextvars_context``.

    ``asyncio.Task`` does this for a whole task; this does it for a single awaitable so that
    an async generator's ``asend`` / ``athrow`` / ``aclose`` steps can run in the copied
    context without spawning a task. (``asyncio.create_task(..., context=...)`` needs
    Python 3.11 and would change cancellation semantics.)
    """

    __slots__ = ("_contextvars_context", "_awaitable")

    def __init__(self, contextvars_context: contextvars.Context, awaitable: Awaitable[Any]) -> None:
        self._contextvars_context = contextvars_context
        self._awaitable = awaitable

    def __await__(self) -> Generator[Any, Any, Any]:
        return _drive_in_contextvars_context(self._contextvars_context, self._awaitable.__await__())


class using_session(_UsingAttributesContextManager):
    """
    Context manager to add session id to the current OpenTelemetry Context. OpenInference
    instrumentations will read this Context and pass the session id as a span attribute,
    following the OpenInference semantic conventions.

    Also usable as a decorator on plain, ``async def``, generator and async generator
    functions; the attributes then apply for the whole call.

    Examples:
        with using_session("my-session-id"):
            # Tracing within this block will include the span attribute:
            # "session.id" = "my-session-id"
            ...
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id=session_id)


class using_user(_UsingAttributesContextManager):
    """
    Context manager to add user id to the current OpenTelemetry Context. OpenInference
    instrumentations will read this Context and pass the user id as a span attribute,
    following the OpenInference semantic conventions.

    Also usable as a decorator on plain, ``async def``, generator and async generator
    functions; the attributes then apply for the whole call.

    Examples:
        with using_user("my-user-id"):
            # Tracing within this block will include the span attribute:
            # "user.id" = "my-user-id"
            ...
    """

    def __init__(self, user_id: str) -> None:
        super().__init__(user_id=user_id)


class using_metadata(_UsingAttributesContextManager):
    """
    Context manager to add metadata to the current OpenTelemetry Context. OpenInference
    instrumentations will read this Context and pass the metadata as a span attribute,
    following the OpenInference semantic conventions.

    Also usable as a decorator on plain, ``async def``, generator and async generator
    functions; the attributes then apply for the whole call.

    Examples:
        metadata = {
            "key-1": value_1,
            "key-2": value_2,
            ...
        }
        with using_metadata(metadata):
            # Tracing within this block will include the span attribute:
            # "metadata" = "{\"key-1\": value_1, \"key-2\": value_2, ... }"
            ...
    """

    def __init__(self, metadata: Dict[str, Any]) -> None:
        super().__init__(metadata=metadata)


class using_tags(_UsingAttributesContextManager):
    """
    Context manager to add tags to the current OpenTelemetry Context. OpenInference
    instrumentations will read this Context and pass the tags as a span attribute,
    following the OpenInference semantic conventions.

    Also usable as a decorator on plain, ``async def``, generator and async generator
    functions; the attributes then apply for the whole call.

    Examples:
        tags = [
            "tag_1",
            "tag_2",
            ...
        ]
        with using_tags(tags):
            # Tracing within this block will include the span attribute:
            # "tag.tags" = "["tag_1","tag_2",...]"
            ...
    """

    def __init__(self, tags: List[str]) -> None:
        super().__init__(tags=tags)


class using_prompt_template(_UsingAttributesContextManager):
    """
    Context manager to add prompt template, with its version and variables a to the
    current OpenTelemetry Context. OpenInference instrumentations will read this
    Context and pass the prompt template as a span attribute, following the
    OpenInference semantic conventions.

    Also usable as a decorator on plain, ``async def``, generator and async generator
    functions; the attributes then apply for the whole call.

    Examples:
        prompt_template = "Please describe the weather forecast for {city} on {date}"
        prompt_template_variables = {"city": "Johannesburg", date:"July 11"}
        with using_prompt_template(
            template=prompt_template,
            version="v1.0",
            variables=prompt_template_variables,
        ):
            # Tracing within this block will include the span attribute:
            # "llm.prompt_template.template" = "Please describe the weather
            forecast for {city} on {date}"
            # "llm.prompt_template.version" = "v1.0"
            # "llm.prompt_template.variables" = "{\"city\": \"Johannesburg\",
            \"date\": \"July 11\"}"
            ...
    """

    def __init__(
        self,
        *,
        template: str = "",
        version: str = "",
        variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            prompt_template=template,
            prompt_template_version=version,
            prompt_template_variables=variables,
        )


class using_attributes(_UsingAttributesContextManager):
    """
    Context manager to add attributes to the current OpenTelemetry Context. OpenInference
    instrumentations will read this Context and pass the attributes to the traced span,
    following the OpenInference semantic conventions.

    Also usable as a decorator on plain, ``async def``, generator and async generator
    functions; the attributes then apply for the whole call.

    It is a convenient context manager to use if you find yourself using many others, provided
    by this package, combined.

    Example:
        tags = [
            "tag_1",
            "tag_2",
            ...
        ]
        metadata = {
            "key-1": value_1,
            "key-2": value_2,
            ...
        }
        prompt_template = "Please describe the weather forecast for {city} on {date}"
        prompt_template_variables = {"city": "Johannesburg", date:"July 11"}
        prompt_template_version = "v1.0"
        with using_attributes(
            session_id="my-session-id",
            user_id="my-user-id",
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            # Tracing within this block will include the span attribute:
            # "session.id" = "my-session-id"
            # "user.id" = "my-user-id"
            # "metadata" = "{\"key-1\": value_1, \"key-2\": value_2, ... }"
            # "tag.tags" = "["tag_1","tag_2",...]"
            # "llm.prompt_template.template" = "Please describe the weather forecast
            for {city} on {date}"
            # "llm.prompt_template.variables" = "{\"city\": \"Johannesburg\",
            \"date\": \"July 11\"}"
            # "llm.prompt_template.version " = "v1.0"



            ...

    The previous example is equivalent to doing:
        with (
            using_session("my-session-id"),
            using_user("my-user-id"),
            using_metadata(metadata),
            using_tags(tags),
            using_prompt_template(
                template=prompt_template,
                version=prompt_template_version,
                variables=prompt_template_variables,
            ),
        ):
            ...

    """

    def __init__(
        self,
        *,
        session_id: str = "",
        user_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        prompt_template: str = "",
        prompt_template_version: str = "",
        prompt_template_variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        )


def get_attributes_from_context() -> Iterator[Tuple[str, AttributeValue]]:
    """
    Yield the OpenInference context attributes currently attached to the OpenTelemetry
    context. ``OITracer`` calls this when a span starts to copy them onto the span.
    """
    for ctx_attr in CONTEXT_ATTRIBUTES:
        if (val := get_value(ctx_attr)) is not None:
            yield ctx_attr, cast(AttributeValue, val)
