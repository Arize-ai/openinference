import contextvars
import inspect
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
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

    Attaches OpenInference context attributes to the current OpenTelemetry context for the
    duration of a ``with`` / ``async with`` block, or of a decorated function call. It
    implements the decorator protocol itself (rather than via ``contextlib.ContextDecorator``)
    so that coroutine functions and generator functions are supported too.
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
        self._session_id = session_id
        self._user_id = user_id
        self._metadata = metadata
        self._tags = tags
        self._prompt_template = prompt_template
        self._prompt_template_version = prompt_template_version
        self._prompt_template_variables = prompt_template_variables

    def _context_with_attributes(self) -> Context:
        """Return the current OpenTelemetry context with this instance's attributes added."""
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
        Return a copy of the current ``contextvars`` context in which the OpenTelemetry
        context carries this instance's attributes.

        A generator body is run inside this copy, so the attributes are visible to the body
        (and stay visible across its ``yield`` suspension points, together with anything the
        body itself makes current, e.g. a span) while never leaking into the consumer's
        context. Nothing needs to be detached afterwards: the copy is simply dropped.
        """
        contextvars_context = contextvars.copy_context()
        contextvars_context.run(attach, self._context_with_attributes())
        return contextvars_context

    def __call__(self, decorated_function: DecoratedCallable) -> DecoratedCallable:
        """
        Use this context manager as a decorator.

        The context attributes stay attached for the whole execution of the decorated
        function, whether it is a plain function, a coroutine function (``async def``), a
        generator function or an async generator function. A plain ``contextlib.ContextDecorator``
        only wraps the synchronous call, which for a coroutine or generator function finishes as
        soon as the coroutine or generator object is created, i.e. before its body runs.

        A decorator instance is shared by every invocation of the function it wraps, so the
        wrappers never store the OpenTelemetry context token on ``self``: concurrent or
        re-entrant calls each keep their own token (or their own ``contextvars`` copy).
        """
        if inspect.isasyncgenfunction(decorated_function):

            @wraps(decorated_function)
            async def async_generator_wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
                contextvars_context = self._copy_of_current_contextvars_with_attributes()
                generator: AsyncGenerator[Any, Any] = decorated_function(*args, **kwargs)

                async def step(operation: Awaitable[Any]) -> Any:
                    return await _RunInContextvarsContext(contextvars_context, operation)

                # Equivalent to ``yield from`` for async generators: values sent to and
                # exceptions thrown into the wrapper are forwarded to the wrapped generator.
                try:
                    item = await step(generator.asend(None))
                    while True:
                        try:
                            received = yield item
                        except GeneratorExit:
                            await step(generator.aclose())
                            raise
                        except BaseException as exception:
                            item = await step(generator.athrow(exception))
                        else:
                            item = await step(generator.asend(received))
                except StopAsyncIteration:
                    return

            return cast(DecoratedCallable, async_generator_wrapper)

        if inspect.isgeneratorfunction(decorated_function):

            @wraps(decorated_function)
            def generator_wrapper(*args: Any, **kwargs: Any) -> Generator[Any, Any, Any]:
                contextvars_context = self._copy_of_current_contextvars_with_attributes()
                generator: Generator[Any, Any, Any] = decorated_function(*args, **kwargs)
                # Equivalent to ``yield from``, with every step run inside the copied context.
                try:
                    item = contextvars_context.run(generator.send, None)
                    while True:
                        try:
                            received = yield item
                        except GeneratorExit:
                            contextvars_context.run(generator.close)
                            raise
                        except BaseException as exception:
                            item = contextvars_context.run(generator.throw, exception)
                        else:
                            item = contextvars_context.run(generator.send, received)
                except StopIteration as stop:
                    return stop.value

            return cast(DecoratedCallable, generator_wrapper)

        if inspect.iscoroutinefunction(decorated_function):

            @wraps(decorated_function)
            async def coroutine_wrapper(*args: Any, **kwargs: Any) -> Any:
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


class _RunInContextvarsContext:
    """
    Awaitable that drives ``awaitable`` with every step executed inside ``contextvars_context``.

    This is what ``asyncio.Task`` does for its coroutine, applied to a single awaitable: the
    futures the awaitable suspends on are passed through to the event loop unchanged, and the
    values sent (or exceptions thrown) into this awaitable are forwarded into ``awaitable``.
    """

    def __init__(self, contextvars_context: contextvars.Context, awaitable: Awaitable[Any]) -> None:
        self._contextvars_context = contextvars_context
        self._awaitable = awaitable

    def __await__(self) -> Generator[Any, Any, Any]:
        run = self._contextvars_context.run
        steps = self._awaitable.__await__()
        try:
            suspended_on = run(steps.send, None)
            while True:
                try:
                    received = yield suspended_on
                except GeneratorExit:
                    run(steps.close)
                    raise
                except BaseException as exception:
                    suspended_on = run(steps.throw, exception)
                else:
                    suspended_on = run(steps.send, received)
        except StopIteration as stop:
            return stop.value


class using_session(_UsingAttributesContextManager):
    """
    Context manager to add session id to the current OpenTelemetry Context. OpenInference
    instrumentations will read this Context and pass the session id as a span attribute,
    following the OpenInference semantic conventions.

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

    Examples:
        prompt_template = "Please describe the weather forecast for {city} on {date}"
        prompt_template_variables = {"city": "Johannesburg", date:"July 11"}
        with using_prompt_template(
            template=prompt_template,
            version=prompt_template_variables,
            variables="v1.0",
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
    for ctx_attr in CONTEXT_ATTRIBUTES:
        if (val := get_value(ctx_attr)) is not None:
            yield ctx_attr, cast(AttributeValue, val)
