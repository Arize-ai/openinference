import asyncio
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union, cast

import wrapt  # type: ignore[import-untyped]
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, Context, get_value
from opentelemetry.sdk.trace.id_generator import IdGenerator, RandomIdGenerator
from opentelemetry.trace import (
    INVALID_SPAN,
    Link,
    Span,
    SpanKind,
    Tracer,
    use_span,
)
from opentelemetry.util.types import Attributes
from typing_extensions import ParamSpec, TypeVar, overload

from openinference.semconv.trace import OpenInferenceSpanKindValues

from ._spans import OpenInferenceSpan
from .config import (
    OpenInferenceSpanKind,
    TraceConfig,
    _chain_context,
    _IdGenerator,
    _tool_context,
    get_span_kind_attributes,
)
from .context_attributes import get_attributes_from_context

ParametersType = ParamSpec("ParametersType")
ReturnType = TypeVar("ReturnType")


class OITracer(wrapt.ObjectProxy):  # type: ignore[misc]
    def __init__(self, wrapped: Tracer, config: TraceConfig) -> None:
        super().__init__(wrapped)
        self._self_config = config
        self._self_id_generator = _IdGenerator()

    @property
    def id_generator(self) -> IdGenerator:
        ans = getattr(self.__wrapped__, "id_generator", None)
        if ans and ans.__class__ is RandomIdGenerator:
            return self._self_id_generator
        return cast(IdGenerator, ans)

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Optional["Sequence[Link]"] = (),
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
        *,
        openinference_span_kind: Optional[OpenInferenceSpanKind] = None,
    ) -> Iterator[OpenInferenceSpan]:
        span = self.start_span(
            name=name,
            openinference_span_kind=openinference_span_kind,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )
        with use_span(
            span,
            end_on_exit=end_on_exit,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ) as current_span:
            yield cast(OpenInferenceSpan, current_span)

    def start_span(
        self,
        name: str,
        context: Optional[Context] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Optional["Sequence[Link]"] = (),
        start_time: Optional[int] = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        *,
        openinference_span_kind: Optional[OpenInferenceSpanKind] = None,
    ) -> OpenInferenceSpan:
        span: Span
        if get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            span = INVALID_SPAN
        else:
            tracer = cast(Tracer, self.__wrapped__)
            span = tracer.__class__.start_span(
                self,
                name=name,
                context=context,
                kind=kind,
                attributes=None,
                links=links,
                start_time=start_time,
                record_exception=record_exception,
                set_status_on_exception=set_status_on_exception,
            )
        span = OpenInferenceSpan(span, config=self._self_config)
        if attributes:
            span.set_attributes(dict(attributes))
        if openinference_span_kind is not None:
            span.set_attributes(get_span_kind_attributes(openinference_span_kind))
        span.set_attributes(dict(get_attributes_from_context()))
        return span

    @overload  # for @tracer.agent usage (no parameters)
    def agent(
        self,
        wrapped_function: Callable[ParametersType, ReturnType],
        /,
        *,
        name: None = None,
    ) -> Callable[ParametersType, ReturnType]: ...

    @overload  # for @tracer.agent(name="name") usage (with parameters)
    def agent(
        self,
        wrapped_function: None = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]]: ...

    def agent(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
    ]:
        return self._chain(
            wrapped_function,
            kind=OpenInferenceSpanKindValues.AGENT,  # chains and agents differ only in span kind
            name=name,
        )

    @overload  # for @tracer.chain usage (no parameters)
    def chain(
        self,
        wrapped_function: Callable[ParametersType, ReturnType],
        /,
        *,
        name: None = None,
    ) -> Callable[ParametersType, ReturnType]: ...

    @overload  # for @tracer.chain(name="name") usage (with parameters)
    def chain(
        self,
        wrapped_function: None = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]]: ...

    def chain(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        name: Optional[str] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
    ]:
        return self._chain(wrapped_function, kind=OpenInferenceSpanKindValues.CHAIN, name=name)

    def _chain(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        kind: OpenInferenceSpanKindValues,
        name: Optional[str] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
    ]:
        @wrapt.decorator  # type: ignore[misc]
        def sync_wrapper(
            wrapped: Callable[ParametersType, ReturnType],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _chain_context(
                tracer=tracer,
                name=name,
                kind=kind,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as chain_context:
                output = wrapped(*args, **kwargs)
                return chain_context.process_output(output)

        @wrapt.decorator  #  type: ignore[misc]
        async def async_wrapper(
            wrapped: Callable[ParametersType, Awaitable[ReturnType]],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _chain_context(
                tracer=tracer,
                name=name,
                kind=kind,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as chain_context:
                output = await wrapped(*args, **kwargs)
                return chain_context.process_output(output)

        if wrapped_function is not None:
            if asyncio.iscoroutinefunction(wrapped_function):
                return async_wrapper(wrapped_function)  # type: ignore[no-any-return]
            return sync_wrapper(wrapped_function)  # type: ignore[no-any-return]
        return lambda f: async_wrapper(f) if asyncio.iscoroutinefunction(f) else sync_wrapper(f)

    @overload  # for @tracer.tool usage (no parameters)
    def tool(
        self,
        wrapped_function: Callable[ParametersType, ReturnType],
        /,
        *,
        name: None = None,
        description: Optional[str] = None,
        parameters: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Callable[ParametersType, ReturnType]: ...

    @overload  # for @tracer.tool(name="name") usage (with parameters)
    def tool(
        self,
        wrapped_function: None = None,
        /,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]]: ...

    def tool(
        self,
        wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
        /,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Union[
        Callable[ParametersType, ReturnType],
        Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
    ]:
        @wrapt.decorator  # type: ignore[misc]
        def sync_wrapper(
            wrapped: Callable[ParametersType, ReturnType],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _tool_context(
                tracer=tracer,
                name=name,
                description=description,
                parameters=parameters,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as tool_context:
                output = wrapped(*args, **kwargs)
                return tool_context.process_output(output)

        @wrapt.decorator  #  type: ignore[misc]
        async def async_wrapper(
            wrapped: Callable[ParametersType, Awaitable[ReturnType]],
            instance: Any,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> ReturnType:
            tracer = self
            with _tool_context(
                tracer=tracer,
                name=name,
                description=description,
                parameters=parameters,
                wrapped=wrapped,
                instance=instance,
                args=args,
                kwargs=kwargs,
            ) as tool_context:
                output = await wrapped(*args, **kwargs)
                return tool_context.process_output(output)

        if wrapped_function is not None:
            if asyncio.iscoroutinefunction(wrapped_function):
                return async_wrapper(wrapped_function)  # type: ignore[no-any-return]
            return sync_wrapper(wrapped_function)  # type: ignore[no-any-return]
        return lambda f: async_wrapper(f) if asyncio.iscoroutinefunction(f) else sync_wrapper(f)
