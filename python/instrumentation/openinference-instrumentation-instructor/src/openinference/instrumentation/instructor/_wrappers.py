import json
from collections.abc import AsyncIterator as AsyncIteratorABC
from collections.abc import Iterable as IterableABC
from collections.abc import Iterator as IteratorABC
from enum import Enum
from functools import wraps
from inspect import signature
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    get_origin,
)
from urllib.parse import urlparse

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.util.types import AttributeValue

from instructor.utils import is_async
from openinference.instrumentation import infer_llm_provider_from_host, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMSystemValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_V2_CREATE_WRAPPER_MARKER = "__openinference_instructor_v2_create_wrapper__"


class SafeJSONEncoder(json.JSONEncoder):
    """
    Safely encodes non-JSON-serializable objects.
    """

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            if hasattr(o, "dict") and callable(o.dict):  # pydantic v1 models, e.g., from Cohere
                return o.dict()
            return repr(o)


def _flatten(mapping: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    if not mapping:
        return
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, List) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            if isinstance(value, Enum):
                value = value.value
            yield key, value


def _get_input_value(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """
    Parses a method call's inputs into a JSON string. Ensures a consistent
    output regardless of whether the those inputs are passed as positional or
    keyword arguments.
    """

    # For typical class methods, the corresponding instance of inspect.Signature
    # does not include the self parameter. However, the inspect.Signature
    # instance for __call__ does include the self parameter.
    method_signature = signature(method)
    first_parameter_name = next(iter(method_signature.parameters), None)
    signature_contains_self_parameter = first_parameter_name in ["self"]
    bound_arguments = method_signature.bind(
        *(
            [None]  # the value bound to the method's self argument is discarded below, so pass None
            if signature_contains_self_parameter
            else []  # no self parameter, so no need to pass a value
        ),
        *args,
        **kwargs,
    )
    return safe_json_dumps(
        {
            **{
                argument_name: argument_value
                for argument_name, argument_value in bound_arguments.arguments.items()
                if argument_name not in ["self", "kwargs"]
            },
            **bound_arguments.arguments.get("kwargs", {}),
        },
        cls=SafeJSONEncoder,
    )


class _PatchWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    @classmethod
    def _get_messages(cls, request_params: Any) -> Dict[str, Any]:
        llm_messages = {}
        if messages := request_params.get("messages"):
            prefix = f"{LLM_INPUT_MESSAGES}"
            if isinstance(messages, Iterable) and not isinstance(messages, (str, bytes)):
                for idx, message in enumerate(messages):
                    llm_messages[f"{prefix}.{idx}.{MESSAGE_CONTENT}"] = message["content"]
                    llm_messages[f"{prefix}.{idx}.{MESSAGE_ROLE}"] = message["role"]
        return llm_messages

    @classmethod
    def _get_input_value(cls, request_params: Any) -> Any:
        return request_params.get("messages")

    @classmethod
    def _clean_request_params(cls, attributes: Dict[str, Any]) -> Dict[str, Any]:
        attributes = dict(attributes)
        if "response_model" in attributes:
            attributes["response_model"] = (
                attributes["response_model"].__name__ if attributes.get("response_model") else None
            )
        if "hooks" in attributes:
            attributes.pop("hooks")
        return attributes

    @classmethod
    def _get_invocation_params(cls, request_params: Any) -> Dict[str, Any]:
        """
        Clean up invocation parameters for span attributes.
        - Removes messages (can be large / sensitive).
        - Handles tenacity.Retrying objects safely without a hard dependency.
        - Ensures all values are JSON serializable.
        """
        attributes = dict(request_params)

        # Drop messages (too big / sensitive)
        if "messages" in request_params:
            attributes.pop("messages", None)

        if "max_retries" in request_params:
            max_retries = attributes.get("max_retries")

            # Handle tenacity.Retrying safely
            if max_retries is not None and max_retries.__class__.__name__ == "Retrying":
                try:
                    # Capture all key retry configs
                    attributes["max_retries"] = {
                        "stop": repr(getattr(max_retries, "stop", None)),
                        "wait": repr(getattr(max_retries, "wait", None)),
                        "sleep": repr(getattr(max_retries, "sleep", None)),
                        "retry": repr(getattr(max_retries, "retry", None)),
                        "before": repr(getattr(max_retries, "before", None)),
                        "after": repr(getattr(max_retries, "after", None)),
                    }
                except Exception:
                    attributes.pop("max_retries", None)
            else:
                # Ensure JSON-serializability for other types
                try:
                    json.dumps(max_retries)
                except (TypeError, ValueError):
                    attributes["max_retries"] = str(max_retries)

        return attributes

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        new_func = wrapped(*args, **kwargs)

        create = kwargs.get("create")
        client = kwargs.get("client")
        if client is None and create is None and args:
            # Modern call style: instructor.patch(client, mode=...) passes the
            # client positionally.
            client = args[0]

        if not callable(new_func):
            # Modern instructor.patch returns a patched client object, not a
            # callable. Spans for its create() calls are emitted by the
            # v2 create factory wrapper; return the client as-is.
            return new_func

        if getattr(new_func, _V2_CREATE_WRAPPER_MARKER, False):
            # Instructor v2 created this callable through the already-instrumented
            # factory. The legacy wrapper below would create a duplicate TOOL span.
            return new_func

        if create is not None:
            func = create
        elif client is not None:
            func = client.chat.completions.create
        else:
            raise ValueError("Either client or create must be provided")
        func_is_async = is_async(func)

        def patched_new_func(*args: Any, **kwargs: Any) -> Any:
            span_name = "instructor.patch"
            attributes = self._clean_request_params(kwargs)
            with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL,
                            INPUT_VALUE_MIME_TYPE: "application/json",
                            # TODO(harrison): figure out why i cant use args with _get_input_value
                            INPUT_VALUE: self._get_input_value(attributes),
                            LLM_INVOCATION_PARAMETERS: json.dumps(
                                self._get_invocation_params(attributes)
                            ),
                            **self._get_messages(attributes),
                        }
                    )
                ),
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                try:
                    resp = new_func(*args, **kwargs)
                    if resp is not None and hasattr(resp, "dict"):
                        span.set_attribute(OUTPUT_VALUE, json.dumps(resp.dict()))
                        span.set_attribute(OUTPUT_MIME_TYPE, "application/json")

                    if model_name := kwargs.get("model"):
                        span.set_attribute(LLM_MODEL_NAME, model_name)
                    if (endpoint := extract_llm_endpoint_from_sdk_instance(create, client)) and (
                        provider := infer_llm_provider_from_host(endpoint)
                    ):
                        span.set_attribute(LLM_PROVIDER, provider.value)
                    span.set_attribute(LLM_SYSTEM, OpenInferenceLLMSystemValues.OPENAI.value)

                    span.set_status(trace_api.StatusCode.OK)
                    return resp
                except Exception as e:
                    span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        async def async_patched_new_func(*args: Any, **kwargs: Any) -> Any:
            span_name = "instructor.async_patch"
            attributes = self._clean_request_params(kwargs)
            with self._tracer.start_as_current_span(
                span_name,
                attributes=dict(
                    _flatten(
                        {
                            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL,
                            INPUT_VALUE_MIME_TYPE: "application/json",
                            # TODO(harrison): figure out why i cant use args with _get_input_value
                            INPUT_VALUE: self._get_input_value(attributes),
                            LLM_INVOCATION_PARAMETERS: json.dumps(
                                self._get_invocation_params(attributes)
                            ),
                            **self._get_messages(attributes),
                        }
                    )
                ),
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                try:
                    resp = await new_func(*args, **kwargs)
                    if resp is not None and hasattr(resp, "dict"):
                        span.set_attribute(OUTPUT_VALUE, json.dumps(resp.dict()))
                        span.set_attribute(OUTPUT_MIME_TYPE, "application/json")

                    if model_name := kwargs.get("model"):
                        span.set_attribute(LLM_MODEL_NAME, model_name)
                    if (endpoint := extract_llm_endpoint_from_sdk_instance(create, client)) and (
                        provider := infer_llm_provider_from_host(endpoint)
                    ):
                        span.set_attribute(LLM_PROVIDER, provider.value)
                    span.set_attribute(LLM_SYSTEM, OpenInferenceLLMSystemValues.OPENAI.value)

                    span.set_status(trace_api.StatusCode.OK)
                    return resp
                except Exception as e:
                    span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        new_create = async_patched_new_func if func_is_async else patched_new_func

        if client is not None:
            client.chat.completions.create = new_create
            return client
        else:
            return new_create


class _HandleResponseWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        if instance:
            span_name = f"{instance.__class__.__name__}.{wrapped.__name__}"
        else:
            span_name = wrapped.__name__
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL,
                        INPUT_VALUE_MIME_TYPE: "application/json",
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                    }
                )
            ),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
                response_model = response[0]
                if response_model is not None and hasattr(response_model, "model_json_schema"):
                    span.set_attribute(OUTPUT_VALUE, json.dumps(response_model.model_json_schema()))
                    span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
                elif response_model is None and isinstance(response[1], str):
                    span.set_attribute(OUTPUT_VALUE, response[1])
                elif response_model is None:
                    span.set_attribute(OUTPUT_VALUE, json.dumps(response[1]))
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                raise
            span.set_status(trace_api.StatusCode.OK)
            # span.set_attribute(OUTPUT_VALUE, response[1])
        return response


def extract_llm_endpoint_from_sdk_instance(
    create: Any = None,
    client: Any = None,
) -> Optional[str]:
    """Extract the LLM API endpoint from an SDK instance when possible."""
    instance = None
    if create is not None:
        # create is a bound method
        owner = getattr(create, "__self__", None)
        if owner is not None:
            # Completions -> client
            instance = getattr(owner, "_client", None)
    elif client is not None:
        instance = client
    else:
        return None

    endpoint = (
        getattr(instance, "api_base", None)
        or getattr(instance, "base_url", None)
        or getattr(instance, "endpoint", None)
        or getattr(instance, "host", None)
    )

    if not isinstance(endpoint, str) and endpoint is not None:
        endpoint = str(endpoint)

    if isinstance(endpoint, str):
        return urlparse(endpoint).hostname

    return None


INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_VALUE_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT


_NO_ITEM = object()


class _StreamAccumulator:
    """Holds the stream output reported on the span.

    Iterable streams yield separate results, so all of them are kept. Other
    streams (e.g. partial models) yield successive snapshots, so only the
    latest one is kept.
    """

    def __init__(self, keep_all: bool) -> None:
        self._keep_all = keep_all
        self._items: List[Any] = []
        self._last: Any = _NO_ITEM

    def add(self, item: Any) -> None:
        if self._keep_all:
            self._items.append(item)
        else:
            self._last = item

    def output(self) -> Any:
        return self._items if self._keep_all else self._last


class _SyncIteratorProxy(IteratorABC[Any]):
    def __init__(
        self,
        iterator: IteratorABC[Any],
        span: trace_api.Span,
        finish_span: Callable[[trace_api.Span, Any], None],
        keep_all: bool = True,
    ) -> None:
        self._iterator = iterator
        self._span = span
        self._finish_span = finish_span
        self._output = _StreamAccumulator(keep_all)
        self._finished = False

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        if self._finished:
            raise StopIteration
        exhausted = False
        try:
            with trace_api.use_span(self._span, end_on_exit=False):
                try:
                    item = next(self._iterator)
                except StopIteration:
                    exhausted = True
                    item = None
        except BaseException:
            self._end()
            raise
        if exhausted:
            self._finish()
            raise StopIteration
        self._output.add(item)
        return item

    def close(self) -> None:
        if self._finished:
            return
        try:
            with trace_api.use_span(self._span, end_on_exit=False):
                close = getattr(self._iterator, "close", None)
                if callable(close):
                    close()
        except BaseException:
            self._end()
            raise
        self._finish()

    def __del__(self) -> None:
        # The stream was dropped without being exhausted or closed.
        try:
            self._finish()
        except BaseException:
            pass

    def _finish(self) -> None:
        if self._finished:
            return
        try:
            self._finish_span(self._span, self._output.output())
        finally:
            self._end()

    def _end(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._span.end()


class _AsyncIteratorProxy(AsyncIteratorABC[Any]):
    def __init__(
        self,
        iterator: AsyncIteratorABC[Any],
        span: trace_api.Span,
        finish_span: Callable[[trace_api.Span, Any], None],
        keep_all: bool = True,
    ) -> None:
        self._iterator = iterator
        self._span = span
        self._finish_span = finish_span
        self._output = _StreamAccumulator(keep_all)
        self._finished = False

    def __aiter__(self) -> AsyncIteratorABC[Any]:
        return self

    async def __anext__(self) -> Any:
        if self._finished:
            raise StopAsyncIteration
        exhausted = False
        try:
            with trace_api.use_span(self._span, end_on_exit=False):
                try:
                    item = await self._iterator.__anext__()
                except StopAsyncIteration:
                    exhausted = True
                    item = None
        except BaseException:
            self._end()
            raise
        if exhausted:
            self._finish()
            raise StopAsyncIteration
        self._output.add(item)
        return item

    async def aclose(self) -> None:
        if self._finished:
            return
        try:
            with trace_api.use_span(self._span, end_on_exit=False):
                close = getattr(self._iterator, "aclose", None)
                if callable(close):
                    await close()
        except BaseException:
            self._end()
            raise
        self._finish()

    def __del__(self) -> None:
        # The stream was dropped without being exhausted or closed.
        try:
            self._finish()
        except BaseException:
            pass

    def _finish(self) -> None:
        if self._finished:
            return
        try:
            self._finish_span(self._span, self._output.output())
        finally:
            self._end()

    def _end(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._span.end()


class _V2CreateFactoryWrapper:
    """Instrument the public create callable generated by Instructor v2."""

    def __init__(self, tracer: trace_api.Tracer, is_async: bool) -> None:
        self._tracer = tracer
        self._is_async = is_async
        self._enabled = True

    def disable(self) -> None:
        """Stop tracing create callables that were already handed out."""
        self._enabled = False

    @staticmethod
    def _is_iterable_response_model(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
        try:
            response_model = kwargs.get("response_model", args[0] if args else None)
            if get_origin(response_model) in (IterableABC, list):
                return True
            from instructor.dsl.iterable import IterableBase

            return isinstance(response_model, type) and issubclass(response_model, IterableBase)
        except Exception:
            return False

    @staticmethod
    def _get_attributes(
        args: Tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Dict[str, AttributeValue]:
        attributes: Dict[str, AttributeValue] = {
            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value
        }
        try:
            input_value = kwargs.get("messages", kwargs)
            attributes[INPUT_VALUE] = safe_json_dumps(input_value, cls=SafeJSONEncoder)
            attributes[INPUT_VALUE_MIME_TYPE] = "application/json"
        except Exception:
            pass
        try:
            response_model = kwargs.get("response_model", args[0] if args else None)
            if response_model is not None:
                response_model_name = getattr(response_model, "__name__", None)
                attributes["instructor.response_model"] = (
                    response_model_name if response_model_name is not None else repr(response_model)
                )
        except Exception:
            pass
        return attributes

    @staticmethod
    def _normalize_output(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if isinstance(response, Mapping):
            return {
                key: _V2CreateFactoryWrapper._normalize_output(value)
                for key, value in response.items()
            }
        if isinstance(response, (list, tuple)):
            return [_V2CreateFactoryWrapper._normalize_output(value) for value in response]
        return response

    @classmethod
    def _set_output_attributes(cls, span: trace_api.Span, response: Any) -> None:
        try:
            output = cls._normalize_output(response)
            output_value = safe_json_dumps(output, cls=SafeJSONEncoder)
            span.set_attribute(OUTPUT_VALUE, output_value)
            span.set_attribute(OUTPUT_MIME_TYPE, "application/json")
        except Exception:
            pass

    @classmethod
    def _finish_span(cls, span: trace_api.Span, response: Any) -> None:
        if response is not _NO_ITEM:
            cls._set_output_attributes(span, response)
        span.set_status(trace_api.StatusCode.OK)

    @classmethod
    def _wrap_sync_iterator(
        cls, iterator: IteratorABC[Any], span: trace_api.Span, keep_all: bool
    ) -> Iterator[Any]:
        return _SyncIteratorProxy(iterator, span, cls._finish_span, keep_all)

    @classmethod
    def _wrap_async_iterator(
        cls, iterator: AsyncIteratorABC[Any], span: trace_api.Span, keep_all: bool
    ) -> AsyncIteratorABC[Any]:
        return _AsyncIteratorProxy(iterator, span, cls._finish_span, keep_all)

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        create = wrapped(*args, **kwargs)

        @wraps(create)
        def create_sync(*create_args: Any, **create_kwargs: Any) -> Any:
            if not self._enabled or context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return create(*create_args, **create_kwargs)
            attributes = self._get_attributes(create_args, create_kwargs)
            span = self._tracer.start_span("instructor.create", attributes=attributes)
            try:
                with trace_api.use_span(span, end_on_exit=False):
                    response = create(*create_args, **create_kwargs)
            except BaseException:
                span.end()
                raise
            if isinstance(response, IteratorABC):
                keep_all = self._is_iterable_response_model(create_args, create_kwargs)
                return self._wrap_sync_iterator(response, span, keep_all)
            try:
                self._finish_span(span, response)
                return response
            finally:
                span.end()

        @wraps(create)
        async def create_async(*create_args: Any, **create_kwargs: Any) -> Any:
            if not self._enabled or context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return await create(*create_args, **create_kwargs)
            attributes = self._get_attributes(create_args, create_kwargs)
            span = self._tracer.start_span("instructor.async_create", attributes=attributes)
            try:
                with trace_api.use_span(span, end_on_exit=False):
                    response = await create(*create_args, **create_kwargs)
            except BaseException:
                span.end()
                raise
            if isinstance(response, AsyncIteratorABC):
                keep_all = self._is_iterable_response_model(create_args, create_kwargs)
                return self._wrap_async_iterator(response, span, keep_all)
            try:
                self._finish_span(span, response)
                return response
            finally:
                span.end()

        wrapped_create = create_async if self._is_async else create_sync
        setattr(wrapped_create, _V2_CREATE_WRAPPER_MARKER, True)
        return wrapped_create
