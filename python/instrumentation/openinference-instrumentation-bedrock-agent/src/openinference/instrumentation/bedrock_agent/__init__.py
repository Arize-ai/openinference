import datetime
import json
import logging
from functools import wraps
from importlib import import_module
from inspect import signature
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

from botocore.client import BaseClient
from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)
from openinference.instrumentation.bedrock_agent._messages import _MessagesCallback
from openinference.instrumentation.bedrock_agent.package import _instruments
from openinference.instrumentation.bedrock_agent.utils import _EventStream, _use_span
from openinference.instrumentation.bedrock_agent.version import __version__
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import Tracer
from wrapt import wrap_function_wrapper

ClientCreator = TypeVar("ClientCreator", bound=Callable[..., BaseClient])

_MODULE = "botocore.client"
_BASE_MODULE = "botocore"
_MINIMUM_CONVERSE_BOTOCORE_VERSION = "1.34.116"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class InstrumentedClient(BaseClient):  # type: ignore
    """
    Proxy class representing an instrumented boto client.
    """
    invoke_agent: Callable[..., Any]
    _unwrapped_invoke_agent: Callable[..., Any]


def _client_creation_wrapper(
        tracer: Tracer, module_version: str
) -> Callable[[ClientCreator], ClientCreator]:
    def _client_wrapper(
            wrapped: ClientCreator,
            instance: Optional[Any],
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
    ) -> BaseClient:
        """Instruments boto client creation."""
        client = wrapped(*args, **kwargs)
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return client

        call_signature = signature(wrapped)
        bound_arguments = call_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        if bound_arguments.arguments.get("service_name") == 'bedrock-agent-runtime':
            client = cast(InstrumentedClient, client)

            client._unwrapped_invoke_agent = client.invoke_agent
            client.invoke_agent = _model_invocation_wrapper(tracer)(client)
        return client

    return _client_wrapper  # type: ignore


def _model_invocation_wrapper(tracer: Tracer) -> Callable[[InstrumentedClient], Callable[..., Any]]:
    def _invocation_wrapper(wrapped_client: InstrumentedClient) -> Callable[..., Any]:
        """Instruments a bedrock_agent client's `invoke_agent` or `converse` method."""

        @wraps(wrapped_client.invoke_agent)
        def instrumented_response(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
                return wrapped_client._unwrapped_invoke_agent(*args, **kwargs)  # type: ignore

            metadata = {
                "agent_id": kwargs.get('agentId', ''),
                "agent_alias_id": kwargs.get('agentAliasId', ''),
                "service": "bedrock-agent",
                "environment": "production",
                "request_timestamp": datetime.datetime.now().isoformat()
            }
            span = tracer.start_span("bedrock_agent.invoke_agent")

            prompt_variables = {"input_text": kwargs.get('inputText', '')}
            attributes = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                SpanAttributes.LLM_PROVIDER: "aws",
                SpanAttributes.LLM_SYSTEM: "bedrock",
                SpanAttributes.INPUT_VALUE: kwargs.get('inputText', ''),
                SpanAttributes.SESSION_ID: kwargs.get('sessionId', 'default-session'),
                "agent.id": kwargs.get('agentId', ''),
                "agent.alias_id": kwargs.get('agentAliasId', ''),
                "agent.metadata": json.dumps(metadata),
                "metadata": json.dumps(metadata),
                "tracing.enable_trace": kwargs.get('enableTrace', False),
                "prompt_template_variables": prompt_variables
            }
            span.set_attributes(attributes)
            response = wrapped_client._unwrapped_invoke_agent(*args, **kwargs)
            response["completion"] = _EventStream(
                response["completion"],
                _MessagesCallback(span, tracer, kwargs),
                _use_span(span),
            )
            return response

        return instrumented_response

    return _invocation_wrapper


class BedrockAgentInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = (
        "_tracer",
        "_original_client_creator",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        boto = import_module(_MODULE)
        botocore = import_module(_BASE_MODULE)
        self._original_client_creator = boto.ClientCreator.create_client

        wrap_function_wrapper(
            module=_MODULE,
            name="ClientCreator.create_client",
            wrapper=_client_creation_wrapper(
                tracer=self._tracer, module_version=botocore.__version__
            ),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        boto = import_module(_MODULE)
        boto.ClientCreator.create_client = self._original_client_creator
        self._original_client_creator = None
