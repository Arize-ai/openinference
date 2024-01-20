import json
from importlib import import_module
from typing import Any, Callable, Collection, Mapping, Tuple

from openinference.instrumentation.dspy.package import _instruments
from openinference.instrumentation.dspy.version import __version__
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

_DSPY_MODULE = "dspy"
_DSP_MODULE = "dsp"

class DSPyInstrumentor(BaseInstrumentor):
    """
    OpenInference Instrumentor for DSPy
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments



    def _instrument(self, **kwargs):
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)    
        from dsp.modules.lm import LM
        language_model_classes = LM.__subclasses__()
        for lm in language_model_classes:
            wrap_function_wrapper(
                module=_DSP_MODULE,
                name=lm.__name__ + ".basic_request",
                wrapper=_LMBasicRequest(tracer),
            )

    def _uninstrument(self, **kwargs):
        from dsp.modules.lm import LM
        language_model_classes = LM.__subclasses__()
        for lm in language_model_classes:
            lm.request = lm.request.__wrapped__


class _LMBasicRequest():
    """
    Wrapper for DSP LM.basic_request
    Captures all calls to language models (lm)
    """
    def __init__(self, tracer: trace_api.Tracer):
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[type, Any],
        kwargs: Mapping[str, Any],
    ) -> None:
        prompt = args[0]
        kwargs = {**instance.kwargs, **kwargs}
        span_name = instance.__class__.__name__ + ".request"
        span = self._tracer.start_span(span_name)
        span.set_attributes({
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: instance.kwargs.get("model"),
            SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(kwargs),
            SpanAttributes.INPUT_VALUE: prompt,
            SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
        })
        response = wrapped(*args, **kwargs)
        span.set_attributes({
            SpanAttributes.OUTPUT_VALUE: json.dumps(response),
            SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
        })
        span.end()
        return response