from importlib.metadata import version
from typing import Any, Generator, Tuple, cast
from unittest.mock import patch

import guardrails
import pytest
from guardrails import Guard
from guardrails.validator_base import (  # type: ignore[import-untyped]
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from openinference.instrumentation import OITracer
from openinference.instrumentation.guardrails import GuardrailsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydash.strings import words as _words

GUARDRAILS_VERSION = cast(
    Tuple[int, int, int],
    tuple(map(int, version("guardrails-ai").split(".")[:3])),
)

if GUARDRAILS_VERSION < (0, 5, 0):
    from guardrails.utils.llm_response import LLMResponse  # type: ignore
else:
    from guardrails.classes.llm.llm_response import LLMResponse  # type: ignore


@register_validator(name="two-words", data_type="string")
class TwoWords(Validator):  # type: ignore[misc]
    def _get_fix_value(self, value: str) -> str:
        words = value.split()
        if len(words) == 1:
            words = _words(value)
        if len(words) == 1:
            value = f"{value} {value}"
            words = value.split()
        return " ".join(words[:2])

    def validate(self, value: Any, *args: Any, **kwargs: Any) -> ValidationResult:
        if len(value.split()) != 2:
            return FailResult(
                error_message="must be exactly two words",
                fix_value=self._get_fix_value(str(value)),
            )
        return PassResult()


@pytest.fixture()
def setup_guardrails_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    GuardrailsInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    GuardrailsInstrumentor().uninstrument()


# Ensure we're using the common OITracer from common opeinference-instrumentation pkg
def test_oitracer(
    setup_guardrails_instrumentation: Any,
) -> None:
    assert isinstance(GuardrailsInstrumentor()._tracer, OITracer)


@patch(
    "guardrails.llm_providers.ArbitraryCallable._invoke_llm",
    return_value=LLMResponse(output="More Than Two"),
)
def test_guardrails_instrumentation(
    mock_invoke_llm: Any,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_guardrails_instrumentation: Any,
) -> None:
    # we expect the guard to raise an exception here because the mock LLMResponse has more
    # than two words
    guard = Guard().use(TwoWords, on_fail="exception")
    with pytest.raises(Exception):
        guard(
            llm_api=lambda prompt: "yoo whatever",
            prompt="oh look im a bad person",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) >= 3  # Expecting at least 3 spans from the guardrails module

    expected_span_names = {
        "ArbitraryCallable.__call__",
        "AsyncValidatorService.after_run_validator",
        "Runner.step",
    }
    found_span_names = set(span.name for span in spans)
    assert expected_span_names.issubset(found_span_names), "Missing expected spans"

    for span in spans:
        attributes = dict(span.attributes or dict())
        if span.name == "ArbitraryCallable.__call__":
            assert attributes.get("openinference.span.kind") == "LLM"
            assert span.status.is_ok

        elif span.name == "AsyncValidatorService.after_run_validator":
            assert attributes.get("validator_name") == "two-words"
            assert attributes.get("validator_on_fail") == "EXCEPTION"
            # 3 letter response
            assert attributes["output.value"] == "fail"
            # this may be counter intuitive but the exception from the validator actually
            # occurs in the span for guard_parse, not post_validation
            assert span.status.is_ok, "post_validation span status should be OK"

        elif span.name == "Runner.step":
            assert attributes["openinference.span.kind"] == "GUARDRAIL"
            assert "input.value" in attributes
            assert not span.status.is_ok, "guard_parse span status should not be OK"


def test_guardrails_uninstrumentation(tracer_provider: TracerProvider) -> None:
    # Store references to the original functions
    original_prompt_callable_base_call = guardrails.llm_providers.PromptCallableBase.__call__
    original_runner_step = guardrails.run.Runner.step
    original_validator_service_base_after_run_validator = (
        guardrails.validator_service.ValidatorServiceBase.after_run_validator
    )

    # Instrument the Guardrails to wrap methods
    GuardrailsInstrumentor().instrument(tracer_provider=tracer_provider)

    # Ensure methods are wrapped
    assert hasattr(
        guardrails.llm_providers.PromptCallableBase.__call__, "__wrapped__"
    ), "Expected PromptCallableBase.__call__ to be wrapped"
    assert hasattr(guardrails.run.Runner.step, "__wrapped__"), "Expected Runner.step to be wrapped"
    assert hasattr(
        guardrails.validator_service.ValidatorServiceBase.after_run_validator, "__wrapped__"
    ), "Expected ValidatorServiceBase.after_run_validator to be wrapped"

    # Uninstrument the Guardrails to unwrap methods
    GuardrailsInstrumentor().uninstrument()

    # Ensure methods are unwrapped and point to the original functions
    assert (
        guardrails.llm_providers.PromptCallableBase.__call__ is original_prompt_callable_base_call
    ), "Expected PromptCallableBase.__call__ to be unwrapped"
    assert (
        guardrails.run.Runner.step is original_runner_step
    ), "Expected Runner.step to be unwrapped"
    assert (
        guardrails.validator_service.ValidatorServiceBase.after_run_validator
        is original_validator_service_base_after_run_validator
    ), "Expected ValidatorServiceBase.after_run_validator to be unwrapped"
