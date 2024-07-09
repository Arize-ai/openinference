from unittest.mock import patch

import guardrails
import pytest
from guardrails import Guard
from guardrails.utils.llm_response import LLMResponse
from guardrails.validators import TwoWords
from openinference.instrumentation.guardrails import GuardrailsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_guardrails_instrumentation(tracer_provider: TracerProvider) -> None:
    GuardrailsInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    GuardrailsInstrumentor().uninstrument()


@patch(
    "guardrails.llm_providers.ArbitraryCallable._invoke_llm",
    return_value=LLMResponse(output="More Than Two"),
)
def test_guardrails_instrumentation(
    mock_invoke_llm,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_guardrails_instrumentation,
):
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
        if span.name == "ArbitraryCallable.__call__":
            assert span.attributes["openinference.span.kind"] == "LLM"
            assert span.status.is_ok

        elif span.name == "AsyncValidatorService.after_run_validator":
            assert span.attributes["validator_name"] == "two-words"
            assert span.attributes["validator_on_fail"].name == "EXCEPTION"

            # note that validator result should fail because the mock response returns a
            # 3 letter response
            assert span.attributes["output.value"] == "fail"
            # this may be counter intuitive but the exception from the validator actually
            # occurs in the span for guard_parse, not post_validation
            assert span.status.is_ok, "post_validation span status should be OK"

        elif span.name == "Runner.step":
            assert span.attributes["openinference.span.kind"] == "GUARDRAIL"
            assert "input.value" in span.attributes
            assert not span.status.is_ok, "guard_parse span status should not be OK"


def test_guardrails_uninstrumentation(tracer_provider: TracerProvider):
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
