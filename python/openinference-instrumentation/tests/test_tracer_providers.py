from secrets import token_hex

import pytest
from opentelemetry.sdk.environment_variables import (
    OTEL_ATTRIBUTE_COUNT_LIMIT,
    OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT,
)
from opentelemetry.sdk.trace import SpanLimits
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation._tracer_providers import (
    _DEFAULT_OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT,
    TracerProvider,
    _create_span_limits_with_large_defaults,
)


class TestSpanLimits:
    """Test span limits behavior and user configuration precedence.

    This test class verifies that span attribute limits are correctly handled
    with the following precedence order:
    1. Explicit span_limits parameter (highest precedence)
    2. OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT environment variable
    3. OTEL_ATTRIBUTE_COUNT_LIMIT environment variable
    4. Default limit (10,000) (lowest precedence)

    Each test creates actual spans and verifies that the correct limit is enforced
    by checking how many attributes are retained vs dropped.
    """

    def test_default_span_attribute_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the default span attribute limit (10,000) is enforced.

        When no environment variables or explicit span_limits are set, the system should
        automatically apply its default span attribute limit of 10,000. Excess attributes
        beyond this limit should be dropped.

        Args:
            monkeypatch: Pytest fixture for modifying environment variables.
        """
        # Ensure no user configuration is present
        monkeypatch.delenv(OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, raising=False)
        monkeypatch.delenv(OTEL_ATTRIBUTE_COUNT_LIMIT, raising=False)

        tracer_provider = TracerProvider()
        in_memory_span_exporter = InMemorySpanExporter()
        tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

        # Create more attributes than the default limit
        default_limit = _DEFAULT_OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT
        attributes = {str(i): i for i in range(default_limit + 1)}
        assert len(attributes) > default_limit
        name = token_hex(8)
        with tracer_provider.get_tracer(__name__).start_span(name) as span:
            span.set_attributes(attributes)

        spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
        assert len(spans[name].attributes or {}) == default_limit

    def test_respects_user_env_var_span_attribute_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that user-configured OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT is respected.

        When a user sets OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, the system should use that value
        instead of its default, demonstrating that user configuration takes precedence.

        Args:
            monkeypatch: Pytest fixture for modifying environment variables.
        """
        user_limit = 500
        monkeypatch.setenv(OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, str(user_limit))
        monkeypatch.delenv(OTEL_ATTRIBUTE_COUNT_LIMIT, raising=False)

        tracer_provider = TracerProvider()
        in_memory_span_exporter = InMemorySpanExporter()
        tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

        # Create more attributes than the user limit
        attributes = {str(i): i for i in range(user_limit + 1)}
        assert len(attributes) > user_limit
        name = token_hex(8)
        with tracer_provider.get_tracer(__name__).start_span(name) as span:
            span.set_attributes(attributes)

        spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
        assert len(spans[name].attributes or {}) == user_limit

    def test_respects_user_env_var_general_attribute_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that user-configured OTEL_ATTRIBUTE_COUNT_LIMIT is respected.

        When a user sets OTEL_ATTRIBUTE_COUNT_LIMIT (general limit), the system should use
        that value for span attributes when no span-specific limit is set.

        Args:
            monkeypatch: Pytest fixture for modifying environment variables.
        """
        user_limit = 200
        monkeypatch.delenv(OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, raising=False)
        monkeypatch.setenv(OTEL_ATTRIBUTE_COUNT_LIMIT, str(user_limit))

        tracer_provider = TracerProvider()
        in_memory_span_exporter = InMemorySpanExporter()
        tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

        # Create more attributes than the user limit
        attributes = {str(i): i for i in range(user_limit + 1)}
        assert len(attributes) > user_limit
        name = token_hex(8)
        with tracer_provider.get_tracer(__name__).start_span(name) as span:
            span.set_attributes(attributes)

        spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
        assert len(spans[name].attributes or {}) == user_limit

    def test_respects_explicit_span_limits_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that explicitly passed span_limits parameter is respected.

        When a user passes an explicit span_limits parameter to register(),
        the system should use that instead of its default or environment variables.

        Args:
            monkeypatch: Pytest fixture for modifying environment variables.
        """

        # Set environment variables that should be ignored
        monkeypatch.setenv(OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, "1000")
        monkeypatch.setenv(OTEL_ATTRIBUTE_COUNT_LIMIT, "500")

        # User explicitly provides span_limits
        user_limit = 100
        explicit_span_limits = SpanLimits(max_span_attributes=user_limit)

        tracer_provider = TracerProvider(span_limits=explicit_span_limits)
        in_memory_span_exporter = InMemorySpanExporter()
        tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

        # Create more attributes than the explicit limit
        attributes = {str(i): i for i in range(user_limit + 1)}
        assert len(attributes) > user_limit
        name = token_hex(8)
        with tracer_provider.get_tracer(__name__).start_span(name) as span:
            span.set_attributes(attributes)

        spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
        # Should use the explicit limit, not env vars or default
        assert len(spans[name].attributes or {}) == user_limit


class TestCreateSpanLimitsWithLargeDefaults:
    """Test the _create_span_limits_with_large_defaults factory function.

    This test class verifies that SpanLimits objects are correctly created with
    appropriate attribute count limits based on environment variable configuration
    and default preferences.
    """

    def test_default_without_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the default limit (10,000) is used when no env vars are set.

        When neither OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT nor OTEL_ATTRIBUTE_COUNT_LIMIT
        environment variables are set, the function should return a SpanLimits object
        configured with the preferred default of 10,000 span attributes.

        Args:
            monkeypatch: Pytest fixture for modifying environment variables.
        """
        monkeypatch.delenv(OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, raising=False)
        monkeypatch.delenv(OTEL_ATTRIBUTE_COUNT_LIMIT, raising=False)

        span_limits = _create_span_limits_with_large_defaults()
        assert span_limits.max_span_attributes == _DEFAULT_OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT

    def test_respects_env_var_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that OpenTelemetry environment variable precedence is respected.

        This test verifies the correct precedence order for attribute count limits:
        1. OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT (span-specific, highest precedence)
        2. OTEL_ATTRIBUTE_COUNT_LIMIT (general, lower precedence)
        3. Default limit (10,000, lowest precedence)

        The function should defer to OpenTelemetry's built-in precedence handling
        when any environment variables are set, and only use the default
        when no environment configuration is present.

        Args:
            monkeypatch: Pytest fixture for modifying environment variables.
        """
        # Test span-specific env var
        monkeypatch.setenv(OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, "5000")
        monkeypatch.delenv(OTEL_ATTRIBUTE_COUNT_LIMIT, raising=False)
        span_limits = _create_span_limits_with_large_defaults()
        assert span_limits.max_span_attributes == 5000

        # Test general env var when span-specific not set
        monkeypatch.delenv(OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, raising=False)
        monkeypatch.setenv(OTEL_ATTRIBUTE_COUNT_LIMIT, "3000")
        span_limits = _create_span_limits_with_large_defaults()
        assert span_limits.max_span_attributes == 3000

        # Test span-specific takes precedence over general
        monkeypatch.setenv(OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT, "8000")
        monkeypatch.setenv(OTEL_ATTRIBUTE_COUNT_LIMIT, "4000")
        span_limits = _create_span_limits_with_large_defaults()
        assert span_limits.max_span_attributes == 8000
