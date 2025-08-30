import json

import openai
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation import REDACTED_VALUE, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes


class TestInputValueImageHiding:
    """Test that input.value respects hide_input_images configuration."""

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    def test_input_value_hides_base64_images(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer_provider: TracerProvider,
    ) -> None:
        """Test that input.value redacts base64 images when hide_input_images=True."""

        # Uninstrument and re-instrument with hide_input_images=True
        OpenAIInstrumentor().uninstrument()
        config = TraceConfig(
            hide_inputs=False,  # Keep inputs visible
            hide_input_images=True,  # But hide images
            base64_image_max_length=0,  # Redact all base64 images
        )
        OpenAIInstrumentor().instrument(config=config, tracer_provider=tracer_provider)

        # Small 1x1 pixel base64 image for testing
        base64_image = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
            "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        client = openai.OpenAI(api_key="sk-fake-test-key-for-unit-tests")
        resp = client.chat.completions.create(
            extra_headers={"Accept-Encoding": "gzip"},
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": base64_image}},
                    ],
                }
            ],
        )

        # Verify we got a response
        assert resp.choices[0].message.content is not None

        # Check the span attributes
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1
        span = spans[-1]  # Get the OpenAI span

        attributes = dict(span.attributes or {})
        input_value = attributes.get(SpanAttributes.INPUT_VALUE)

        assert input_value is not None
        assert isinstance(input_value, str)

        # Parse the input value JSON
        input_data = json.loads(input_value)

        # The image URL should be redacted in input.value
        message_content = input_data["messages"][0]["content"]
        image_content = next((c for c in message_content if c.get("type") == "image_url"), None)
        assert image_content is not None
        assert image_content["image_url"]["url"] == REDACTED_VALUE

        # Clean up
        OpenAIInstrumentor().uninstrument()
        in_memory_span_exporter.clear()

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    def test_input_value_preserves_images_when_disabled(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer_provider: TracerProvider,
    ) -> None:
        """Test that input.value preserves images when hide_input_images=False."""

        # Uninstrument and re-instrument with hide_input_images=False
        OpenAIInstrumentor().uninstrument()
        config = TraceConfig(
            hide_inputs=False,
            hide_input_images=False,  # Don't hide images
        )
        OpenAIInstrumentor().instrument(config=config, tracer_provider=tracer_provider)

        # Small 1x1 pixel base64 image for testing
        base64_image = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
            "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        client = openai.OpenAI(api_key="sk-fake-test-key-for-unit-tests")
        resp = client.chat.completions.create(
            extra_headers={"Accept-Encoding": "gzip"},
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": base64_image}},
                    ],
                }
            ],
        )

        # Verify we got a response
        assert resp.choices[0].message.content is not None

        # Check the span attributes
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1
        span = spans[-1]  # Get the OpenAI span

        attributes = dict(span.attributes or {})
        input_value = attributes.get(SpanAttributes.INPUT_VALUE)

        assert input_value is not None
        assert isinstance(input_value, str)

        # Parse the input value JSON
        input_data = json.loads(input_value)

        # The image URL should be preserved in input.value
        message_content = input_data["messages"][0]["content"]
        image_content = next((c for c in message_content if c.get("type") == "image_url"), None)
        assert image_content is not None
        assert "data:image" in image_content["image_url"]["url"]
        assert image_content["image_url"]["url"] != REDACTED_VALUE

        # Clean up
        OpenAIInstrumentor().uninstrument()
        in_memory_span_exporter.clear()

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    def test_input_value_hides_regular_url_images(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer_provider: TracerProvider,
    ) -> None:
        """Test that input.value redacts regular URL images when hide_input_images=True."""

        # Uninstrument and re-instrument with hide_input_images=True
        OpenAIInstrumentor().uninstrument()
        config = TraceConfig(
            hide_inputs=False,
            hide_input_images=True,  # Hide images
        )
        OpenAIInstrumentor().instrument(config=config, tracer_provider=tracer_provider)

        client = openai.OpenAI(api_key="sk-fake-test-key-for-unit-tests")

        # The test URL will cause a 400 error, but we still want to check span data
        try:
            client.chat.completions.create(
                extra_headers={"Accept-Encoding": "gzip"},
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/test-image.jpg"},
                            },
                        ],
                    }
                ],
            )
        except openai.BadRequestError:
            # Expected error for invalid image URL
            pass

        # Check the span attributes
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) >= 1
        span = spans[-1]  # Get the OpenAI span

        attributes = dict(span.attributes or {})
        input_value = attributes.get(SpanAttributes.INPUT_VALUE)

        assert input_value is not None
        assert isinstance(input_value, str)

        # Parse the input value JSON
        input_data = json.loads(input_value)

        # The image URL should be redacted in input.value
        message_content = input_data["messages"][0]["content"]
        image_content = next((c for c in message_content if c.get("type") == "image_url"), None)
        assert image_content is not None
        assert image_content["image_url"]["url"] == REDACTED_VALUE

        # Clean up
        OpenAIInstrumentor().uninstrument()
        in_memory_span_exporter.clear()
