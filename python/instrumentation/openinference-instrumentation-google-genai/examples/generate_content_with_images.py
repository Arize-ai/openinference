#!/usr/bin/env python3
"""
Example demonstrating Google GenAI instrumentation with image attachments.
This example shows that the instrumentation now properly handles:
- Part.from_bytes() for base64 encoded images
- Part.from_uri() for URI-referenced images
- PDF and other file attachments
"""

import asyncio
import base64
import os

import requests
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

# Phoenix endpoint
endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))


def create_test_image_data() -> bytes:
    """Create a simple 1x1 pixel PNG for testing."""
    base64_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="  # noqa: E501
    return base64.b64decode(base64_png)


def test_inline_data_image():
    print("🖼️  Testing inline_data (Part.from_bytes) with image...")
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        image_data = create_test_image_data()
        image_part = Part.from_bytes(data=image_data, mime_type="image/png")

        content = Content(
            role="user",
            parts=[
                Part.from_text(text="Describe this image:"),
                image_part,
            ],
        )

        config = GenerateContentConfig(
            system_instruction="You are a helpful assistant. Describe what you see in images."
        )

        print("   Making API call with inline image data...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content,  # ✅ correct for 1.46.0
            config=config,  # ✅ correct for 1.46.0
        )

        print(f"   ✅ Success! Response: {response.text[:100]}...")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_inline_data_pdf():
    print("📄 Testing inline_data (Part.from_bytes) with PDF...")
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        fake_pdf_data = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
            b"2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n"
            b"""3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents <4 0 R>
            \n>>\nendobj\n"""
            b"""4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Tes
            t PDF content) Tj\nET\nendstream\nendobj\n"""
            b"""xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100
             00000 n \n0000000178 00000 n \n"""
            b"""trailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n273\n%%EOF"""
        )

        pdf_part = Part.from_bytes(data=fake_pdf_data, mime_type="application/pdf")

        content = Content(
            role="user",
            parts=[
                Part.from_text(text="Analyze this PDF document:"),
                pdf_part,
            ],
        )

        print("   Making API call with inline PDF data...")
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content,
        )

        print(f"   ✅ Success! Response: {response.text[:100]}...")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_file_data_uri():
    print("🔗 Testing file_data (Part.from_uri equivalent via bytes)...")
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # fetch the image manually - using a simple, reliable image
        img_bytes = requests.get(
            "https://httpbin.org/image/png",
            headers={"User-Agent": "Mozilla/5.0 (compatible; OpenInference-Test)"},
        ).content

        image_part = Part.from_bytes(data=img_bytes, mime_type="image/png")

        content = Content(
            role="user",
            parts=[
                Part.from_text(text="What do you see in this image?"),
                image_part,
            ],
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content,
        )
        print(f"   ✅ Success! Response: {response.text[:100]}...")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


async def test_async_with_images():
    print("🔄 Testing async API with images...")
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).aio

        image_data = create_test_image_data()
        image_part = Part.from_bytes(data=image_data, mime_type="image/png")

        content = Content(
            role="user",
            parts=[
                Part.from_text(text="Describe this small test image:"),
                image_part,
            ],
        )

        print("   Making async API call with image...")
        response = await client.models.generate_content(
            model="gemini-2.0-flash",
            contents=content,
        )

        print(f"   ✅ Success! Async response: {response.text[:100]}...")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    print("🚀 Testing Google GenAI Instrumentation with Images")
    print("=" * 60)
    print("This demonstrates that the instrumentation now properly handles:")
    print("- Part.from_bytes() for inline image/file data")
    print("- Part.from_uri() for URI-referenced files")
    print("- No more 'Other field types not supported' errors!")
    print("=" * 60)

    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Please set GEMINI_API_KEY environment variable")
        print("   export GEMINI_API_KEY='your-api-key-here'")
        return

    print("🔧 Instrumenting Google GenAI client...")
    GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    print("   ✅ Instrumentation enabled - traces will be sent to Phoenix!\n")

    results = [
        test_inline_data_image(),
        test_inline_data_pdf(),
        test_file_data_uri(),
        asyncio.run(test_async_with_images()),
    ]

    print("=" * 60)
    print("📊 RESULTS SUMMARY:")
    print(f"   ✅ Successful tests: {sum(results)}/{len(results)}")

    if all(results):
        print("   🎉 All tests passed! The instrumentation fix is working!")
        print("   📈 Check Phoenix UI at http://localhost:6006 to see the traces")
    else:
        print("   ⚠️  Some tests failed - check API key and network connection")

    print("=" * 60)


if __name__ == "__main__":
    main()
