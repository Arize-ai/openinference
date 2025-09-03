#!/usr/bin/env python3

import os

from dotenv import load_dotenv
from mistralai import Mistral
from phoenix.otel import register

from openinference.instrumentation.mistralai import MistralAIInstrumentor

load_dotenv()

tracer = register(
    project_name="mistral-ocr",
    endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
)

# Initialize instrumentation
MistralAIInstrumentor().instrument(tracer_provider=tracer)

# Initialize client
client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))


def test_ocr_with_working_image():
    """Test OCR with a working image URL that should display in Phoenix"""

    # Using a reliable image URL - this is a simple diagram/chart
    image_url = "https://upload.wikimedia.org/wikipedia/commons/d/d1/Ai_lizard.png"
    try:
        print("🔍 Testing OCR with working image URL...")
        print(f"Image URL: {image_url}")

        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": image_url,
            },
            include_image_base64=True,
        )

        print("✅ OCR completed successfully!")

        if hasattr(ocr_response, "pages") and ocr_response.pages:
            print(f"📄 Pages processed: {len(ocr_response.pages)}")

            for i, page in enumerate(ocr_response.pages):
                print(f"\n--- Page {i + 1} ---")
                if hasattr(page, "markdown") and page.markdown:
                    print("📝 Markdown content:")
                    print(
                        page.markdown[:200] + "..." if len(page.markdown) > 200 else page.markdown
                    )

                if hasattr(page, "images") and page.images:
                    print(f"🖼️  Extracted images: {len(page.images)}")
                    for j, img in enumerate(page.images):
                        if hasattr(img, "id"):
                            print(f"  - Image {j + 1}: {img.id}")

        print("\n🔗 View traces in your Phoenix project")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure:")
        print("1. MISTRAL_API_KEY environment variable is set")
        print("2. You have sufficient credits")
        print("3. The image URL is accessible")


if __name__ == "__main__":
    test_ocr_with_working_image()
