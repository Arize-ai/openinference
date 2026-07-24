# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.60.0",
#     "openai-agents>=0.18.0,<0.19",
#     "openinference-instrumentation-openai-agents>=1.6.1",
#     "openinference-instrumentation-openai>=0.1.52",
#     "openinference-instrumentation>=0.1.54",
#     "openinference-semantic-conventions>=0.1.30",
#     "opentelemetry-sdk>=1.42.0",
#     "opentelemetry-exporter-otlp-proto-http>=1.42.0",
#     "pillow>=10.0.0",
# ]
# ///
"""Image path: a real instrumented OpenAI Agents SDK app, redaction vs blob upload.

Runs the same vision agent (``Agent`` + ``Runner.run`` with an ``input_image``)
twice. The agents SDK calls the OpenAI Responses API under the hood, so the app
is instrumented at both layers: openinference-instrumentation-openai-agents for
the agent spans and openinference-instrumentation-openai for the LLM span that
carries ``message_content.image.image.url``. The app code never changes — only
the ``TraceConfig`` handed to the OpenAI instrumentor does:

  run 1 — default config     the >32 KB base64 image is replaced with
                             ``__REDACTED__`` (today's released behavior).
  run 2 — blob-upload config the same attribute key records the blob store URI
                             (a repo-relative file path from the demo store).

Prerequisites: OPENAI_API_KEY; a local ``phoenix serve`` (http://localhost:6006).
Run:  uv run --script internal_docs/specs/blob_uploads/scripts/image_blob_demo.py
"""

# NOTE: no `from __future__ import annotations` here — TraceConfig.__post_init__
# introspects dataclass field annotations with typing.get_args(), which requires
# evaluated (non-string) annotations on the BlobOffloadingTraceConfig subclass.
import asyncio
import base64
import os
import random
import re
import sys
from dataclasses import dataclass, field
from io import BytesIO
from typing import Callable, Optional, Union

from agents import Agent, Runner
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from openinference.semconv.trace import (
    ImageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from PIL import Image, ImageDraw

from common import Blob, BlobUploader, LocalBlobStore

PROJECT_NAME = "blob-upload-image-demo"
_DATA_URI = re.compile(r"^data:(?P<mime>[^;,]+);base64,(?P<payload>.*)$", re.DOTALL)


# ---------------------------------------------------------------------------
# Proposed TraceConfig extension (techspec §2.2): one branch ahead of today's
# image-redaction rule in the mask() choke point — eligible data URIs are
# decoded, uploaded, and replaced with the destination URI. Everything else
# falls through to released behavior.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlobOffloadingTraceConfig(TraceConfig):
    blob_uploader: Optional[BlobUploader] = field(
        default=None,
        metadata={
            "env_var": "OPENINFERENCE_DEMO_BLOB_UPLOADER_UNSET",
            "default_value": None,
        },
    )

    def mask(
        self,
        key: str,
        value: Union[AttributeValue, Callable[[], AttributeValue]],
    ) -> Optional[AttributeValue]:
        v = value() if callable(value) else value
        if (
            self.blob_uploader is not None
            and not (
                self.hide_inputs or self.hide_input_messages or self.hide_input_images
            )
            and isinstance(v, str)
            and SpanAttributes.LLM_INPUT_MESSAGES in key
            and MessageContentAttributes.MESSAGE_CONTENT_IMAGE in key
            and key.endswith(ImageAttributes.IMAGE_URL)
            and len(v) > (self.base64_image_max_length or 0)
            and (match := _DATA_URI.match(v)) is not None
        ):
            data = base64.b64decode(match.group("payload"))
            uri = self.blob_uploader.upload(
                Blob(data=data, mime_type=match.group("mime"), attribute_key=key)
            )
            if uri is not None:
                return uri
        return super().mask(key, value)


def make_demo_png() -> bytes:
    """A labeled banner over seeded RGB noise — noise is incompressible, so the
    PNG lands in the hundreds-of-KB range a real photo occupies, far over the
    32,000-char base64 budget TraceConfig allows an image today."""
    rng = random.Random(42)
    width, height = 640, 400
    img = Image.frombytes("RGB", (width, height), rng.randbytes(width * height * 3))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 56], fill=(16, 24, 48))
    draw.text((16, 12), "OpenInference blob-upload demo", fill=(240, 240, 255))
    draw.text((16, 32), "synthetic test pattern (seeded noise)", fill=(160, 170, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def run_vision_agent(data_uri: str) -> str:
    """The real app: an Agents SDK agent asked about an image."""
    agent = Agent(
        name="Vision Assistant",
        instructions="You describe images concisely.",
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    )
    result = await Runner.run(
        agent,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Describe this image in one sentence.",
                    },
                    {"type": "input_image", "image_url": data_uri, "detail": "low"},
                ],
            }
        ],
    )
    return str(result.final_output)


IMAGE_URL_SUFFIX = (
    f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
)


def print_llm_spans(memory: InMemorySpanExporter, label: str, since: int) -> int:
    """Print the spans that carry the image attribute; list the rest by name."""
    spans = memory.get_finished_spans()[since:]
    for span in spans:
        attributes = span.attributes or {}
        image_keys = [k for k in attributes if k.endswith(IMAGE_URL_SUFFIX)]
        if not image_keys:
            continue
        total = sum(len(k) + len(str(v)) for k, v in attributes.items())
        print(f"\n── {span.name} — {label}  ({len(attributes)} attrs, {total:,} B) ──")
        for key in sorted(attributes):
            text = str(attributes[key]).replace("\n", "\\n")
            if len(text) > 76:
                text = f"{text[:76]}… ({len(text):,} chars)"
            print(f"  {key} = {text}")
    other = ", ".join(
        s.name
        for s in spans
        if not any(k.endswith(IMAGE_URL_SUFFIX) for k in (s.attributes or {}))
    )
    print(f"  (other spans in this trace: {other})")
    return len(memory.get_finished_spans())


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set — this demo makes real vision calls.")

    png = make_demo_png()
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    print(
        f"generated image: {len(png):,} B PNG → {len(data_uri):,} chars as a data URI"
    )

    phoenix = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    ).rstrip("/")
    provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": PROJECT_NAME})
    )
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(f"{phoenix}/v1/traces"))
    )
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)

    store = LocalBlobStore()
    seen = 0
    for label, config in [
        ("default config (image __REDACTED__)", TraceConfig()),
        ("blob upload (external URI)", BlobOffloadingTraceConfig(blob_uploader=store)),
    ]:
        OpenAIInstrumentor().instrument(tracer_provider=provider, config=config)
        answer = await run_vision_agent(data_uri)
        OpenAIInstrumentor().uninstrument()
        print(f"\n[{label}] model: {answer}")
        provider.force_flush()
        seen = print_llm_spans(memory, label, seen)

    OpenAIAgentsInstrumentor().uninstrument()
    provider.shutdown()

    print(f"\nPhoenix: {phoenix}  → project {PROJECT_NAME!r}")
    print("Compare the two runs' LLM spans: message_content.image.image.url is")
    print("__REDACTED__ in the first and a repo-relative blob-store path in the")
    print("second — displaying/resolving that URI is the backend's concern.")


if __name__ == "__main__":
    asyncio.run(main())
