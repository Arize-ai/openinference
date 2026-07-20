# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opentelemetry-api==1.42.1",
#     "opentelemetry-sdk==1.42.1",
#     "opentelemetry-exporter-otlp-proto-http==1.42.1",
#     "openinference-semantic-conventions==0.1.30",
#     "openinference-instrumentation==0.1.54",
#     "pillow>=10.0.0",
# ]
# ///
"""Image path: multimodal chat span, before (inline base64 / redaction) vs after (blob upload).

Emits three identically-shaped LLM chat spans — a user message whose
``message.contents`` carry a text part and an image part
(``message_content.image.image.url``) — through the *released*
``OITracer`` / ``TraceConfig`` masking pipeline. The only thing that varies
is the ``TraceConfig``:

  before / default   ``TraceConfig()`` — the >32 KB base64 data URI is
                     replaced with ``__REDACTED__`` (content destroyed).
  before / raised    ``TraceConfig(base64_image_max_length=100_000_000)`` —
                     the full base64 rides inline (span bloat).
  after              ``BlobOffloadingTraceConfig(blob_uploader=...)`` — the
                     proposed uploader field: the same masking choke point
                     decodes the data URI, uploads the bytes, and records the
                     destination URI. Zero instrumentor changes.

Run:  uv run --script internal_docs/specs/blob_uploads/scripts/image_blob_demo.py
"""

# NOTE: no `from __future__ import annotations` here — TraceConfig.__post_init__
# introspects dataclass field annotations with typing.get_args(), which requires
# evaluated (non-string) annotations on the BlobOffloadingTraceConfig subclass.
import base64
import json
import random
import sys
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Optional, Union

from openinference.instrumentation import OITracer, TraceConfig
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.util.types import AttributeValue
from PIL import Image, ImageDraw

from common import (
    Blob,
    BlobUploader,
    Checker,
    LocalHttpBlobUploader,
    attributes_size_bytes,
    decode_data_uri,
    fetch_url,
    format_bytes,
    maybe_wait_for_browsing,
    print_size_table,
    setup_tracing,
)

PROJECT_NAME = "blob-upload-image-demo"
REDACTED_VALUE = "__REDACTED__"

USER_QUESTION = "What is in this image?"
ASSISTANT_ANSWER = (
    "The image is a synthetic test pattern: a labeled banner over a field of "
    "random RGB noise, generated locally by the OpenInference blob-upload demo."
)

# Attribute keys for the image content part (spec/multimodal_attributes.md):
#   llm.input_messages.0.message.contents.1.message_content.image.image.url
MSG0 = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
TEXT_PART = f"{MSG0}.{MessageAttributes.MESSAGE_CONTENTS}.0"
IMAGE_PART = f"{MSG0}.{MessageAttributes.MESSAGE_CONTENTS}.1"
IMAGE_URL_KEY = f"{IMAGE_PART}.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
OUT_MSG0 = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"


# ---------------------------------------------------------------------------
# Proposed TraceConfig extension (mirrors ../blob_uploads.md §5.1)
#
# TraceConfig.mask() is the choke point every OITracer attribute already flows
# through — it is where oversized base64 images get redacted today. The
# proposal adds a `blob_uploader` field and one branch ahead of the redaction
# branch: eligible data URIs are decoded, uploaded, and replaced by the
# destination URI. Everything else falls through to today's behavior, so a
# `None` from the uploader (backpressure/shutdown) degrades to redaction, and
# hide_* flags (evaluated in super().mask()) still win because the offload
# branch only matches content the config is willing to record.
# ---------------------------------------------------------------------------


def _is_image_data_uri(value: Any) -> bool:
    return (
        isinstance(value, str) and value.startswith("data:image/") and "base64" in value
    )


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
            and not self.hide_input_images
            and not self.hide_inputs
            and not self.hide_input_messages
            and _is_image_data_uri(v)
            and SpanAttributes.LLM_INPUT_MESSAGES in key
            and MessageContentAttributes.MESSAGE_CONTENT_IMAGE in key
            and key.endswith(ImageAttributes.IMAGE_URL)
            and isinstance(v, str)
            and len(v) > (self.base64_image_max_length or 0)
        ):
            try:
                mime, data = decode_data_uri(v)
            except ValueError:
                return super().mask(key, value)
            uri = self.blob_uploader.upload(
                Blob(data=data, mime_type=mime, attribute_key=key)
            )
            if uri is not None:
                return uri
        return super().mask(key, value)


# ---------------------------------------------------------------------------
# Demo image: a labeled banner over seeded RGB noise. Noise is incompressible,
# so the PNG lands in the hundreds-of-KB range a real photo occupies — far
# over the 32,000-char base64 budget TraceConfig allows an image today.
# ---------------------------------------------------------------------------


def make_demo_png() -> bytes:
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


def emit_chat_span(tracer: OITracer, name: str, image_data_uri: str) -> None:
    """One multimodal chat span. Identical for all three variants — only the
    TraceConfig wired into the OITracer differs, which is the point."""
    with tracer.start_as_current_span(name) as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, "gpt-4o-mini")
        span.set_attribute(SpanAttributes.LLM_PROVIDER, "openai")
        span.set_attribute(
            SpanAttributes.INPUT_VALUE,
            json.dumps(
                {"messages": [{"role": "user", "content": USER_QUESTION + " <image>"}]}
            ),
        )
        span.set_attribute(
            SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
        )
        span.set_attribute(f"{MSG0}.{MessageAttributes.MESSAGE_ROLE}", "user")
        span.set_attribute(
            f"{TEXT_PART}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
        )
        span.set_attribute(
            f"{TEXT_PART}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
            USER_QUESTION,
        )
        span.set_attribute(
            f"{IMAGE_PART}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "image"
        )
        span.set_attribute(IMAGE_URL_KEY, image_data_uri)
        span.set_attribute(f"{OUT_MSG0}.{MessageAttributes.MESSAGE_ROLE}", "assistant")
        span.set_attribute(
            f"{OUT_MSG0}.{MessageAttributes.MESSAGE_CONTENT}", ASSISTANT_ANSWER
        )
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, ASSISTANT_ANSWER)


def main() -> int:
    checker = Checker()
    ctx = setup_tracing(PROJECT_NAME)
    uploader = LocalHttpBlobUploader()

    png = make_demo_png()
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    print(
        f"\ngenerated image: {format_bytes(len(png))} PNG "
        f"→ {format_bytes(len(data_uri))} as a base64 data URI"
    )
    assert len(data_uri) > 32_000, "demo image must exceed the default base64 budget"

    raw_tracer = ctx.provider.get_tracer("blob-upload-image-demo")
    variants = {
        "before — default config (image redacted)": OITracer(
            raw_tracer, config=TraceConfig()
        ),
        "before — raised limit (inline base64)": OITracer(
            raw_tracer, config=TraceConfig(base64_image_max_length=100_000_000)
        ),
        "after — blob upload (external URI)": OITracer(
            raw_tracer, config=BlobOffloadingTraceConfig(blob_uploader=uploader)
        ),
    }
    for name, tracer in variants.items():
        emit_chat_span(tracer, name, data_uri)

    checker.check(
        "blob uploads drained", uploader.force_flush(), "background queue empty"
    )
    ctx.provider.force_flush()

    spans = {s.name: s for s in ctx.finished_spans()}
    redacted = spans["before — default config (image redacted)"]
    inline = spans["before — raised limit (inline base64)"]
    offloaded = spans["after — blob upload (external URI)"]

    # --- before, default config: released behavior destroys the image ---
    checker.check(
        "before/default: image attribute is __REDACTED__",
        redacted.attributes.get(IMAGE_URL_KEY) == REDACTED_VALUE,
        "TraceConfig.mask() replaces >32k base64 image URLs",
    )

    # --- before, raised limit: content survives but inflates the span ---
    inline_url = str(inline.attributes.get(IMAGE_URL_KEY, ""))
    checker.check(
        "before/raised: full base64 rides on the span",
        inline_url == data_uri,
        f"{format_bytes(len(inline_url))} attribute value",
    )

    # --- after: URI-only attribute, bytes recoverable, texture intact ---
    blob_url = str(offloaded.attributes.get(IMAGE_URL_KEY, ""))
    checker.check(
        "after: image attribute is an external URI",
        blob_url.startswith("http://"),
        blob_url,
    )
    checker.check(
        "after: image round-trips byte-for-byte",
        blob_url.startswith("http://") and fetch_url(blob_url) == png,
        f"GET {blob_url} == original PNG",
    )
    checker.check(
        "after: sibling text part untouched by offload",
        offloaded.attributes.get(
            f"{TEXT_PART}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"
        )
        == USER_QUESTION,
    )

    print_size_table(
        [
            (
                "LLM span (before/default)",
                attributes_size_bytes(redacted.attributes),
                "image __REDACTED__ — content lost",
            ),
            (
                "LLM span (before/raised)",
                attributes_size_bytes(inline.attributes),
                "inline base64 — span bloat",
            ),
            (
                "LLM span (after/blob)",
                attributes_size_bytes(offloaded.attributes),
                "external URI — content preserved",
            ),
        ]
    )

    print(f"\nPhoenix: {ctx.phoenix_base_url}  → project {PROJECT_NAME!r}")
    print("Open each span's 'input' messages: default shows a redacted image,")
    print("raised-limit and blob-upload spans both render the image — but check")
    print("the attribute sizes above for what that costs on the wire.")

    ctx.shutdown()
    code = checker.exit_code()
    maybe_wait_for_browsing(uploader)
    return code


if __name__ == "__main__":
    sys.exit(main())
