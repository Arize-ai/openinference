"""Regression test: a mixed-type message content list must not crash the patch wrapper."""

from typing import Any, Callable, Dict

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.instructor._wrappers import _PatchWrapper


def test_patch_call_with_mixed_type_message_content_does_not_raise() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    wrapper = _PatchWrapper(tracer=provider.get_tracer(__name__))

    def fake_patch(**kwargs: Any) -> Callable[..., Any]:
        def new_func(*args: Any, **kw: Any) -> Any:
            return {"ok": True}

        return new_func

    def fake_create(*args: Any, **kwargs: Any) -> Any:
        return {"ok": True}

    patched = wrapper(wrapped=fake_patch, instance=None, args=(), kwargs={"create": fake_create})

    result = patched(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}, "follow-up"]}],
        model="gpt-4o",
    )

    assert result == {"ok": True}
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attributes: Dict[str, Any] = dict(spans[0].attributes or {})
    content_key = "llm.input_messages.0.message.content"
    assert attributes.get(f"{content_key}.0.type") == "text"
    assert attributes.get(f"{content_key}.0.text") == "hi"
    assert attributes.get(f"{content_key}.1") == "follow-up"
