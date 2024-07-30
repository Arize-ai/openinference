from importlib.metadata import version

import pytest
from httpx import Response
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respx import MockRouter

LANGCHAIN_VERSION = tuple(map(int, version("langchain-core").split(".")[:3]))


def test_image_in_message(
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: MockRouter,
    image_base64: str,
) -> None:
    question = "What's in this image?"
    answer = "Nothing."
    image_url = f"data:image/jpeg;base64,{image_base64}"
    image = dict(type="image_url", image_url=dict(url=image_url))
    content = [dict(type="text", text=question), image]
    url = "https://api.openai.com/v1/chat/completions"
    choice = dict(index=0, message=dict(role="assistant", content=answer), finish_reason="stop")
    respx_mock.post(url).mock(return_value=Response(status_code=200, json=dict(choices=[choice])))
    ChatOpenAI(model="gpt-4o").invoke([HumanMessage(content=content)])  # type: ignore[arg-type]
    assert (spans := in_memory_span_exporter.get_finished_spans())
    span = spans[0]
    assert (attributes := dict(span.attributes or {}))
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND, None)
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE, None)
    assert attributes.pop(SpanAttributes.INPUT_VALUE, None)
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE, None)
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE, None)
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME, None)
    assert attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS, None)
    if LANGCHAIN_VERSION >= (0, 2):
        assert attributes.pop(SpanAttributes.METADATA, None)
    assert attributes == {
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.contents.0.message_content.type": "text",
        "llm.input_messages.0.message.contents.0.message_content.text": question,
        "llm.input_messages.0.message.contents.1.message_content.type": "image",
        "llm.input_messages.0.message.contents.1.message_content.image.image.url": image_url,
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": answer,
    }


@pytest.fixture
def image_base64() -> str:
    return "R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw=="
