import json
from collections import defaultdict
from typing import Any

import pytest
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.mark.vcr(
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
    decode_compressed_response=True,
)
async def test_instrumenting_model_client(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
):
    openai_model_client = OpenAIChatCompletionClient(
        model="gpt-4o-2024-08-06",
    )
    result = await openai_model_client.create(
        [UserMessage(content="What is the capital of France?", source="user")]
    )
    await openai_model_client.close()
    print(result)

    spans = sorted(in_memory_span_exporter.get_finished_spans(), key=lambda x: x.start_time or 0)
    spans_by_name: dict[str, list[ReadableSpan]] = defaultdict(list)

    for span in spans:
        spans_by_name[span.name].append(span)
    assert "OpenAIChatCompletionClient.create" in spans_by_name
    s = spans_by_name["OpenAIChatCompletionClient.create"][0]
    assert s.status.is_ok

    attrs = dict(s.attributes or {})
    assert attrs.pop("openinference.span.kind", None) == "LLM"
    assert attrs.pop("output.mime_type", None) == "application/json"
    assert (
        attrs.pop("llm.input_messages.0.message.content", None) == "What is the capital of France?"
    )
    assert attrs.pop("llm.input_messages.0.message.role", None) == "user"

    output_value = attrs.pop("output.value", None)
    assert output_value is not None
    output_json = json.loads(output_value)
    assert output_json["finish_reason"] == "stop"
    assert output_json["content"] == "The capital of France is Paris."
    assert output_json["usage"]["prompt_tokens"] == 15
    assert output_json["usage"]["completion_tokens"] == 7
    assert output_json["cached"] is False
    assert output_json["logprobs"] is None
    assert output_json["thought"] is None

    assert not attrs
