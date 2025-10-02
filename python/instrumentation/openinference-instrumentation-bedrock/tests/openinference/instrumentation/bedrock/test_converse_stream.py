from typing import Any, Dict

import boto3
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "top_song",
                "description": "Get the most popular song played on a radio station.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "sign": {
                                "type": "string",
                                "description": "The call sign for the radio station",
                            }
                        },
                        "required": ["sign"],
                    }
                },
            }
        }
    ]
}

im = "llm.input_messages"
om = "llm.output_messages"


class TestConverseStream:
    @pytest.mark.vcr(
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: _["headers"].clear() or _,
    )
    def test_invoke_text_message(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        session = boto3.session.Session()
        client = session.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="123",
            aws_secret_access_key="321",
        )
        response = client.converse_stream(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "What is sum of 1 to 10?"}],
                }
            ],
        )
        response = list(response["stream"])
        assert len(response) == 174
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes: Dict[str, Any] = dict(span.attributes or {})
        assert attributes is not None

        assert attributes.pop(f"{im}.0.message.contents.0.message_content.type") == "text"
        assert attributes.pop("input.mime_type") == "application/json"
        assert attributes.pop(f"{im}.0.message.role") == "user"
        assert attributes.pop(f"{om}.0.message.contents.0.message_content.type") == "text"
        assert attributes.pop(f"{om}.0.message.role") == "assistant"
        assert attributes.pop("llm.token_count.completion") == 181
        assert attributes.pop("llm.token_count.prompt") == 18
        assert attributes.pop("llm.token_count.total") == 199
        assert attributes.pop("openinference.span.kind") == "LLM"
        assert attributes.pop("output.mime_type") == "application/json"

        assert attributes.pop("input.value").startswith(
            '[{"role": "user", "content": [{"text": "What is sum of 1 to 10?"}]}]'
        )
        assert attributes.pop(f"{im}.0.message.contents.0.message_content.text").startswith(
            "What is sum of 1 to 10?"
        )
        assert attributes.pop("llm.invocation_parameters").startswith('{"stop_reason": "end_turn"}')
        assert attributes.pop("llm.model_name").startswith("anthropic.claude-3-haiku-20240307-v1:0")
        assert attributes.pop(f"{om}.0.message.contents.0.message_content.text").startswith(
            "The sum of the numbers from 1 to 10 is 55."
        )
        assert attributes.pop("output.value").startswith(
            '{"role": "assistant", "content": [{"text": "The sum of the numbers from 1 to 10 is 55.'
        )
        assert not attributes

    @pytest.mark.vcr(
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: _["headers"].clear() or _,
    )
    def test_invoke_tool_message(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        session = boto3.session.Session()
        client = session.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="123",
            aws_secret_access_key="321",
        )
        response = client.converse_stream(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "What is the most popular song on WZPZ?"}],
                }
            ],
            toolConfig=tool_config,
        )
        response = list(response["stream"])
        assert len(response) == 9
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes: Dict[str, Any] = dict(span.attributes or {})
        assert attributes is not None
        assert attributes.pop("input.mime_type") == "application/json"
        assert attributes.pop(f"{im}.0.message.contents.0.message_content.type") == "text"
        assert attributes.pop(f"{im}.0.message.role") == "user"
        assert attributes.pop(f"{om}.0.message.role") == "assistant"
        assert attributes.pop(f"{om}.0.message.tool_calls.0.tool_call.function.name") == "top_song"
        assert attributes.pop("llm.token_count.completion") == 38
        assert attributes.pop("llm.token_count.prompt") == 353
        assert attributes.pop("llm.token_count.total") == 391
        assert attributes.pop("openinference.span.kind") == "LLM"
        assert attributes.pop("output.mime_type") == "application/json"
        assert attributes.pop("input.value").startswith(
            '[{"role": "user", "content": [{"text": "What is the most popular song on WZPZ?"}]}]'
        )
        assert attributes.pop(
            "llm.input_messages.0.message.contents.0.message_content.text"
        ).startswith("What is the most popular song on WZPZ?")
        assert attributes.pop("llm.invocation_parameters").startswith('{"stop_reason": "tool_use"}')
        assert attributes.pop("llm.model_name").startswith("anthropic.claude-3-haiku-20240307-v1:0")
        assert attributes.pop(
            f"{om}.0.message.tool_calls.0.tool_call.function.arguments"
        ).startswith('{"sign": "WZPZ"}')
        assert attributes.pop(f"{om}.0.message.tool_calls.0.tool_call.id").startswith("tooluse_")
        assert attributes.pop("llm.tools.0.tool.json_schema").startswith(
            '{"name": "top_song", "description": "Get the most popular song played on a'
        )
        assert attributes.pop("output.value").startswith(
            '{"role": "assistant", "content": [{"toolUse": {"toolUseId": "tooluse_'
        )
        assert attributes == {}

    @pytest.mark.vcr(
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: _["headers"].clear() or _,
    )
    def test_invoke_tool_response_message(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        session = boto3.session.Session()
        client = session.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="123",
            aws_secret_access_key="321",
        )
        response = client.converse_stream(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "What is the most popular song on WZPZ?"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "Okay, let's find the most popular song played on radio "
                            "station WZPZ"
                        },
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_ZQEZysOVRqitr-89GxHizA",
                                "name": "top_song",
                                "input": {"sign": "WZPZ"},
                            }
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": "tooluse_ZQEZysOVRqitr-89GxHizA",
                                "content": [{"text": "Rock and Roll Hall"}],
                            }
                        }
                    ],
                },
            ],
            toolConfig=tool_config,
        )
        response = list(response["stream"])
        assert len(response) > 0
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes: Dict[str, Any] = dict(span.attributes or {})
        assert attributes is not None
        assert attributes.pop("input.mime_type") == "application/json"
        assert (
            attributes.pop("llm.input_messages.0.message.contents.0.message_content.type") == "text"
        )
        assert attributes.pop("llm.input_messages.0.message.role") == "user"
        assert (
            attributes.pop("llm.input_messages.1.message.contents.0.message_content.type") == "text"
        )
        assert attributes.pop("llm.input_messages.1.message.role") == "assistant"
        assert (
            attributes.pop("llm.input_messages.1.message.tool_calls.0.tool_call.function.name")
            == "top_song"
        )
        assert attributes.pop("llm.input_messages.2.message.content") == "Rock and Roll Hall"
        assert attributes.pop("llm.input_messages.2.message.role") == "user"
        assert (
            attributes.pop("llm.input_messages.2.message.tool_call_id")
            == "tooluse_ZQEZysOVRqitr-89GxHizA"
        )
        assert (
            attributes.pop("llm.output_messages.0.message.contents.0.message_content.type")
            == "text"
        )
        assert attributes.pop("llm.output_messages.0.message.role") == "assistant"
        assert attributes.pop("llm.token_count.completion") == 23
        assert attributes.pop("llm.token_count.prompt") == 443
        assert attributes.pop("llm.token_count.total") == 466
        assert attributes.pop("openinference.span.kind") == "LLM"
        assert attributes.pop("output.mime_type") == "application/json"
        assert "What is the most popular song on WZPZ?" in attributes.pop("input.value")
        assert attributes.pop(
            "llm.input_messages.0.message.contents.0.message_content.text"
        ).startswith("What is the most popular song on WZPZ?")
        assert attributes.pop(
            "llm.input_messages.1.message.contents.0.message_content.text"
        ).startswith("Okay, let's find the most popular song played on radio station WZPZ")
        assert attributes.pop(
            "llm.input_messages.1.message.tool_calls.0.tool_call.function.arguments"
        ).startswith('{"sign": "WZPZ"}')
        assert attributes.pop("llm.input_messages.1.message.tool_calls.0.tool_call.id").startswith(
            "tooluse_"
        )
        assert '{"stop_reason": "end_turn"}' in attributes.pop("llm.invocation_parameters")
        assert "anthropic.claude-3-haiku-20240307" in attributes.pop("llm.model_name")
        assert "song played on radio station WZPZ" in attributes.pop(
            "llm.output_messages.0.message.contents.0.message_content.text"
        )
        assert "Get the most popular song" in attributes.pop("llm.tools.0.tool.json_schema")
        assert "radio station WZPZ" in attributes.pop("output.value")
        assert attributes == {}
