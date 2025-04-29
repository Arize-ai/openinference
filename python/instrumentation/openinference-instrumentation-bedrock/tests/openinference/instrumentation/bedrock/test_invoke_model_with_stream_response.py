import json

import boto3
import pytest
from anthropic.types import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from anthropic.types.message_create_params import MessageCreateParamsBase
from botocore.eventstream import EventStream
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typing_extensions import assert_never

from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


class TestInvokeModelWithStreamResponse:
    @pytest.mark.vcr(
        before_record_request=lambda _: _.headers.clear() or _,
    )
    def test_invoke(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        client = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="123",
            aws_secret_access_key="321",
        )
        system_message = "You're super friendly"
        messages: list[MessageParam] = [
            MessageParam(
                role="user",
                content=[
                    TextBlockParam(
                        type="text",
                        text="Hello! Here's a red pixel.",
                    ),
                    ImageBlockParam(
                        type="image",
                        source=Base64ImageSourceParam(
                            type="base64",
                            media_type="image/png",
                            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
                        ),
                    ),
                ],
            ),
            MessageParam(
                role="assistant",
                content="Thank you!",
            ),
            MessageParam(
                role="user",
                content=[
                    TextBlockParam(
                        type="text",
                        text="what's current temperature and population in Los Angeles?",
                    )
                ],
            ),
            MessageParam(
                role="assistant",
                content=[
                    TextBlockParam(
                        type="text",
                        text="Certainly! I'd be happy to help you with "
                        "information about the current temperature and population for both Los "
                        "Angeles. Let's get that information for you using the available tools.",
                    ),
                    ToolUseBlockParam(
                        type="tool_use",
                        id="toolu_bdrk_01LzqVRttWAxkjdq19wmG2Cn",
                        name="get_current_weather_info",
                        input={"city": "Los Angeles"},
                    ),
                    ToolUseBlockParam(
                        type="tool_use",
                        id="toolu_bdrk_01EDzXjPQZSVnEzttRY98wyT",
                        name="get_current_population_info",
                        input={"city": "Los Angeles"},
                    ),
                ],
            ),
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id="toolu_bdrk_01LzqVRttWAxkjdq19wmG2Cn",
                        content="nice",
                    ),
                    ToolResultBlockParam(
                        type="tool_result",
                        tool_use_id="toolu_bdrk_01EDzXjPQZSVnEzttRY98wyT",
                        content=[TextBlockParam(type="text", text="large")],
                    ),
                ],
            ),
            MessageParam(
                role="user",
                content="what's current temperature and population in San Francisco?",
            ),
        ]
        tools: list[ToolParam] = [
            ToolParam(
                name="get_current_weather_info",
                description="Get the current weather for a city",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for",
                        }
                    },
                    "required": ["city"],
                },
            ),
            ToolParam(
                name="get_current_population_info",
                description="Get the current population for a city",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the population for",
                        }
                    },
                    "required": ["city"],
                },
            ),
        ]
        invocation_parameters = {
            "max_tokens": 200,
        }
        message_create_params = MessageCreateParamsBase(
            max_tokens=invocation_parameters["max_tokens"],
            model=model_id,
            system=system_message,
            messages=messages,
            tools=tools,
        )
        params = dict(message_create_params)
        params.pop("model")
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            **params,
        }
        response = client.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(request_body),
        )
        assert isinstance(response["body"], EventStream)
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"].decode())
            if "delta" in chunk and "text" in chunk["delta"]:
                print(chunk["delta"]["text"], end="")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes or dict())
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
        assert (
            attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.{0}.{MESSAGE_CONTENT_TEXT}")
            == system_message
        )
        for i, message in enumerate(messages, 1):
            if (
                isinstance(message["content"], list)
                and message["content"]
                and any(block["type"] == "tool_result" for block in message["content"])
            ):
                assert attributes.pop(f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}") == "tool"
            else:
                assert attributes.pop(f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}") == message["role"]
            if isinstance(message["content"], str):
                assert (
                    attributes.pop(
                        f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENTS}.{0}.{MESSAGE_CONTENT_TEXT}"
                    )
                    == message["content"]
                )
            elif isinstance(message["content"], list):
                k = 0
                for j, content in enumerate(message["content"], 0):
                    assert not isinstance(content, (TextBlock, ToolUseBlock))
                    if content["type"] == "text":
                        assert (
                            attributes.pop(
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TEXT}"
                            )
                            == content["text"]
                        )
                    elif content["type"] == "tool_use":
                        assert (
                            attributes.pop(
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALLS}.{k}.{TOOL_CALL_ID}"
                            )
                            == content["id"]
                        )
                        assert (
                            attributes.pop(
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALLS}.{k}.{TOOL_CALL_FUNCTION_NAME}"
                            )
                            == content["name"]
                        )
                        assert attributes.pop(
                            f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALLS}.{k}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
                        ) == json.dumps(content["input"])
                        k += 1
                    elif content["type"] == "image":
                        assert (
                            attributes.pop(
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}"
                            )
                            == f"data:{content['source']['media_type']};"
                            f"{content['source']['type']},{content['source']['data']}"
                        )
                    elif content["type"] == "tool_result":
                        if "content" in content:
                            assert attributes.pop(
                                f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TEXT}"
                            ) == (
                                content["content"]
                                if isinstance(content["content"], str)
                                else "\n\n".join(
                                    b["text"] for b in content["content"] if b["type"] == "text"
                                )
                            )
                    elif content["type"] == "document":
                        pass
                    else:
                        assert_never(content)
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"
        )
        for i, tool in enumerate(tools):
            assert isinstance(
                tool_json_schema := attributes.pop(f"{LLM_TOOLS}.{i}.{TOOL_JSON_SCHEMA}"), str
            )
            assert json.loads(tool_json_schema) == tools[i]
            assert attributes.pop(
                f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.{i}.{TOOL_CALL_ID}"
            )
            assert (
                attributes.pop(
                    f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.{i}.{TOOL_CALL_FUNCTION_NAME}"
                )
                == tool["name"]
            )
            assert (
                attributes.pop(
                    f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.{i}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
                )
                == '{"city": "San Francisco"}'
            )
        assert attributes.pop(OUTPUT_MIME_TYPE)
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert json.loads(output_value)
        assert attributes.pop(INPUT_MIME_TYPE)
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == request_body
        assert isinstance(parameters := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(parameters) == invocation_parameters
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
        assert attributes.pop(LLM_MODEL_NAME) == model_id
        assert not attributes


IMAGE_URL = ImageAttributes.IMAGE_URL
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
JSON = OpenInferenceMimeTypeValues.JSON
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA
USER_ID = SpanAttributes.USER_ID
