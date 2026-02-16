import json
from pathlib import Path
from typing import Any, Dict

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

im = "llm.input_messages"
om = "llm.output_messages"
TEST_DIR = Path(__file__).parent


def starts_with(left_value: Any, right_value: str) -> bool:
    return str(left_value).startswith(right_value)


@pytest.mark.aio
@pytest.mark.asyncio
async def test_async_converse(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    import aioboto3
    from aioresponses import aioresponses

    mock_response = {
        "metrics": {"latencyMs": 864},
        "output": {
            "message": {
                "content": [{"text": "The sum of the numbers from 1 to 10 is 55."}],
                "role": "assistant",
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 18,
            "outputTokens": 86,
            "serverToolUsage": {},
            "totalTokens": 104,
        },
    }

    with aioresponses() as m:
        import re

        m.post(
            re.compile(
                r"https://bedrock-runtime\.us-east-1\.amazonaws\.com/model/anthropic\.claude-3-haiku-20240307-v1(%3A|:)0/converse"
            ),
            payload=mock_response,
            status=200,
            headers={
                "Connection": "keep-alive",
                "Content-Length": "364",
                "Content-Type": "application/json",
                "x-amzn-RequestId": "3c2998d5-04a7-460c-8d98-f9e5769cccfe",
            },
        )

        session = aioboto3.session.Session(
            region_name="us-east-1",
        )
        async with session.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        ) as client:
            response = await client.converse(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": "What is sum of 1 to 10?"}],
                    }
                ],
            )
            im = "llm.input_messages"
            om = "llm.output_messages"
            assert response is not None
            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.status.is_ok
            attributes: Dict[str, Any] = dict(span.attributes or {})
            assert attributes is not None

            assert attributes.pop(f"{im}.0.message.contents.0.message_content.type") == "text"
            assert attributes.pop(f"{im}.0.message.role") == "user"
            assert attributes.pop(f"{om}.0.message.role") == "assistant"
            assert attributes.pop("llm.token_count.completion") == 86
            assert attributes.pop("llm.token_count.prompt") == 18
            assert attributes.pop("llm.token_count.total") == 104
            assert attributes.pop("openinference.span.kind") == "LLM"

            assert attributes.pop("input.value").startswith("What is sum of 1 to 10?")
            assert attributes.pop(f"{im}.0.message.contents.0.message_content.text").startswith(
                "What is sum of 1 to 10?"
            )
            assert attributes.pop("llm.model_name").startswith(
                "anthropic.claude-3-haiku-20240307-v1:0"
            )
            assert attributes.pop(f"{om}.0.message.content").startswith(
                "The sum of the numbers from 1 to 10 is 55."
            )
            assert attributes.pop("output.value").startswith(
                "The sum of the numbers from 1 to 10 is 55."
            )
            assert not attributes


@pytest.mark.aio
@pytest.mark.asyncio
async def test_async_retrieve(
    in_memory_span_exporter: InMemorySpanExporter, read_aio_cassette: Any
) -> None:
    import aioboto3
    from aioresponses import aioresponses

    session = aioboto3.session.Session(
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    with aioresponses() as m:
        cassette_file_path = f"{TEST_DIR}/cassettes/test_async_retrieve.yaml"
        read_aio_cassette(cassette_file_path, m)
        attributes = dict(
            knowledgeBaseId="QKERWOBDH0", retrievalQuery={"text": "What is task Decomposition?"}
        )
        async with session.client("bedrock-agent-runtime") as client:
            response = await client.retrieve(**attributes)
            assert isinstance(response, dict)
            spans = in_memory_span_exporter.get_finished_spans()
            assert len(spans) == 1
            attrs: Dict[str, Any] = dict(spans[0].attributes or {})
            assert attrs.pop("input.mime_type") == "text/plain"
            assert attrs.pop("input.value").startswith("What is task")

            invocation = attrs.pop("llm.invocation_parameters")
            assert '"knowledgeBaseId": "QKERWOBDH0"' in invocation

            assert attrs.pop("openinference.span.kind") == "RETRIEVER"

            for i in range(5):
                prefix = f"retrieval.documents.{i}.document"

                content = attrs.pop(f"{prefix}.content")
                assert isinstance(content, str)
                assert content[:15].strip() != ""  # crude check for non-empty

                metadata = attrs.pop(f"{prefix}.metadata")
                assert '"customDocumentLocation": {"id": "1232345"}' in metadata

                score = attrs.pop(f"{prefix}.score")
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0
            assert not attrs, f"Unexpected extra attributes: {attrs}"


@pytest.mark.aio
@pytest.mark.asyncio
async def test_async_retrieve_and_generate(
    in_memory_span_exporter: InMemorySpanExporter, read_aio_cassette: Any
) -> None:
    import aioboto3
    from aioresponses import aioresponses

    session = aioboto3.session.Session(
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    with aioresponses() as m:
        cassette_file_path = f"{TEST_DIR}/cassettes/test_async_retrieve_and_generate.yaml"
        read_aio_cassette(cassette_file_path, m)
        async with session.client("bedrock-agent-runtime") as client:
            attributes = {
                "input": {"text": "What is Task Decomposition?"},
                "retrieveAndGenerateConfiguration": {
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": "QKERWOBDH0",
                        "modelArn": "anthropic.claude-3-haiku-20240307-v1:0",
                    },
                    "type": "KNOWLEDGE_BASE",
                },
            }
            response = await client.retrieve_and_generate(**attributes)
            assert isinstance(response, dict)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs: Dict[str, Any] = dict(spans[0].attributes or {})
    assert attrs.pop("input.mime_type") == "text/plain"
    assert attrs.pop("input.value").startswith("What is Task")

    invocation = attrs.pop("llm.invocation_parameters")
    assert '"retrieveAndGenerateConfiguration"' in invocation
    assert '"knowledgeBaseId": "QKERWOBDH0"' in invocation
    #
    assert attrs.pop("llm.model_name") == "anthropic.claude-3-haiku-20240307-v1:0"
    assert attrs.pop("openinference.span.kind") == "RETRIEVER"
    assert attrs.pop("output.mime_type") == "text/plain"

    output_val = attrs.pop("output.value")
    assert output_val.startswith("Task Decomposition is a method used in LLM-agent")
    assert "Decomposing the complex task" in output_val

    for i in range(3):
        prefix = f"retrieval.documents.{i}.document"

        content = attrs.pop(f"{prefix}.content")
        assert isinstance(content, str)
        assert "Task Decomposition" in content

        metadata = attrs.pop(f"{prefix}.metadata")
        assert '"customDocumentLocation": {"id": "1232345"}' in metadata
        assert '"x-amz-bedrock-kb-data-source-id": "MVDS8BEVHQ"' in metadata
    # Final assertion: no unexpected attributes remain
    assert not attrs, f"Unexpected attributes found: {attrs}"


@pytest.mark.aio
@pytest.mark.asyncio
async def test_async_invoke_agent_test(
    in_memory_span_exporter: InMemorySpanExporter, read_aio_cassette: Any
) -> None:
    import aioboto3
    from aioresponses import aioresponses

    session = aioboto3.session.Session(
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    with aioresponses() as m:
        cassette_file_path = f"{TEST_DIR}/cassettes/test_async_invoke_agent_test.yaml"
        read_aio_cassette(cassette_file_path, m)
        # if True:
        agent_id = "XNW1LGJJZT"
        agent_alias_id = "K0P4LV9GPO"
        session_id = "default-session123"
        async with session.client("bedrock-agent-runtime") as client:
            attributes = dict(
                inputText="When is a good time to visit the Taj Mahal?",
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                enableTrace=True,
            )
            response = await client.invoke_agent(**attributes)
            events = []
            async for event in response["completion"]:
                events.append(event)
            spans = in_memory_span_exporter.get_finished_spans()
            assert len(events) == 5
            assert len(spans) == 5
            span_names = [span.name for span in spans]
            assert span_names == [
                "LLM",
                "preProcessingTrace",
                "LLM",
                "postProcessingTrace",
                "bedrock_agent.invoke_agent",
            ]
            llm_span = [span for span in spans if span.name == "LLM"][-1]
            llm_span_attributes = dict(llm_span.attributes or {})
            assert llm_span_attributes.pop("input.mime_type") == "text/plain"
            assert starts_with(llm_span_attributes.pop("input.value"), "You are a classifying ag")

            assert starts_with(
                llm_span_attributes.pop("llm.input_messages.0.message.content"),
                "You are a classifying agent",
            )
            assert llm_span_attributes.pop("llm.input_messages.0.message.role") == "assistant"

            invocation_params = json.loads(
                str(llm_span_attributes.pop("llm.invocation_parameters"))
            )
            assert invocation_params["maximumLength"] == 2048
            assert isinstance(invocation_params["stopSequences"], list)
            assert invocation_params["temperature"] == 0.0
            assert invocation_params["topK"] == 250
            assert invocation_params["topP"] == 1.0
            assert (
                llm_span_attributes.pop("llm.model_name")
                == "anthropic.claude-3-sonnet-20240229-v1:0"
            )
            assert llm_span_attributes.pop("llm.provider") == "aws"
            assert starts_with(
                llm_span_attributes.pop("llm.output_messages.0.message.content"),
                ':\n<thinking>\nThe input "When is a good time to visit the Taj Mahal?"',
            )
            assert llm_span_attributes.pop("llm.output_messages.0.message.role") == "assistant"
            assert llm_span_attributes.pop("llm.token_count.completion") == 185
            assert llm_span_attributes.pop("llm.token_count.prompt") == 462
            assert llm_span_attributes.pop("llm.token_count.total") == 647

            assert llm_span_attributes.pop("openinference.span.kind") == "LLM"
            assert llm_span_attributes.pop("output.mime_type") == "text/plain"
            assert starts_with(
                llm_span_attributes.pop("output.value"), ':\n<thinking>\nThe input "When is a'
            )
            assert llm_span_attributes.pop("metadata") is not None
            assert not llm_span_attributes, f"Unexpected attributes: {llm_span_attributes}"
