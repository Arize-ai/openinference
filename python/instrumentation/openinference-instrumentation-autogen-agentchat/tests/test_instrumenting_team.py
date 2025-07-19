import json
from typing import Mapping, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from openinference.semconv.trace import SpanAttributes


class TestTeam:
    @pytest.mark.asyncio
    @pytest.mark.vcr(
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
        decode_compressed_response=True,
    )
    async def test_team_run(
        self,
        tracer_provider: trace_api.TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import TextMentionTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        model_client = OpenAIChatCompletionClient(
            model="gpt-4o-2024-08-06",
        )

        # Create two agents: a primary and a critic
        primary_agent = AssistantAgent(
            "primary",
            model_client=model_client,
            system_message="You are a helpful AI assistant.",
        )

        critic_agent = AssistantAgent(
            "critic",
            model_client=model_client,
            system_message="""
            Provide constructive feedback.
            Respond with 'APPROVE' when your feedbacks are addressed.
            """,
        )

        # Termination condition: stop when the critic says "APPROVE"
        text_termination = TextMentionTermination("APPROVE")

        # Create a team with both agents
        team = RoundRobinGroupChat(
            [primary_agent, critic_agent], termination_condition=text_termination
        )

        # Run the team on a task
        _ = await team.run(task="Write a short poem about the fall season.")
        await model_client.close()

        # Verify that spans were created
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 9
        final_span = spans[-1]
        assert "RoundRobinGroupChat" in final_span.name
        assert final_span.status.is_ok

        attributes = dict(cast(Mapping[str, AttributeValue], final_span.attributes))
        # Verify input value
        input_value = json.loads(attributes["input.value"])
        assert isinstance(input_value, dict)
        assert "task" in input_value
        assert input_value["task"] == "Write a short poem about the fall season."

        # Verify participant information
        assert attributes["participant_names"] == ("primary", "critic")
        assert len(attributes["participant_descriptions"]) == 2
        assert attributes["openinference.span.kind"] == "CHAIN"

        # Check span attributes:
        primary_span = next(s for s in spans if s.name == "primary.on_messages_stream")
        primary_attrs = dict(cast(Mapping[str, AttributeValue], primary_span.attributes))
        assert primary_attrs[SpanAttributes.GRAPH_NODE_ID] == "primary"
        assert primary_attrs[SpanAttributes.GRAPH_NODE_PARENT_ID] == "start"

        critic_span = next(s for s in spans if s.name == "critic.on_messages_stream")
        critic_attrs = dict(cast(Mapping[str, AttributeValue], critic_span.attributes))
        assert critic_attrs[SpanAttributes.GRAPH_NODE_ID] == "critic"
        assert critic_attrs[SpanAttributes.GRAPH_NODE_PARENT_ID] == "primary"
