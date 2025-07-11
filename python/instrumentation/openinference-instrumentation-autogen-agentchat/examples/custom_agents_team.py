import asyncio
from typing import AsyncGenerator, Callable, List, Sequence

from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor

# Set up the tracer provider with both OTLP and Console exporters
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Instrument the AutogenAgentChat
AutogenAgentChatInstrumentor().instrument(tracer_provider=tracer_provider)


class CountDownAgent(BaseChatAgent):
    """A custom agent that counts down from a given number."""

    def __init__(self, name: str, count: int = 3):
        super().__init__(name, "A simple agent that counts down.")
        self._count = count

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        response = None
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                response = message
        return response

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseChatMessage | Response, None]:
        inner_messages = []
        for i in range(self._count, 0, -1):
            msg = TextMessage(content=f"{i}...", source=self.name)
            inner_messages.append(msg)
            yield msg

        yield Response(
            chat_message=TextMessage(
                content="Countdown complete! Ready to proceed.", source=self.name
            ),
            inner_messages=inner_messages,
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


class ArithmeticAgent(BaseChatAgent):
    """A custom agent that performs arithmetic operations on numbers."""

    def __init__(self, name: str, description: str, operator_func: Callable[[int], int]):
        super().__init__(name, description=description)
        self._operator_func = operator_func
        self._message_history: List[BaseChatMessage] = []

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        self._message_history.extend(messages)

        # Try to extract a number from the last message
        try:
            last_content = self._message_history[-1].content
            # Extract number from text (simple parsing)
            import re

            if numbers := re.findall(r"\d+", last_content):
                number = int(numbers[0])
                result = self._operator_func(number)
                response_message = TextMessage(
                    content=f"Applied operation to {number}, result: {result}", source=self.name
                )
            else:
                response_message = TextMessage(
                    content="No number found in the message. Please provide a number.",
                    source=self.name,
                )
        except (ValueError, IndexError):
            response_message = TextMessage(
                content="Could not process the input. Please provide a valid number.",
                source=self.name,
            )

        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._message_history.clear()


class AnalyzerAgent(BaseChatAgent):
    """A custom agent that analyzes text and provides insights."""

    def __init__(self, name: str):
        super().__init__(name, "An agent that analyzes text and provides insights.")

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        if not messages:
            return Response(
                chat_message=TextMessage(content="No messages to analyze.", source=self.name)
            )

        last_message = messages[-1]
        content = last_message.content

        # Simple text analysis
        word_count = len(content.split())
        char_count = len(content)
        sentence_count = len([s for s in content.split(".") if s.strip()])

        analysis = f"""Text Analysis Results:
- Word count: {word_count}
- Character count: {char_count}
- Sentence count: {sentence_count}
- Average words per sentence: {word_count / max(sentence_count, 1):.1f}
- Contains numbers: {"Yes" if any(c.isdigit() for c in content) else "No"}"""

        response_message = TextMessage(content=analysis, source=self.name)
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4",
    )

    # Create standard AssistantAgent
    coordinator = AssistantAgent(
        "coordinator",
        model_client=model_client,
        system_message="""You are a coordinator agent. Your job is to:
1. Understand the user's request
2. Coordinate with other agents to fulfill the request
3. Provide a final summary
When you're ready to finish, say 'TASK COMPLETE'.""",
    )

    # Create custom agents
    countdown_agent = CountDownAgent("countdown", count=5)

    doubler_agent = ArithmeticAgent("doubler", "An agent that doubles numbers", lambda x: x * 2)

    analyzer_agent = AnalyzerAgent("analyzer")

    # Create termination condition
    text_termination = TextMentionTermination("TASK COMPLETE")

    # Create team with mix of standard and custom agents
    team = RoundRobinGroupChat(
        [coordinator, countdown_agent, doubler_agent, analyzer_agent],
        termination_condition=text_termination,
    )

    print("Starting team chat with custom agents...")
    print("=" * 50)

    task = """
    Please help me with this multi-step task:
    1. Start with a countdown from 5
    2. Double the number 42
    3. Analyze this text: 'Custom agents in AutoGen provide powerful extensibility for building \
specialized AI workflows.'
    4. Provide a summary of all operations performed
    """

    result = await team.run(task=task)
    await model_client.close()

    print("=" * 50)
    print("Team chat completed!")
    print(f"Final result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
