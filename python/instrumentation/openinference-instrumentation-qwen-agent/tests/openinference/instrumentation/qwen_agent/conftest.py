import os
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence

import pytest

# The dashscope SDK reads DASHSCOPE_API_KEY at import time, and qwen_agent
# imports dashscope eagerly. Default it so tests run without credentials.
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-0123456789")

from opentelemetry import trace as trace_api  # noqa: E402
from opentelemetry.sdk import trace as trace_sdk  # noqa: E402
from opentelemetry.sdk.resources import Resource  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from qwen_agent.llm.base import BaseChatModel, register_llm  # noqa: E402
from qwen_agent.llm.schema import ASSISTANT, Message  # noqa: E402

from openinference.instrumentation import TraceConfig  # noqa: E402
from openinference.instrumentation.qwen_agent import QwenAgentInstrumentor  # noqa: E402

# A "turn" is the sequence of cumulative message lists a single
# BaseChatModel.chat call yields; a script is one turn per expected call.
Turn = Sequence[Sequence[Message]]


@register_llm("oi_fake")
class FakeChatModel(BaseChatModel):  # type: ignore[misc]
    """A scripted BaseChatModel that never touches the network.

    Every backend routes through ``BaseChatModel.chat``, overriding only
    ``_chat_with_functions`` / ``_chat_stream`` / ``_chat_no_stream``, so a fake
    at that layer exercises the same code path as DashScope or an
    OpenAI-compatible server.
    """

    #: Set by tests before the agent runs.
    script: List[Turn] = []
    #: Recorded so tests can assert on what the agent asked for.
    calls: List[Dict[str, Any]] = []

    @classmethod
    def configure(cls, script: Sequence[Turn]) -> None:
        cls.script = [turn for turn in script]
        cls.calls = []

    def _next_turn(self, messages: Any, functions: Any) -> Turn:
        type(self).calls.append({"messages": messages, "functions": functions})
        if not type(self).script:
            return [[Message(role=ASSISTANT, content="")]]
        return type(self).script.pop(0)

    def _chat_with_functions(
        self,
        messages: Any,
        functions: Any,
        stream: bool,
        delta_stream: bool,
        generate_cfg: Dict[str, Any],
        lang: str = "en",
    ) -> Any:
        turn = self._next_turn(messages, functions)
        if not stream:
            return list(turn[-1])
        return iter([list(chunk) for chunk in turn])

    def _chat_stream(
        self,
        messages: Any,
        delta_stream: bool,
        generate_cfg: Dict[str, Any],
    ) -> Iterator[List[Message]]:
        turn = self._next_turn(messages, None)
        return iter([list(chunk) for chunk in turn])

    def _chat_no_stream(
        self,
        messages: Any,
        generate_cfg: Dict[str, Any],
    ) -> List[Message]:
        turn = self._next_turn(messages, None)
        return list(turn[-1])


@register_llm("oi_interrupt")
class InterruptingChatModel(BaseChatModel):  # type: ignore[misc]
    """Raises KeyboardInterrupt from the eager part of `chat`."""

    def _chat_with_functions(
        self,
        messages: Any,
        functions: Any,
        stream: bool,
        delta_stream: bool,
        generate_cfg: Dict[str, Any],
        lang: str = "en",
    ) -> Any:
        raise KeyboardInterrupt("interrupted")

    def _chat_stream(
        self, messages: Any, delta_stream: bool, generate_cfg: Dict[str, Any]
    ) -> Iterator[List[Message]]:
        raise KeyboardInterrupt("interrupted")

    def _chat_no_stream(self, messages: Any, generate_cfg: Dict[str, Any]) -> List[Message]:
        raise KeyboardInterrupt("interrupted")


@register_llm("oi_interrupt_stream")
class InterruptingStreamChatModel(BaseChatModel):  # type: ignore[misc]
    """Raises KeyboardInterrupt part-way through streaming."""

    def _chat_with_functions(
        self,
        messages: Any,
        functions: Any,
        stream: bool,
        delta_stream: bool,
        generate_cfg: Dict[str, Any],
        lang: str = "en",
    ) -> Any:
        return self._chat_stream(messages, delta_stream, generate_cfg)

    def _chat_stream(
        self, messages: Any, delta_stream: bool, generate_cfg: Dict[str, Any]
    ) -> Iterator[List[Message]]:
        def gen() -> Iterator[List[Message]]:
            yield [Message(role=ASSISTANT, content="partial")]
            raise KeyboardInterrupt("interrupted")

        return gen()

    def _chat_no_stream(self, messages: Any, generate_cfg: Dict[str, Any]) -> List[Message]:
        raise KeyboardInterrupt("interrupted")


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    QwenAgentInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    in_memory_span_exporter.clear()
    yield
    QwenAgentInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.fixture
def instrument_with_config(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[Any, None, None]:
    """Instrument with a caller-supplied TraceConfig."""

    def _instrument(config: Optional[TraceConfig] = None) -> None:
        # The autouse `instrument` fixture has already run and BaseInstrumentor
        # ignores a second instrument() call, so remove it first.
        QwenAgentInstrumentor().uninstrument()
        QwenAgentInstrumentor().instrument(
            tracer_provider=tracer_provider,
            config=config,
            skip_dep_check=True,
        )
        in_memory_span_exporter.clear()

    yield _instrument
    QwenAgentInstrumentor().uninstrument()
    in_memory_span_exporter.clear()
