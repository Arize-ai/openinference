from importlib.metadata import version
from json import loads
from typing import Iterator, Tuple, cast

import pytest
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.semconv.trace import MessageAttributes, SpanAttributes, ToolCallAttributes

LLAMA_INDEX_LLMS_OPENAI_VERSION = cast(
    Tuple[int, int], tuple(map(int, version("llama_index.llms.openai").split(".")[:2]))
)
LLAMA_INDEX_LLMS_ANTHROPIC_VERSION = cast(
    Tuple[int, int], tuple(map(int, version("llama_index.llms.anthropic").split(".")[:2]))
)


def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    raise NotImplementedError


TOOL = FunctionTool.from_defaults(get_weather)


class TestToolCallsInChatResponse:
    @pytest.mark.skipif(
        LLAMA_INDEX_LLMS_OPENAI_VERSION < (0, 3),
        reason="ignore older versions to simplify test upkeep",
    )
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    async def test_openai(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        llm = OpenAI(model="gpt-4o-mini", api_key="sk-")
        await self._test(llm, in_memory_span_exporter)

    @pytest.mark.skipif(
        LLAMA_INDEX_LLMS_ANTHROPIC_VERSION < (0, 6),
        reason="ignore older versions to simplify test upkeep",
    )
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    async def test_anthropic(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        llm = Anthropic(model="claude-3-5-haiku-20241022", api_key="sk-")
        await self._test(llm, in_memory_span_exporter)

    @classmethod
    async def _test(
        cls,
        llm: FunctionCallingLLM,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        await llm.achat(
            **llm._prepare_chat_with_tools([TOOL], "what's the weather in San Francisco?"),
        )
        spans = in_memory_span_exporter.get_finished_spans()
        span = spans[-1]
        assert span.attributes
        assert span.attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_ID}")
        assert (
            span.attributes.get(
                f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}"
            )
            == "get_weather"
        )
        assert isinstance(
            arguments := span.attributes.get(
                f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
            ),
            str,
        )
        assert loads(arguments) == {"location": "San Francisco"}


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LlamaIndexInstrumentor().uninstrument()


LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
