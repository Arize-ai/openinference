from contextlib import suppress
from importlib.metadata import version
from random import randint
from typing import Iterator

import pytest
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider
from packaging.version import Version

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

llms = [
    OpenAI(max_retries=0),
    Anthropic(max_retries=0),
]

if Version(version("llama-index-llms-openai")) >= Version("0.3.30"):
    from llama_index.llms.openai import OpenAIResponses

    llms.append(OpenAIResponses(max_retries=0))


@pytest.mark.disable_socket
@pytest.mark.parametrize("llm", llms)
def test_openai_chat_with_tools(
    llm: FunctionCallingLLM,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    n = randint(1, 5)
    tools = [FunctionTool.from_defaults(fn=lambda: None) for _ in range(n)]
    with suppress(Exception):
        llm.chat_with_tools(tools, "")
    spans = in_memory_span_exporter.get_finished_spans()
    assert spans
    span = next(s for s in spans if s.name.endswith("chat"))
    attributes = dict(span.attributes or {})
    for i in range(len(tools)):
        assert attributes[f"llm.tools.{i}.tool.json_schema"]


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
