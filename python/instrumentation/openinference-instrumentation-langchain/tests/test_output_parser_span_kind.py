"""Tests that LangChain output-parser runs are traced as CHAIN spans.

LangChain emits callback runs with ``run_type="parser"`` for output parsers such
as ``StrOutputParser`` and ``PydanticOutputParser``. ``OpenInferenceSpanKindValues``
has no ``PARSER`` member, so historically these spans were recorded as ``UNKNOWN``.
To match the JavaScript instrumentor, any unrecognized run type now falls back to
``CHAIN``. See https://github.com/Arize-ai/openinference/issues/3485.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from openinference.instrumentation.langchain._tracer import _langchain_run_type_to_span_kind
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


@pytest.mark.parametrize(
    "run_type, expected",
    [
        ("llm", OpenInferenceSpanKindValues.LLM),
        ("LLM", OpenInferenceSpanKindValues.LLM),
        ("tool", OpenInferenceSpanKindValues.TOOL),
        ("retriever", OpenInferenceSpanKindValues.RETRIEVER),
        ("chain", OpenInferenceSpanKindValues.CHAIN),
        ("prompt", OpenInferenceSpanKindValues.PROMPT),
        # "parser" has no dedicated span kind and must fall back to CHAIN.
        ("parser", OpenInferenceSpanKindValues.CHAIN),
        # Any future/unrecognized run type must also fall back to CHAIN, matching JS.
        ("some_new_run_type", OpenInferenceSpanKindValues.CHAIN),
    ],
)
def test_run_type_to_span_kind(run_type: str, expected: OpenInferenceSpanKindValues) -> None:
    assert _langchain_run_type_to_span_kind(run_type) is expected


def test_str_output_parser_span_is_chain(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    output = StrOutputParser().invoke(AIMessage(content="hello world"))
    assert output == "hello world"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "StrOutputParser"
    assert span.attributes is not None
    assert (
        span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.CHAIN.value
    )
    assert span.attributes[SpanAttributes.OUTPUT_VALUE] == "hello world"


def test_pydantic_output_parser_span_is_chain(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    class Person(BaseModel):
        name: str
        age: int

    parser = PydanticOutputParser(pydantic_object=Person)
    output = parser.invoke(AIMessage(content='{"name": "Alice", "age": 30}'))
    assert output == Person(name="Alice", age=30)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "PydanticOutputParser"
    assert span.attributes is not None
    assert (
        span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
        == OpenInferenceSpanKindValues.CHAIN.value
    )
