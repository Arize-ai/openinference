import json

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes


def test_chat_prompt_template(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    adjective = "helpful"
    name = "bob"
    system_msg = "You are a {adjective} AI bot. Your name is {name}."
    reply = "cool"
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{reply}"),  # FIXME: currently we can only capture the first template
        ]
    ).partial(adjective=adjective)
    variables = dict(adjective=adjective, name=name)
    input = dict(name=name, reply=reply)
    template.invoke(input)
    assert (spans := in_memory_span_exporter.get_finished_spans())
    span = spans[0]
    assert (attributes := dict(span.attributes or {}))
    assert (llm_prompt_template := attributes.pop(LLM_PROMPT_TEMPLATE, None))
    assert isinstance(llm_prompt_template, str)
    assert llm_prompt_template == system_msg
    assert (variables_json_str := attributes.pop(LLM_PROMPT_TEMPLATE_VARIABLES, None))
    assert isinstance(variables_json_str, str)
    assert json.loads(variables_json_str) == variables
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None)
    assert (input_value_json_str := attributes.pop(INPUT_VALUE, None))
    assert isinstance(input_value_json_str, str)
    assert json.loads(input_value_json_str) == input
    assert attributes.pop(INPUT_MIME_TYPE, None) == JSON
    assert attributes.pop(OUTPUT_VALUE, None)
    assert attributes.pop(OUTPUT_MIME_TYPE, None) == JSON
    assert attributes == {}


def test_prompt_template(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    adjective = "helpful"
    name = "bob"
    template_str = "You are a {adjective} AI bot. Your name is {name}."
    template = PromptTemplate.from_template(template_str)
    variables = dict(adjective=adjective, name=name)
    template.invoke(variables)
    assert (spans := in_memory_span_exporter.get_finished_spans())
    span = spans[0]
    assert (attributes := dict(span.attributes or {}))
    assert (llm_prompt_template := attributes.pop(LLM_PROMPT_TEMPLATE, None))
    assert isinstance(llm_prompt_template, str)
    assert llm_prompt_template == template_str
    assert (variables_json_str := attributes.pop(LLM_PROMPT_TEMPLATE_VARIABLES, None))
    assert isinstance(variables_json_str, str)
    assert json.loads(variables_json_str) == variables
    assert attributes.pop(OPENINFERENCE_SPAN_KIND, None)
    assert (input_value_json_str := attributes.pop(INPUT_VALUE, None))
    assert isinstance(input_value_json_str, str)
    assert json.loads(input_value_json_str) == variables
    assert attributes.pop(INPUT_MIME_TYPE, None) == JSON
    assert attributes.pop(OUTPUT_VALUE, None)
    assert attributes.pop(OUTPUT_MIME_TYPE, None) == JSON
    assert attributes == {}


INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
JSON = OpenInferenceMimeTypeValues.JSON.value
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
