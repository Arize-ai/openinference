from langchain_core.prompts import PromptTemplate
from openinference.instrumentation.langchain._tracer import _get_cls_name


def test_get_cls_name() -> None:
    serialized = PromptTemplate(template="", input_variables=[]).to_json()
    assert _get_cls_name(serialized) == "PromptTemplate"
