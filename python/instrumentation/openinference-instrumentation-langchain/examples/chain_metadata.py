from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LangChainInstrumentor().instrument()


if __name__ == "__main__":
    prompt_template = "Tell me a {adjective} joke"
    prompt = PromptTemplate(input_variables=["adjective"], template=prompt_template)
    llm = LLMChain(llm=OpenAI(), prompt=prompt, metadata={"category": "jokes"})
    completion = llm.predict(adjective="funny", metadata={"variant": "funny"})
    print(completion)
