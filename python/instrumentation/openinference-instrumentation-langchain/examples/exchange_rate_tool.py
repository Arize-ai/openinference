import requests
from langchain import agents
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.langchain import LangChainInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


@tool
def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "EUR",
    currency_date: str = "latest",
):
    """Retrieves the exchange rate between two currencies on a specified date."""
    return requests.get(
        f"https://api.frankfurter.app/{currency_date}",
        params={"from": currency_from, "to": currency_to},
    ).json()


tools = [get_exchange_rate]
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
agent = agents.create_tool_calling_agent(llm, tools, prompt)
agent_executor = agents.AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    agent_executor.invoke(
        {"input": "What is the exchange rate from US dollars to Swedish currency today?"}
    )
