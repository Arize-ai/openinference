import os
from importlib import import_module

import autogen
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.autogen import AutogenInstrumentor


# mypy: ignore-errors
def main():
    trace_provider = TracerProvider()

    endpoint = "http://127.0.0.1:6006/v1/traces"
    exporter = OTLPSpanExporter(endpoint=endpoint)
    span_processor = SimpleSpanProcessor(exporter)
    trace_provider.add_span_processor(span_processor)

    trace.set_tracer_provider(trace_provider)

    openai_instrumentation = import_module("openinference.instrumentation.openai")
    openai_instrumentation.OpenAIInstrumentor().instrument(tracer_provider=trace_provider)
    AutogenInstrumentor().instrument()

    config_list = [
        {
            "model": "gpt-4o",
            "api_key": os.environ["OPENAI_API_KEY"],
        }
    ]

    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config={
            "seed": 42,  # seed for caching and reproducibility
            "config_list": config_list,
            "temperature": 0,  # temperature for sampling
        },
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,  # set to True or an image name like "python:3" to use docker
        },
    )

    user_proxy.initiate_chat(
        assistant,
        message="""What date is today? Compare the year-to-date gain for META and TESLA.""",
    )


if __name__ == "__main__":
    main()
