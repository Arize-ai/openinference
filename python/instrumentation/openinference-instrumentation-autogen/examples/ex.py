from openinference.instrumentation.autogen import AutogenInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference-instrumentation-openai import OpenAIInstrumentor


def main():
    endpoint = "http://0.0.0.0:6006/v1/traces"
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    OpenAIInstrumentor().instrument(tracer_provider=trace_provider)
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
