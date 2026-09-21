"""Compare OpenAI cache usage with instrumented spans; export to local Phoenix.

See cache_tokens.md for installation, execution, and Phoenix readback commands.
"""

import argparse
import json
from uuid import uuid4

from openai import OpenAI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.resource import ResourceAttributes
from openinference.semconv.trace import SpanAttributes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--api", choices=("responses", "chat"), default="responses")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--endpoint", default="http://localhost:6006/v1/traces")
    parser.add_argument("--project", default="openai-cache-tokens")
    args = parser.parse_args()

    memory = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({ResourceAttributes.PROJECT_NAME: args.project})
    )
    provider.add_span_processor(SimpleSpanProcessor(memory))
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=args.endpoint)))
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    # A unique leading value starts a fresh cache prefix on each run. Keep this
    # entire message unchanged between calls and above the cache minimum.
    prefix = f"Reference ID: {uuid4()}.\n" + "\n".join(
        f"Reference entry {i}: cache reads reuse input; cache writes store input for later reuse."
        for i in range(100)
    )
    counts = []
    try:
        with OpenAI() as client:
            for question in ("Reply with only OK.", "Reply with only DONE."):
                if args.api == "responses":
                    result = client.responses.create(
                        model=args.model,
                        instructions=prefix,
                        input=question,
                        reasoning={"effort": "low"},
                        max_output_tokens=128,
                        stream=args.stream,
                    )
                    if args.stream:
                        usage = None
                        for event in result:
                            if event.type in ("response.completed", "response.incomplete"):
                                usage = event.response.usage
                    else:
                        usage = result.usage
                    assert usage is not None, "No final usage received"
                    details = usage.input_tokens_details
                    prompt, completion = usage.input_tokens, usage.output_tokens
                else:
                    kwargs = {"stream_options": {"include_usage": True}} if args.stream else {}
                    result = client.chat.completions.create(
                        model=args.model,
                        messages=[
                            {"role": "system", "content": prefix},
                            {"role": "user", "content": question},
                        ],
                        reasoning_effort="low",
                        max_completion_tokens=128,
                        stream=args.stream,
                        **kwargs,
                    )
                    if args.stream:
                        usage = None
                        for chunk in result:
                            if chunk.usage is not None:
                                usage = chunk.usage
                    else:
                        usage = result.usage
                    assert usage is not None, "No final usage received"
                    details = usage.prompt_tokens_details
                    prompt, completion = usage.prompt_tokens, usage.completion_tokens
                read = getattr(details, "cached_tokens", None)
                write = getattr(details, "cache_write_tokens", None)
                assert read is not None and write is not None, "Model did not report cache counts"
                expected = {
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT: prompt,
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: completion,
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL: usage.total_tokens,
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: read,
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: write,
                }
                spans = memory.get_finished_spans()
                assert len(spans) == 1, f"Expected one LLM span, got {len(spans)}"
                span = spans[0]
                for key, value in expected.items():
                    assert (span.attributes or {}).get(key) == value, (key, value, span.attributes)
                assert read + write <= prompt, "Cache counts must be subsets of input tokens"
                assert span.context is not None
                print(
                    json.dumps(
                        {
                            "trace_id": f"{span.context.trace_id:032x}",
                            "span_id": f"{span.context.span_id:016x}",
                            "usage": usage.model_dump(),
                            "attributes": expected,
                        }
                    )
                )
                counts.append((read, write))
                memory.clear()
        assert counts[0][1] > 0, "Cold request did not produce cache writes"
        assert counts[1][0] > 0, "Warm request did not produce cache reads; try another run"
    finally:
        provider.shutdown()
        OpenAIInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
