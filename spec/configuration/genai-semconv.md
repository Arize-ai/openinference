# GenAI Semantic Conventions Export

**This capability is experimental.** The OpenTelemetry [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
are still in development, and the mapping described here may change as they evolve.

When `OPENINFERENCE_ENABLE_GENAI_SEMCONV` is enabled, an instrumentation emits OpenTelemetry `gen_ai.*` attributes on
each span in addition to its OpenInference attributes. One trace can then be read by backends that understand either
convention. The setting is off by default.

| Environment variable | Type | Default | Effect |
| --- | --- | --- | --- |
| `OPENINFERENCE_ENABLE_GENAI_SEMCONV` | bool | `false` | Emits OTel GenAI semantic convention attributes alongside OpenInference attributes. Experimental |

In code, the setting is `TraceConfig(enable_genai_semconv=True)`. It follows the usual
[precedence](./trace-config.md#precedence): code, then environment variable, then default.

## Language Status

| Language | `OPENINFERENCE_ENABLE_GENAI_SEMCONV` | Weaver conformance tests |
| --- | --- | --- |
| Python | ✅ Supported (experimental) | ✅ Anthropic, OpenAI, and Google GenAI |
| JavaScript | 🗓️ Planned ([#2983](https://github.com/Arize-ai/openinference/issues/2983)) | 🗓️ Planned ([#3114](https://github.com/Arize-ai/openinference/issues/3114)) |
| Java | 🗓️ Planned ([#2983](https://github.com/Arize-ai/openinference/issues/2983)) | 🗓️ Planned ([#3114](https://github.com/Arize-ai/openinference/issues/3114)) |
| Go | 🗓️ Planned ([#2983](https://github.com/Arize-ai/openinference/issues/2983)) | 🗓️ Planned ([#3114](https://github.com/Arize-ai/openinference/issues/3114)) |

Languages marked 🗓️ Planned ignore the variable. Broader OTel GenAI support is discussed in
[#2205](https://github.com/Arize-ai/openinference/issues/2205).

JavaScript also ships [`@arizeai/openinference-genai`](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-genai),
which converts the other way: it turns `gen_ai.*` spans into OpenInference attributes. It does not implement this
setting.

## How It Works

The export is a dual-write. The instrumentation records OpenInference attributes as usual. When the span ends, it
derives the `gen_ai.*` attributes from the OpenInference attributes recorded on the span:

- The span kind sets `gen_ai.operation.name`. `LLM` becomes `chat`, or `text_completion` when the span carries
  completions API prompts or choices (`llm.prompts` or `llm.choices`). `TOOL` becomes `execute_tool`, `EMBEDDING`
  becomes `embeddings`, `RETRIEVER` becomes `retrieval`, and `AGENT` becomes `invoke_agent`.
- `llm.provider` and `llm.system` set `gen_ai.provider.name`. `session.id` sets `gen_ai.conversation.id`.
- `llm.model_name` (or `embedding.model_name`) and `llm.invocation_parameters` (or
  `embedding.invocation_parameters`) set `gen_ai.request.*`. Token counts set `gen_ai.usage.*`. Output messages set
  `gen_ai.response.finish_reasons`, `gen_ai.response.id`, and `gen_ai.response.model` when available.
- `agent.name` sets `gen_ai.agent.name`. On `TOOL` spans, `tool.name`, `tool.description`, `tool.id`,
  `tool.parameters`, and `output.value` set `gen_ai.tool.*`. On `RETRIEVER` spans, a non-JSON `input.value` sets
  `gen_ai.retrieval.query.text`. On `EMBEDDING` spans, the vector length sets `gen_ai.embeddings.dimension_count`.
- Input and output messages, tool definitions, and retrieved documents are written as JSON strings to
  `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.tool.definitions`, and `gen_ai.retrieval.documents`.

An instrumentation MUST NOT overwrite a `gen_ai.*` attribute that is already on the span. A value the instrumentation
set explicitly, such as `gen_ai.operation.name = "plan"`, takes precedence over the derived one.

The `gen_ai.*` values are derived after [data masking](./data-masking.md) has been applied. Content that a setting
hides from the OpenInference attributes is therefore hidden from the `gen_ai.*` attributes as well.

In Python the conversion lives in [`_genai_conversion.py`](https://github.com/Arize-ai/openinference/blob/main/python/openinference-instrumentation/src/openinference/instrumentation/_genai_conversion.py).
It runs for spans created through `OITracer`, which is the tracer most Python instrumentors use.

## Conformance Testing with Weaver

The setting is checked against the official OTel semantic conventions registry using
[Weaver](https://github.com/open-telemetry/weaver)'s `registry live-check`. The harness lives in
[`python/openinference-instrumentation/scripts/conformance/`](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation/scripts/conformance).

```bash
uv run python/openinference-instrumentation/scripts/conformance/run.py
```

The runner does the following:

1. It starts a mock server that returns fixed Anthropic, OpenAI, and Google GenAI responses, so no API keys are
   needed.
2. It starts `weaver registry live-check` as an OTLP receiver loaded with the GenAI semconv registry.
3. It runs one script per provider. Each script instruments its client with
   `TraceConfig(enable_genai_semconv=True)`, covers chat and tool calling, and exports spans to Weaver. The OpenAI
   script also covers embeddings. The Anthropic script also creates manual `TOOL`, `EMBEDDING`, and `RETRIEVER` spans
   through `OITracer` to reach conversion paths the SDK instrumentation does not.
4. It prints a summary of the `gen_ai.*` registry attributes seen, the registry attributes missing, and Weaver's
   violation and improvement counts.

Weaver reports OpenInference's own attributes (`llm.*`, `input.*`, `output.*`) as non-registry attributes. This is
expected. Conformance is judged on the `gen_ai.*` attributes. The runner pins its Weaver and semconv versions at the
top of `run.py`. See the harness
[README](https://github.com/Arize-ai/openinference/blob/main/python/openinference-instrumentation/scripts/conformance/README.md)
for how to read the summary and add scenarios.

A language that adds support for this setting SHOULD add an equivalent Weaver live-check harness.
