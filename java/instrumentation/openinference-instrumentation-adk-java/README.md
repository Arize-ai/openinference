# OpenInference Instrumentation for Google ADK Java

A Java agent (`-javaagent`) that decorates the OpenTelemetry spans created by
[Google ADK for Java](https://github.com/google/adk-java) with
[OpenInference](https://github.com/Arize-ai/openinference/tree/main/spec) attributes:

- `invocation` spans become `CHAIN` spans carrying `session.id`, `user.id`, `agent.name` and the
  new user message as `input.value`.
- `agent_run [...]` spans become `AGENT` spans with the final response as `output.value`.
- `call_llm` spans become `LLM` spans with `llm.model_name`, `llm.provider`, input and output
  messages (including tool calls and tool results), tool schemas and token counts.
- `tool_call [...]` spans become `TOOL` spans with `tool.name`, `tool.description`, `tool.parameters`
  and the tool result as `output.value` (ADK reports the result on a separate `tool_response [...]`
  span, which is kept as a `CHAIN` step).

## Usage

Build or download the fat `-all` jar and pass it to the JVM. The application must register a
global OpenTelemetry SDK before its first ADK call; ADK's `Telemetry` class captures
`GlobalOpenTelemetry` when it is loaded.

```bash
./gradlew :instrumentation:openinference-instrumentation-adk-java:shadowJar
java -javaagent:openinference-instrumentation-adk-java-<version>-all.jar -cp <classpath> com.example.Main
```

Tracing can be suppressed with `SuppressTracing.begin()` and sensitive fields masked through
`TraceConfig` (register an `OITracer` with `OpenInferenceAgent.register(...)`), as with the other
OpenInference Java instrumentors.

## Examples

See [`java/examples/adk-java-example`](../../examples/adk-java-example) for a runnable agent with a
function tool that exports to a local Phoenix.
