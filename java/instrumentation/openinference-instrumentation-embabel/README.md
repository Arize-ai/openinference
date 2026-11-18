# OpenInference Instrumentation for Embabel

Adds [OpenInference](https://github.com/Arize-ai/openinference/tree/main/spec) attributes to the
OpenTelemetry spans that the [Embabel](https://github.com/embabel/embabel-agent) agent framework
emits through its `embabel-agent-observability` module, so an Embabel run shows up in Phoenix or
Arize with agent, LLM and tool views.

Embabel already traces every agent run, action, LLM call and tool call. This module maps those
spans at export time:

| Embabel span (`embabel.event.type`) | OpenInference span kind | Attributes added |
|---|---|---|
| `agent_process` | `AGENT` | `agent.name`, `session.id` (from `gen_ai.conversation.id`) |
| Spring AI chat model span (`gen_ai.operation.name=chat`) | `LLM` | `llm.model_name`, `llm.provider`, `llm.token_count.*`, `llm.invocation_parameters`, `llm.input_messages.*`, `llm.output_messages.*` |
| `tool_call` | `TOOL` | `tool.name`, `tool.description`, `tool.parameters`, tool arguments and result as `input.value` / `output.value` |
| `embedding` | `EMBEDDING` | `embedding.model_name` |
| `rag` | `RETRIEVER` | the query as `input.value` |
| every other Embabel step (`action`, `goal`, `planning`, `tool_loop`, `llm_call`, ...) | `CHAIN` | |

`input.mime_type` / `output.mime_type` are set wherever Embabel recorded an input or output.
Embabel's own attributes are kept. Spans that are not Embabel or GenAI spans, and spans that already
carry `openinference.span.kind`, are exported unchanged.

The `llm <model>` wrapper span and the `llm.invocation` cost record stay `CHAIN` spans: the chat
model span nested under them is the generation, and marking them `LLM` as well would count the same
tokens twice.

## Installation

```xml
<dependency>
    <groupId>com.arize</groupId>
    <artifactId>openinference-instrumentation-embabel</artifactId>
    <version>${openinference.version}</version>
</dependency>
```

The application also needs Embabel's observability starter and an OTLP exporter, for example
`io.opentelemetry:opentelemetry-exporter-otlp` with
`management.otlp.tracing.endpoint=http://localhost:6006/v1/traces` for a local Phoenix.

## Usage

Wrap the application's `SpanExporter` with `OpenInferenceEmbabelSpanExporter`, and register
`OpenInferenceEmbabelSpanProcessor` so that session, user, metadata and tags set with
`ContextAttributes` reach Embabel's spans. With Spring Boot:

```java
@Configuration
class OpenInferenceTracingConfiguration {

    @Bean
    static BeanPostProcessor openInferenceEmbabelSpanExporter() {
        return new BeanPostProcessor() {
            @Override
            public Object postProcessAfterInitialization(Object bean, String beanName) {
                if (bean instanceof SpanExporter exporter && !(bean instanceof OpenInferenceEmbabelSpanExporter)) {
                    return new OpenInferenceEmbabelSpanExporter(exporter);
                }
                return bean;
            }
        };
    }

    @Bean
    OpenInferenceEmbabelSpanProcessor openInferenceEmbabelSpanProcessor() {
        return new OpenInferenceEmbabelSpanProcessor();
    }
}
```

Without Spring Boot's tracing auto-configuration:

```java
SpanExporter otlp = OtlpHttpSpanExporter.builder()
        .setEndpoint("http://localhost:6006/v1/traces")
        .build();
SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
        .addSpanProcessor(new OpenInferenceEmbabelSpanProcessor())
        .addSpanProcessor(BatchSpanProcessor.builder(new OpenInferenceEmbabelSpanExporter(otlp)).build())
        .build();
```

## Masking

Pass a `TraceConfig` to both classes to hide content:

```java
TraceConfig config = TraceConfig.builder().hideInputMessages(true).build();
new OpenInferenceEmbabelSpanExporter(exporter, config);
new OpenInferenceEmbabelSpanProcessor(config);
```

Hidden content is removed from the exported span, including the copies Embabel records under its
own and the GenAI keys (`gen_ai.input.messages`, `gen_ai.tool.call.arguments`,
`embabel.agent.result`, ...). On LLM spans the message flags also hide `input.value` /
`output.value`, which hold the prompt and completion text.

## Suppressing tracing

Spans started inside `SuppressTracing.begin()` are exported unchanged, without OpenInference
attributes. The exporter cannot see the context a span started in, so this needs
`OpenInferenceEmbabelSpanProcessor` registered as shown above. `TraceConfig.suppressTracing(true)`
turns off the mapping for every span.
