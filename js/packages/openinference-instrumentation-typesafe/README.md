# OpenInference TypeSafe AI Instrumentation

Traces `@typesafe-ai/sdk` model invocations with OpenInference attributes. Each
`TypeSafeClient.systemOne` call produces one `LLM` span, including all questions
and SDK retries. `client.models.list()` is not instrumented.

Requires Node.js 20+ and `@typesafe-ai/sdk >=0.6.0 <0.7.0`. Tested against 0.6.0.

```sh
pnpm add @arizeai/openinference-instrumentation-typesafe @typesafe-ai/sdk @opentelemetry/api @opentelemetry/instrumentation
```

## Usage

Configure an OpenTelemetry tracer provider and exporter, then register the
instrumentation **before requiring the SDK** in CommonJS applications:

```ts
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { TypeSafeInstrumentation } from "@arizeai/openinference-instrumentation-typesafe";

registerInstrumentations({ instrumentations: [new TypeSafeInstrumentation()] });
```

For native ESM, bundled applications, or an SDK imported before registration,
pass the imported namespace to `manuallyInstrument`:

```ts
import * as TypeSafe from "@typesafe-ai/sdk";
import { TypeSafeInstrumentation } from "@arizeai/openinference-instrumentation-typesafe";

const instrumentation = new TypeSafeInstrumentation();
instrumentation.manuallyInstrument(TypeSafe);

const client = new TypeSafe.TypeSafeClient(); // reads TYPESAFE_API_KEY
const { data, requestId } = await client
  .systemOne({
    state: { document: "I was charged twice. Please fix this ASAP." },
    questions: {
      category: TypeSafe.choice("What is this ticket about?", {
        billing: null,
        technical: null,
        other: null,
      }),
    },
  })
  .withResponse();

console.log(data.answers.category, requestId);
```

The return value remains an SDK `APIPromise`: `await`, `then`, `catch`, `finally`,
`withResponse()`, `asResponse()`, and `map()` retain their SDK behavior. Telemetry
reads a clone of the buffered response, leaving the caller's raw response body
unread. This adds a response clone and JSON parse per invocation. Spans finish
before the wrapped response is delivered, even if the caller never awaits it.

`disable()` restores all SDK prototypes patched by this instance. `enable()`
reapplies both automatic and manually registered instrumentation. CJS and ESM
builds can be manually instrumented in the same process.

## Captured attributes

| Attribute                                        | Value                                                                        |
| ------------------------------------------------ | ---------------------------------------------------------------------------- |
| Span name / kind                                 | `TypeSafeClient.systemOne` / `LLM`                                           |
| `llm.provider`, `llm.system`                     | `typesafe`                                                                   |
| `llm.model_name`                                 | Response model, falling back to the request or client default                |
| `input.value`                                    | Complete JSON request including state, questions, and resolved request model |
| `output.value`                                   | JSON response including answers, usage, and model                            |
| `input.mime_type`, `output.mime_type`            | `application/json`                                                           |
| `llm.token_count.*`                              | Prompt/completion when present; total only when both exist                   |
| `llm.invocation_parameters`                      | Model and explicitly supplied timeout/retry overrides                        |
| `metadata.typesafe` (inside the `metadata` JSON) | Request ID and question types/confidence                                     |
| `http.response.status_code`                      | HTTP status on SDK API errors                                                |

TypeSafe evaluates state against typed questions and returns structured answers.
The complete state, questions, instructions, and criteria are captured in
`input.value`; answers, probabilities, and score legends are captured in
`output.value`. No `llm.input_messages` or `llm.output_messages` attributes are
emitted, including when the state contains a conversation. Noul answers report a
probability of yes; no synthetic confidence is assigned to them. The span kind
is always `LLM`, regardless of whether the application uses the call as a
guardrail, router, or evaluator.

Metadata from the active OpenInference context is preserved, with `typesafe`
reserved for the instrumentor. For example:

```json
{
  "workflow": "support-routing",
  "typesafe": {
    "request_id": "req_123",
    "questions": { "category": { "type": "choice", "confidence": 0.95 } }
  }
}
```

The same shape can be used by the companion Python instrumentation in
[#3769](https://github.com/Arize-ai/openinference/issues/3769).

## Configuration and privacy

```ts
const instrumentation = new TypeSafeInstrumentation({
  tracerProvider, // optional; defaults to the global provider
  instrumentationConfig: { enabled: true },
  traceConfig: { hideInputs: true, hideOutputs: true },
});
```

Standard OpenInference context attributes (session, user, tags, metadata), tracing
suppression, and `TraceConfig` environment variables are supported. `hideInputs`
redacts the input payload and removes question metadata. `hideOutputs` redacts
the output payload and removes confidence metadata. `hideInputMessages` and
`hideOutputMessages` have no effect on this instrumentation because it emits no
chat-message attributes; use `hideInputs` / `hideOutputs` to hide the payloads.

Request headers, credentials, and abort signals are never copied into invocation
parameters. Standard exception events and status messages are recorded on SDK
errors, including synchronous validation errors and cancellation. As with other
instrumentors, payload masking does not redact exception messages or caller-supplied
context metadata.

## Examples

With Phoenix running at `http://localhost:6006`, export `TYPESAFE_API_KEY` and run
from this repository:

```sh
cd js
pnpm install --frozen-lockfile -r
pnpm --filter '@arizeai/openinference-instrumentation-typesafe...' run build
cd packages/openinference-instrumentation-typesafe
pnpm exec tsx examples/basic-usage.ts
```

| Example                                                 | Phoenix project               | Description                                                                                     |
| ------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------- |
| [basic-usage.ts](examples/basic-usage.ts)               | `typesafe-basic-usage`        | Classification and `withResponse()`                                                             |
| [all-question-types.ts](examples/all-question-types.ts) | `typesafe-all-question-types` | Choice, noul, score, and structured inputs                                                      |
| [guardrail-routing.ts](examples/guardrail-routing.ts)   | `typesafe-guardrail-routing`  | A TypeSafe decision gates an OpenAI call under one parent trace; also requires `OPENAI_API_KEY` |

Examples print spans and flush to local Phoenix before exit. The routing example
emits an OpenAI child span only when TypeSafe chooses `allow` with confidence
at least 0.8; otherwise it routes to human review.

Implementation follows the SDK's [client](https://github.com/typesafe-ai/typesafe-sdk-js/blob/66880ccded6cb642dc1809620c2b108c33730214/src/client.ts),
[APIPromise](https://github.com/typesafe-ai/typesafe-sdk-js/blob/66880ccded6cb642dc1809620c2b108c33730214/src/api-promise.ts),
and [question types](https://docs.typesafe.ai/primitives/advanced).
