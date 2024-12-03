# OpenInference Core

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-core.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-core)

This package provides OpenInference Core utilities for LLM Traces.

## Installation

```bash
npm install @arizeai/openinference-core # npm
pnpm add @arizeai/openinference-core # pnpm
yarn add @arizeai/openinference-core # yarn
```

## Customizing Spans

The `@arizeai/openinference-core` package offers utilities to track important application metadata such as sessions and users using context attribute propagation:

- `setSession`: to specify a session ID to track and group multi-turn conversations
- `setUser`: to specify a user ID to track different conversations with a given user
- `setMetadata`: to add custom metadata that can provide extra information to support a wide range of operational needs
- `setTag`: to add tags, to filter spans on specific keywords
- `setPromptTemplate`: to reflect the prompt template used, with its version and variables. This is useful for prompt template tracking
- `setAttributes`: to add multiple custom attributes at the same time

> [!NOTE] All @arizeai/openinference auto instrumentation packages will pull attributes off of context and add them to spans

### Examples

`setSession`

```typescript
import { context } from "@opentelemetry/api";
import { setSession } from "@arizeai/openinference-core";

context.with(setSession(context.active(), { sessionId: "session-id" }), () => {
  // Calls within this block will generate spans with the attributes:
  // "session.id" = "session-id"
});
```

Each setter function returns a new active context, so they can be chained together.

```typescript
import { context } from "@opentelemetry/api";
import { setAttributes, setSession } from "@arizeai/openinference-core";

context.with(
  setAttributes(setSession(context.active(), { sessionId: "session-id" }), {
    myAttribute: "test",
  }),
  () => {
    // Calls within this block will generate spans with the attributes:
    // "myAttribute" = "test"
    // "session.id" = "session-id"
  },
);
```

Additionally, they can be used in conjunction with the [OpenInference Semantic Conventions](../openinference-semantic-conventions/).

```typescript
import { context } from "@opentelemetry/api"
import { setAttributes } from "@openinference-core"
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";


context.with(
  setAttributes(
    { [SemanticConventions.SESSION_ID: "session-id" }
  ),
  () => {
      // Calls within this block will generate spans with the attributes:
      // "session.id" = "session-id"
  }
)
```

If you are creating spans manually and want to propagate context attributes you've set to those spans as well you can use the `getAttributesFromContext` utility to do that. you can read more about customizing spans in our [docs](https://docs.arize.com/phoenix/tracing/how-to-tracing/customize-spans).

```typescript
import { getAttributesFromContext } from "@arizeai/openinference-core";
import { context, trace } from "@opentelemetry/api";

const contextAttributes = getAttributesFromContext(context.active());
const tracer = trace.getTracer("example");
const span = tracer.startSpan("example span");
span.setAttributes(contextAttributes);
span.end();
```

## Trace Config

This package also provides support for controlling settings like data privacy and payload sizes. For instance, you may want to keep sensitive information from being logged for security reasons, or you may want to limit the size of the base64 encoded images logged to reduced payload size.

> [!NOTE] These values can also be controlled via environment variables, see more information [here](https://github.com/Arize-ai/openinference/blob/main/spec/configuration.md).

Here is an example of how to configure these settings using the OpenAI auto instrumentation. Note that all of our auto instrumentations will accept a traceConfig object.

```typescript
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";

/**
 * Everything left out of here will fallback to
 * environment variables then defaults
 */
const traceConfig = { hideInputs: true };

const instrumentation = new OpenAIInstrumentation({ traceConfig });
```
