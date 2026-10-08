/**
 * Typesafe example backing the code snippets in docs/trace-config-and-masking.md
 *
 * Exercises: OITracer constructor, TraceConfigOptions, generateTraceConfig,
 * wrapTracer, getTracer, withSafety, safelyJSONStringify, safelyJSONParse
 */

import {
  OpenInferenceSpanKind,
  SEMRESATTRS_PROJECT_NAME,
} from "@arizeai/openinference-semantic-conventions";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { resourceFromAttributes } from "@opentelemetry/resources";
import {
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { trace } from "@opentelemetry/api";

import {
  generateTraceConfig,
  getTracer,
  OITracer,
  safelyJSONParse,
  safelyJSONStringify,
  withSafety,
  withSpan,
  wrapTracer,
} from "../src";

// -- Provider setup -----------------------------------------------------------

const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [SEMRESATTRS_PROJECT_NAME]: "trace-config-example",
  }),
  spanProcessors: [
    new SimpleSpanProcessor(
      new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" }),
    ),
  ],
});
provider.register();

// -- OITracer with masking (docs/trace-config-and-masking.md) -----------------

function oiTracerDemo() {
  const maskedTracer = new OITracer({
    tracer: trace.getTracer("my-service"),
    traceConfig: {
      hideInputs: true,
      hideOutputText: true,
    },
  });

  const traced = withSpan(
    async (prompt: string) => `response to: ${prompt}`,
    {
      tracer: maskedTracer,
      name: "sensitive-operation",
    },
  );

  return traced;
}

// -- Production setup with masking (docs/trace-config-and-masking.md) ---------

function productionSetupDemo() {
  const tracer = new OITracer({
    tracer: trace.getTracer("my-llm-app"),
    traceConfig: {
      hideInputImages: true,
      base64ImageMaxLength: 8000,
      hideEmbeddingVectors: true,
    },
  });

  const chat = withSpan(
    async (prompt: string) => `echo: ${prompt}`,
    {
      tracer,
      name: "chat",
      kind: OpenInferenceSpanKind.LLM,
    },
  );

  return chat;
}

// -- generateTraceConfig ------------------------------------------------------

function generateTraceConfigDemo() {
  const config = generateTraceConfig({ hideInputs: true });
  console.log("hideInputs:", config.hideInputs); // true
  console.log("hideOutputs:", config.hideOutputs); // false (default)
  console.log("base64ImageMaxLength:", config.base64ImageMaxLength); // 32000 (default)
}

// -- wrapTracer / getTracer ---------------------------------------------------

function tracerHelpersDemo() {
  // Wrap an existing OTel tracer
  const existingTracer = trace.getTracer("existing-service");
  const oiTracer = wrapTracer(existingTracer);

  // Create an OITracer from the global provider
  const defaultTracer = getTracer("my-service");

  // Use with withSpan
  withSpan(async () => "hello", { tracer: oiTracer, name: "wrapped" });
  withSpan(async () => "world", { tracer: defaultTracer, name: "default" });
}

// -- withSafety ---------------------------------------------------------------

function withSafetyDemo() {
  const safeParse = withSafety({
    fn: (input: string) => JSON.parse(input) as unknown,
    onError: (err) => console.warn("Parse failed:", err),
  });

  const invalid = safeParse("invalid json"); // returns null, logs warning
  console.log("invalid result:", invalid);

  const valid = safeParse('{"ok": true}'); // returns { ok: true }
  console.log("valid result:", valid);
}

// -- safelyJSONStringify / safelyJSONParse ------------------------------------

function safelyJSONDemo() {
  const stringified = safelyJSONStringify({ key: "value" });
  console.log("stringified:", stringified); // '{"key":"value"}'

  const undefinedResult = safelyJSONStringify(undefined);
  console.log("undefined stringify:", undefinedResult); // null (no throw)

  const parsed = safelyJSONParse('{"key": "value"}');
  console.log("parsed:", parsed); // { key: "value" }

  const badParse = safelyJSONParse("not json");
  console.log("bad parse:", badParse); // null (no throw)
}

// -- Run all demos ------------------------------------------------------------

async function main() {
  const sensitiveFn = oiTracerDemo();
  await sensitiveFn("test prompt");

  const chatFn = productionSetupDemo();
  await chatFn("What is OpenInference?");

  generateTraceConfigDemo();
  tracerHelpersDemo();
  withSafetyDemo();
  safelyJSONDemo();
}

main();
