/* eslint-disable no-console */
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { SpanStatusCode } from "@opentelemetry/api";
import { resourceFromAttributes } from "@opentelemetry/resources";
import {
  BatchSpanProcessor,
  NodeTracerProvider,
} from "@opentelemetry/sdk-trace-node";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

import { OpenInferenceOTLPTraceExporter } from "./openinferenceOTLPTraceExporter.js";

// setup tracing provider and custom exporter

const COLLECTOR_ENDPOINT = process.env.COLLECTOR_ENDPOINT;
const SERVICE_NAME = "openinference-genai-app";

export const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [ATTR_SERVICE_NAME]: SERVICE_NAME,
    [SEMRESATTRS_PROJECT_NAME]: SERVICE_NAME,
  }),
  spanProcessors: [
    new BatchSpanProcessor(
      new OpenInferenceOTLPTraceExporter({
        url: `${COLLECTOR_ENDPOINT}/v1/traces`,
      }),
    ),
  ],
});

provider.register();

// send a test genai span
const main = async () => {
  if (!COLLECTOR_ENDPOINT) {
    throw new Error("COLLECTOR_ENDPOINT is not set");
  }
  const tracer = provider.getTracer("test-genai-span");
  tracer.startActiveSpan("test-genai-span", (span) => {
    console.log("creating span");
    span.setAttributes({
      "gen_ai.provider.name": "openai",
      "gen_ai.operation.name": "chat",
      "gen_ai.request.model": "gpt-4",
      "gen_ai.request.max_tokens": 200,
      "gen_ai.request.temperature": 0.5,
      "gen_ai.request.top_p": 0.9,
      "gen_ai.response.id": "chatcmpl-CQF10eCkoFphJZJQkzzN6EkTD0AVF",
      "gen_ai.response.model": "gpt-4-0613",
      "gen_ai.usage.output_tokens": 52,
      "gen_ai.usage.input_tokens": 97,
      "gen_ai.response.finish_reasons": ["stop"],
      "gen_ai.input.messages":
        '[{"role":"user","parts":[{"type":"text","content":"Weather in Paris?"}]},{"role":"assistant","parts":[{"type":"tool_call","id":"1234","name":"get_weather","arguments":{"location":"Paris"}}]},{"role":"tool","parts":[{"type":"tool_call_response","id":"1234","response":"rainy, 57°F"}]}]',
      "gen_ai.output.messages":
        '[{"role":"assistant","parts":[{"type":"text","content":"The weather in Paris is currently rainy with a temperature of 57°F."}],"finish_reason":"stop"}]',
    });
    span.setStatus({ code: SpanStatusCode.OK });
    span.end();
  });

  console.log("flushing and shutting down");
  await provider.forceFlush();
  await provider.shutdown();
  console.log("view your span at", COLLECTOR_ENDPOINT);
};

main();
