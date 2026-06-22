/* eslint-disable no-console */

/**
 * verify-spans.ts — runnable script that verifies the Eve integration end-to-end.
 *
 * Simulates the span hierarchy Eve produces for a single-turn, two-step
 * agentic run (streamText → toolCall → streamText), then prints the resulting
 * OpenInference attributes to the console so you can confirm the mapping.
 *
 * Does NOT require the Eve SDK or an LLM API key.
 *
 * Run from packages/openinference-eve:
 *   pnpx tsx examples/verify-spans.ts
 */

import { context, SpanStatusCode, trace } from "@opentelemetry/api";
import { ConsoleSpanExporter } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { Resource } from "@opentelemetry/resources";

import { isOpenInferenceSpan, OpenInferenceSimpleSpanProcessor } from "../src";

// ---------------------------------------------------------------------------
// OTel setup (mirrors what Eve's registerOTel call does under the hood)
// ---------------------------------------------------------------------------

const provider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]: "openinference-eve-verify",
  }),
  spanProcessors: [
    new OpenInferenceSimpleSpanProcessor({
      exporter: new ConsoleSpanExporter(),
      spanFilter: isOpenInferenceSpan,
    }),
  ],
});

provider.register();

const tracer = trace.getTracer("eve-verify");

// ---------------------------------------------------------------------------
// Simulate the span hierarchy Eve produces for a weather-agent turn
//
//   ai.eve.turn                         (AGENT, session.id extracted)
//   └── ai.streamText step-0            (AGENT — Vercel AI SDK)
//       └── ai.streamText.doStream      (LLM   — model call)
//       └── ai.toolCall get_weather     (TOOL  — tool execution)
//   └── ai.streamText step-1            (AGENT — second step after tool result)
//       └── ai.streamText.doStream      (LLM   — final answer)
// ---------------------------------------------------------------------------

async function main() {
  console.log("\n=== Eve OpenInference span verification ===\n");
  console.log(
    "Each exported span is printed below. Check that openinference.span.kind,",
  );
  console.log("session.id, and metadata.* are populated correctly.\n");

  // Root turn span — Eve creates one of these per conversational turn.
  const turnSpan = tracer.startSpan("ai.eve.turn", {
    attributes: {
      "operation.name": "ai.eve.turn",
      // Eve runtime context attributes injected automatically
      "eve.session.id": "sess_demo_001",
      "eve.version": "1.0.0",
      "eve.environment": "development",
      "eve.turn.id": "turn_0",
      "eve.turn.sequence": 0,
      "eve.channel.kind": "channel:terminal",
    },
  });

  const turnCtx = trace.setSpan(context.active(), turnSpan);

  // --- Step 0: first streamText (decides to call the weather tool) ----------

  const step0Span = tracer.startSpan(
    "ai.streamText",
    {
      attributes: {
        "operation.name": "ai.streamText weather-agent",
        "ai.operationId": "ai.streamText",
        "eve.session.id": "sess_demo_001",
        "eve.step.index": 0,
        "ai.model.id": "gpt-4o-mini",
        "ai.model.provider": "openai",
        "ai.prompt": JSON.stringify({
          system: "You are a helpful weather assistant.",
          messages: [{ role: "user", content: "What is the weather in Boston?" }],
        }),
        "ai.telemetry.metadata.example": "eve-verify",
      },
    },
    turnCtx,
  );

  const step0Ctx = trace.setSpan(turnCtx, step0Span);

  // The doStream span carries gen_ai.* attributes (populated by the AI SDK).
  const llm0Span = tracer.startSpan(
    "ai.streamText.doStream",
    {
      attributes: {
        "operation.name": "ai.streamText.doStream weather-agent",
        "ai.operationId": "ai.streamText.doStream",
        "eve.session.id": "sess_demo_001",
        "eve.step.index": 0,
        "gen_ai.operation.name": "chat",
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.response.model": "gpt-4o-mini-2024-07-18",
        "gen_ai.usage.input_tokens": 28,
        "gen_ai.usage.output_tokens": 12,
        "gen_gi.response.finish_reasons": ["tool_calls"],
        "ai.prompt.messages": JSON.stringify([
          { role: "system", content: "You are a helpful weather assistant." },
          { role: "user", content: "What is the weather in Boston?" },
        ]),
        "ai.response.toolCalls": JSON.stringify([
          {
            toolCallId: "call_abc123",
            toolName: "get_weather",
            args: { city: "Boston" },
          },
        ]),
        "ai.usage.promptTokens": 28,
        "ai.usage.completionTokens": 12,
      },
    },
    step0Ctx,
  );
  llm0Span.setStatus({ code: SpanStatusCode.OK });
  llm0Span.end();

  // toolCall span — one per tool the model invokes.
  const toolSpan = tracer.startSpan(
    "ai.toolCall",
    {
      attributes: {
        "operation.name": "ai.toolCall",
        "ai.operationId": "ai.toolCall",
        "eve.session.id": "sess_demo_001",
        "eve.step.index": 0,
        "ai.toolCall.id": "call_abc123",
        "ai.toolCall.name": "get_weather",
        "ai.toolCall.args": JSON.stringify({ city: "Boston" }),
        "ai.toolCall.result": JSON.stringify({ city: "Boston", condition: "Sunny", tempF: 72 }),
      },
    },
    step0Ctx,
  );
  toolSpan.setStatus({ code: SpanStatusCode.OK });
  toolSpan.end();

  step0Span.setStatus({ code: SpanStatusCode.OK });
  step0Span.end();

  // --- Step 1: second streamText (produces the final answer) ----------------

  const step1Span = tracer.startSpan(
    "ai.streamText",
    {
      attributes: {
        "operation.name": "ai.streamText weather-agent",
        "ai.operationId": "ai.streamText",
        "eve.session.id": "sess_demo_001",
        "eve.step.index": 1,
        "ai.model.id": "gpt-4o-mini",
        "ai.model.provider": "openai",
      },
    },
    turnCtx,
  );

  const step1Ctx = trace.setSpan(turnCtx, step1Span);

  const llm1Span = tracer.startSpan(
    "ai.streamText.doStream",
    {
      attributes: {
        "operation.name": "ai.streamText.doStream weather-agent",
        "ai.operationId": "ai.streamText.doStream",
        "eve.session.id": "sess_demo_001",
        "eve.step.index": 1,
        "gen_ai.operation.name": "chat",
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.response.model": "gpt-4o-mini-2024-07-18",
        "gen_ai.usage.input_tokens": 45,
        "gen_ai.usage.output_tokens": 18,
        "ai.response.text": "The weather in Boston is currently sunny with a temperature of 72°F.",
        "ai.response.finishReason": "stop",
        "ai.usage.promptTokens": 45,
        "ai.usage.completionTokens": 18,
      },
    },
    step1Ctx,
  );
  llm1Span.setStatus({ code: SpanStatusCode.OK });
  llm1Span.end();

  step1Span.setStatus({ code: SpanStatusCode.OK });
  step1Span.end();

  turnSpan.setStatus({ code: SpanStatusCode.OK });
  turnSpan.end();

  // Give the processor time to export before the process exits.
  await provider.forceFlush();
  await provider.shutdown();

  console.log("\n=== Done ===\n");
  console.log("Expected span kinds:");
  console.log("  ai.eve.turn             → AGENT  (with session.id = sess_demo_001)");
  console.log("  ai.streamText           → AGENT  (per step)");
  console.log("  ai.streamText.doStream  → LLM    (model call, token counts, messages)");
  console.log("  ai.toolCall             → TOOL   (tool name, args, result)");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
