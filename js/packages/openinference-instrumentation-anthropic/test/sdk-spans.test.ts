import Anthropic from "@anthropic-ai/sdk";
import type { BetaRunnableTool } from "@anthropic-ai/sdk/lib/tools/BetaRunnableTool";
import type { SpanContext } from "@opentelemetry/api";
import { context, ProxyTracerProvider, SpanStatusCode, trace } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { AnthropicInstrumentation } from "../src/instrumentation";
import type { StreamEvent } from "./helpers/sse";
import { toSSE } from "./helpers/sse";

/**
 * `@anthropic-ai/sdk` >= 0.134.0 starts a span of its own for each API call.
 * A call that the instrumentation traces has its LLM span instead, which the
 * SDK's request continues.
 */

const OI_SCOPE = "@arizeai/openinference-instrumentation-anthropic";
const SDK_SCOPE = "com.anthropic.sdk.typescript";
const MODEL = "claude-sonnet-4-6";
const PARAMS = {
  model: MODEL,
  max_tokens: 16,
  messages: [{ role: "user" as const, content: "hi" }],
};
const MESSAGE = {
  id: "msg_1",
  type: "message",
  role: "assistant",
  model: MODEL,
  content: [{ type: "text", text: "hi" }],
  stop_reason: "end_turn",
  stop_sequence: null,
  usage: { input_tokens: 10, output_tokens: 2 },
};
const EVENTS: StreamEvent[] = [
  {
    type: "message_start",
    message: {
      ...MESSAGE,
      content: [],
      stop_reason: null,
      usage: { input_tokens: 10, output_tokens: 0 },
    },
  },
  { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
  { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } },
  { type: "content_block_stop", index: 0 },
  {
    type: "message_delta",
    delta: { stop_reason: "end_turn", stop_sequence: null },
    usage: { output_tokens: 2 },
  },
  { type: "message_stop" },
];

const memoryExporter = new InMemorySpanExporter();
const tracerProvider = new NodeTracerProvider({
  spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
});
tracerProvider.register();

/**
 * Answers the Messages API, and records what was active and sent when each
 * request went.
 */
class Server {
  activeSpans: (SpanContext | undefined)[] = [];
  traceparents: (string | null)[] = [];

  fetch: typeof fetch = async (input, init) => {
    this.activeSpans.push(trace.getActiveSpan()?.spanContext());
    const url = input instanceof Request ? input.url : String(input);
    this.traceparents.push(new Headers(init?.headers).get("traceparent"));
    const json = (body: unknown, status = 200) =>
      new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" },
      });
    if (url.includes("/count_tokens")) {
      return json({ input_tokens: 10 });
    }
    if (url.includes("/models")) {
      return json({ data: [], has_more: false, first_id: null, last_id: null });
    }
    const body = JSON.parse(typeof init?.body === "string" ? init.body : "{}");
    if (body.model === "missing") {
      return json(
        { type: "error", error: { type: "not_found_error", message: "model: missing" } },
        404,
      );
    }
    if (body.stream) {
      return new Response(toSSE(EVENTS), {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });
    }
    return json(MESSAGE);
  };
}

function makeClient(server: Server, options: Partial<Anthropic["openTelemetry"]> | false = {}) {
  return new Anthropic({
    apiKey: "fake-api-key",
    maxRetries: 0,
    fetch: server.fetch,
    // The SDK's spans go to the tracer provider registered with the API by default.
    openTelemetry: options === false ? false : { tracerProvider, ...options },
  });
}

/** Lets the spans that end in promise continuations end. */
async function settle(): Promise<ReadableSpan[]> {
  await new Promise((resolve) => setTimeout(resolve, 50));
  return memoryExporter
    .getFinishedSpans()
    .filter((span) => span.instrumentationLibrary.name !== "test");
}

function scopes(spans: ReadableSpan[]): string[] {
  return spans.map((span) => span.instrumentationLibrary.name);
}

/** The request went while the LLM span was active, and carried its trace context. */
function expectRequestContinues(server: Server, span: ReadableSpan) {
  const { traceId, spanId, traceFlags } = span.spanContext();
  expect(server.activeSpans.map((active) => active?.spanId)).toEqual([spanId]);
  expect(server.traceparents).toEqual([
    `00-${traceId}-${spanId}-${traceFlags.toString(16).padStart(2, "0")}`,
  ]);
}

const CALLS: Record<string, (client: Anthropic) => Promise<unknown>> = {
  create: (client) => client.messages.create(PARAMS),
  "create with stream: true": async (client) => {
    for await (const _ of await client.messages.create({ ...PARAMS, stream: true })) {
      // read to the end
    }
  },
  "stream()": (client) => client.messages.stream(PARAMS).finalMessage(),
  "stream() with an event listener": (client) =>
    new Promise((resolve) => client.messages.stream(PARAMS).on("end", () => resolve(undefined))),
  parse: (client) => client.messages.parse(PARAMS),
  countTokens: (client) => client.messages.countTokens({ model: MODEL, messages: PARAMS.messages }),
  withResponse: (client) => client.messages.create(PARAMS).withResponse(),
  "beta create": (client) => client.beta.messages.create(PARAMS),
  "beta create with stream: true": async (client) => {
    for await (const _ of await client.beta.messages.create({ ...PARAMS, stream: true })) {
      // read to the end
    }
  },
  "beta stream()": (client) => client.beta.messages.stream(PARAMS).finalMessage(),
  "beta countTokens": (client) =>
    client.beta.messages.countTokens({ model: MODEL, messages: PARAMS.messages }),
};

/** Whether this SDK starts spans of its own, which it does from 0.134.0. */
const SDK_HAS_SPANS = "openTelemetry" in Anthropic.prototype;

function traceparentOf(span: { spanContext(): SpanContext }): string {
  const { traceId, spanId, traceFlags } = span.spanContext();
  return `00-${traceId}-${spanId}-${traceFlags.toString(16).padStart(2, "0")}`;
}

describe.skipIf(!SDK_HAS_SPANS)("AnthropicInstrumentation - the SDK's own spans", () => {
  const instrumentation = new AnthropicInstrumentation({ tracerProvider });
  instrumentation.disable();
  instrumentation._modules[0].moduleExports = Anthropic;
  let server: Server;

  beforeAll(() => {
    instrumentation.enable();
  });

  beforeEach(() => {
    memoryExporter.reset();
    server = new Server();
  });

  afterEach(() => {
    instrumentation.disable();
    instrumentation.enable();
  });

  for (const [name, call] of Object.entries(CALLS)) {
    it(`has one span for ${name}`, async () => {
      await call(makeClient(server));
      const spans = await settle();
      expect(scopes(spans)).toEqual([OI_SCOPE]);
      expectRequestContinues(server, spans[0]);
    });
  }

  for (const name of ["create", "stream()"]) {
    it(`nests the LLM span of ${name} under the caller's span`, async () => {
      const client = makeClient(server);
      const tracer = tracerProvider.getTracer("test");
      const parent = await tracer.startActiveSpan("parent", async (span) => {
        await CALLS[name](client);
        span.end();
        return span;
      });
      const spans = await settle();
      expect(scopes(spans)).toEqual([OI_SCOPE]);
      expect(spans[0].parentSpanId).toBe(parent.spanContext().spanId);
      expect(spans[0].spanContext().traceId).toBe(parent.spanContext().traceId);
      expectRequestContinues(server, spans[0]);
    });
  }

  it("keeps concurrent calls apart", async () => {
    const client = makeClient(server);
    const names = Object.keys(CALLS).filter((name) => !name.includes("listener"));
    await Promise.all([...names, ...names, ...names].map((name) => CALLS[name](client)));
    const spans = await settle();
    expect(scopes(spans)).toEqual(Array(names.length * 3).fill(OI_SCOPE));
    expect([...server.traceparents].sort()).toEqual(spans.map(traceparentOf).sort());
  });

  it("leaves a call span that the caller passed alone", async () => {
    const callerSpan = tracerProvider.getTracer("test").startSpan("caller's call span");
    const options = {
      __span: { span: callerSpan, name: "anthropic.messages.create", startTime: Date.now() },
    };
    await makeClient(server).messages.create(PARAMS, options);
    await makeClient(server).messages.stream(PARAMS, options).finalMessage();
    callerSpan.end();
    const spans = await settle();
    expect(scopes(spans)).toEqual([OI_SCOPE, OI_SCOPE]);
    expect(server.traceparents).toEqual([traceparentOf(callerSpan), traceparentOf(callerSpan)]);
  });

  for (const name of ["create", "stream()", "countTokens"]) {
    it(`keeps the SDK's span of ${name} when the LLM span has no valid context`, async () => {
      // A tracer provider that makes no spans, while the client's does.
      const noSpans = new AnthropicInstrumentation({ tracerProvider: new ProxyTracerProvider() });
      instrumentation.disable();
      noSpans.disable();
      noSpans._modules[0].moduleExports = Anthropic;
      noSpans.enable();
      try {
        await CALLS[name](makeClient(server));
      } finally {
        noSpans.disable();
      }
      const spans = await settle();
      expect(scopes(spans)).toEqual([SDK_SCOPE]);
      expect(server.traceparents).toEqual([traceparentOf(spans[0])]);
    });
  }

  it("has one span for a failed call", async () => {
    await expect(
      makeClient(server).messages.create({ ...PARAMS, model: "missing" }),
    ).rejects.toThrow();
    const spans = await settle();
    expect(scopes(spans)).toEqual([OI_SCOPE]);
    expect(spans[0].status.code).toBe(SpanStatusCode.ERROR);
  });

  it("keeps the SDK's span of a call it does not trace", async () => {
    await makeClient(server).models.list();
    expect(scopes(await settle())).toEqual([SDK_SCOPE]);
  });

  it("restores the SDK's span when disabled", async () => {
    instrumentation.disable();
    await makeClient(server).messages.stream(PARAMS).finalMessage();
    expect(scopes(await settle())).toEqual([SDK_SCOPE]);
  });

  it("traces a client whose spans are disabled", async () => {
    await makeClient(server, false).messages.stream(PARAMS).finalMessage();
    expect(scopes(await settle())).toEqual([OI_SCOPE]);
  });

  it("has one LLM span for each request of a tool runner", async () => {
    const responses = [
      {
        ...MESSAGE,
        content: [{ type: "tool_use", id: "toolu_1", name: "get_weather", input: {} }],
        stop_reason: "tool_use",
      },
      MESSAGE,
    ];
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: async () =>
        new Response(JSON.stringify(responses.shift() ?? MESSAGE), {
          headers: { "content-type": "application/json" },
        }),
      openTelemetry: { tracerProvider },
    });
    const tool: BetaRunnableTool<Record<string, never>> = {
      name: "get_weather",
      input_schema: { type: "object", properties: {} },
      parse: () => ({}),
      run: () => "sunny",
    };
    await client.beta.messages.toolRunner({ ...PARAMS, tools: [tool] });
    const spans = await settle();
    const names = spans.map((span) => span.name);
    // The SDK's spans of the run and of its tool are not model calls, and stay.
    expect(names.filter((name) => name === "Anthropic Messages")).toHaveLength(2);
    expect(names).not.toContain("anthropic.messages.create");
  });

  it("leaves a suppressed stream() call alone", async () => {
    await context.with(suppressTracing(context.active()), () =>
      makeClient(server).messages.stream(PARAMS).finalMessage(),
    );
    expect(await settle()).toEqual([]);
    expect(server.traceparents).toEqual([null]);
  });
});
