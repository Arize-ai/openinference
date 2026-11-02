import { context, SpanStatusCode, trace } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { ClaudeAgentSDKInstrumentation, _resetPatchState } from "../src/instrumentation";

describe("V2 unstable_v2_prompt() wrapper", () => {
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;
  let instrumentation: ClaudeAgentSDKInstrumentation;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    instrumentation = new ClaudeAgentSDKInstrumentation({
      tracerProvider: provider,
    });
  });

  afterEach(() => {
    instrumentation.disable();
    _resetPatchState();
    exporter.reset();
  });

  it("should create an AGENT span for prompt()", async () => {
    const mockModule = {
      query: () => ({
        [Symbol.asyncIterator]: () => ({ next: async () => ({ done: true, value: undefined }) }),
      }),
      unstable_v2_prompt: async (_message: string, _options: Record<string, unknown>) => {
        return {
          type: "result",
          subtype: "success",
          result: "Prompt response",
          stop_reason: "end_turn",
          usage: { input_tokens: 50, output_tokens: 30 },
          total_cost_usd: 0.003,
          num_turns: 1,
          duration_ms: 800,
          session_id: "sess-v2-1",
        };
      },
    };

    instrumentation.manuallyInstrument(mockModule);

    const result = await mockModule.unstable_v2_prompt("What is 2+2?", {
      model: "claude-sonnet-4-20250514",
    });

    expect(result).toBeDefined();
    expect((result as Record<string, unknown>).result).toBe("Prompt response");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("ClaudeAgent.prompt");
    expect(span.status.code).toBe(SpanStatusCode.OK);
    expect(span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.AGENT,
    );
    expect(span.attributes[SemanticConventions.INPUT_VALUE]).toBe("What is 2+2?");
    expect(span.attributes[SemanticConventions.OUTPUT_VALUE]).toBe("Prompt response");
    expect(span.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("claude-sonnet-4-20250514");
    expect(span.attributes[SemanticConventions.LLM_FINISH_REASON]).toBe("end_turn");
    expect(span.attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT]).toBe(50);
    expect(span.attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]).toBe(30);
    expect(span.attributes[SemanticConventions.LLM_COST_TOTAL]).toBe(0.003);
  });

  it("should handle prompt errors", async () => {
    const mockModule = {
      query: () => ({
        [Symbol.asyncIterator]: () => ({ next: async () => ({ done: true, value: undefined }) }),
      }),
      unstable_v2_prompt: async () => {
        throw new Error("API error");
      },
    };

    instrumentation.manuallyInstrument(mockModule);

    await expect(
      mockModule.unstable_v2_prompt("test", { model: "claude-sonnet-4-20250514" }),
    ).rejects.toThrow("API error");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.ERROR);
  });
});

describe("V2 session wrappers", () => {
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;
  let instrumentation: ClaudeAgentSDKInstrumentation;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    instrumentation = new ClaudeAgentSDKInstrumentation({
      tracerProvider: provider,
    });
  });

  afterEach(() => {
    instrumentation.disable();
    _resetPatchState();
    exporter.reset();
  });

  it("should create an AGENT span for send() + stream()", async () => {
    const messages = [
      {
        type: "system",
        subtype: "init",
        session_id: "sess-v2-session",
        model: "claude-sonnet-4-20250514",
        tools: ["Bash"],
      },
      {
        type: "result",
        subtype: "success",
        result: "Session response",
        stop_reason: "end_turn",
        usage: { input_tokens: 75, output_tokens: 40 },
        total_cost_usd: 0.004,
        num_turns: 1,
        duration_ms: 900,
        session_id: "sess-v2-session",
      },
    ];

    // Create async generator version of mock session
    const mockSession = {
      sessionId: "sess-mock",
      send: async (_msg: string) => {},
      stream: async function* () {
        for (const msg of messages) {
          yield msg;
        }
      },
      close: () => {},
    };

    const mockModule = {
      query: () => ({
        [Symbol.asyncIterator]: () => ({
          next: async () => ({ done: true, value: undefined }),
        }),
      }),
      unstable_v2_createSession: (_options: Record<string, unknown>) => mockSession,
    };

    instrumentation.manuallyInstrument(mockModule);

    const session = mockModule.unstable_v2_createSession({
      model: "claude-sonnet-4-20250514",
    });
    await session.send("Hello session");

    for await (const _msg of session.stream()) {
      // consume messages
    }

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("ClaudeAgent.turn");
    expect(span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.AGENT,
    );
    expect(span.attributes[SemanticConventions.INPUT_VALUE]).toBe("Hello session");
    expect(span.attributes[SemanticConventions.OUTPUT_VALUE]).toBe("Session response");
    expect(span.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("claude-sonnet-4-20250514");
    expect(span.attributes[SemanticConventions.LLM_FINISH_REASON]).toBe("end_turn");
  });

  it("should wrap resumeSession", async () => {
    const mockSession = {
      sessionId: "sess-resume",
      send: async (_msg: string) => {},
      stream: async function* () {
        yield {
          type: "result",
          subtype: "success",
          result: "Resumed",
          usage: { input_tokens: 10, output_tokens: 5 },
          total_cost_usd: 0.001,
          num_turns: 1,
          duration_ms: 100,
          session_id: "sess-resume",
        };
      },
      close: () => {},
    };

    const mockModule = {
      query: () => ({
        [Symbol.asyncIterator]: () => ({
          next: async () => ({ done: true, value: undefined }),
        }),
      }),
      unstable_v2_resumeSession: (_sessionId: string, _options: Record<string, unknown>) =>
        mockSession,
    };

    instrumentation.manuallyInstrument(mockModule);

    const session = mockModule.unstable_v2_resumeSession("sess-resume", {
      model: "claude-sonnet-4-20250514",
    });
    await session.send("Continue");

    for await (const _msg of session.stream()) {
      // consume
    }

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe("ClaudeAgent.turn");
  });

  it("should read finish reason from older assistant message shapes", async () => {
    const messages = [
      {
        type: "assistant",
        message: { stop_reason: "end_turn" },
        parent_tool_use_id: null,
        uuid: "assistant-uuid",
        session_id: "sess-older-v2",
      },
      {
        type: "result",
        subtype: "success",
        result: "Done",
        usage: { input_tokens: 10, output_tokens: 5 },
        total_cost_usd: 0.001,
        num_turns: 1,
        duration_ms: 100,
        session_id: "sess-older-v2",
      },
    ];
    const mockSession = {
      sessionId: "sess-older-v2",
      send: async (_msg: string) => {},
      stream: async function* () {
        for (const msg of messages) {
          yield msg;
        }
      },
      close: () => {},
    };
    const mockModule = {
      query: () => ({
        [Symbol.asyncIterator]: () => ({ next: async () => ({ done: true, value: undefined }) }),
      }),
      unstable_v2_createSession: (_options: Record<string, unknown>) => mockSession,
    };

    instrumentation.manuallyInstrument(mockModule);

    const session = mockModule.unstable_v2_createSession({});
    await session.send("test");
    for await (const _msg of session.stream()) {
      // consume
    }

    const spans = exporter.getFinishedSpans();
    expect(spans[0].attributes[SemanticConventions.LLM_FINISH_REASON]).toBe("end_turn");
  });

  it("should end span on session close()", async () => {
    const mockSession = {
      sessionId: "sess-close",
      send: async (_msg: string) => {},
      stream: async function* (): AsyncGenerator<unknown, void> {
        // never-ending stream
        while (true) {
          yield { type: "assistant", content: "thinking..." };
          await new Promise((r) => setTimeout(r, 10));
        }
      },
      close: () => {},
    };

    const mockModule = {
      query: () => ({
        [Symbol.asyncIterator]: () => ({
          next: async () => ({ done: true, value: undefined }),
        }),
      }),
      unstable_v2_createSession: (_options: Record<string, unknown>) => mockSession,
    };

    instrumentation.manuallyInstrument(mockModule);

    const session = mockModule.unstable_v2_createSession({
      model: "claude-sonnet-4-20250514",
    });
    await session.send("Hello");

    // Close without consuming stream
    session.close();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.OK);
  });

  it("should omit finish reason when stop_reason is null", async () => {
    const mockModule = {
      query: () => ({
        [Symbol.asyncIterator]: () => ({ next: async () => ({ done: true, value: undefined }) }),
      }),
      unstable_v2_prompt: async (_message: string, _options: Record<string, unknown>) => {
        return {
          type: "result",
          subtype: "success",
          result: "No stop reason",
          stop_reason: null,
          usage: { input_tokens: 10, output_tokens: 5 },
          total_cost_usd: 0.001,
          num_turns: 1,
          duration_ms: 100,
          session_id: "sess-v2-null-stop",
        };
      },
    };

    instrumentation.manuallyInstrument(mockModule);

    await mockModule.unstable_v2_prompt("test", { model: "claude-sonnet-4-20250514" });

    const spans = exporter.getFinishedSpans();
    expect(spans[0].attributes).not.toHaveProperty(SemanticConventions.LLM_FINISH_REASON);
  });
});

describe("V2 session stream context propagation", () => {
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;
  let instrumentation: ClaudeAgentSDKInstrumentation;

  const resultMessage = {
    type: "result",
    subtype: "success",
    result: "done",
    stop_reason: "end_turn",
    usage: { input_tokens: 1, output_tokens: 1 },
    total_cost_usd: 0,
    num_turns: 1,
    duration_ms: 1,
    session_id: "sess-context",
  };

  const tick = () => new Promise((resolve) => setTimeout(resolve, 1));

  // Instruments a fake SDK module whose session.stream() is the given async
  // generator, and returns an instrumented session created from it.
  function createInstrumentedSession(stream: () => AsyncGenerator<unknown, void>) {
    const mockSession = {
      sessionId: "sess-context",
      send: async (_msg: string) => {},
      stream,
      close: () => {},
    };
    const mockModule = {
      query: () => ({
        [Symbol.asyncIterator]: () => ({
          next: async () => ({ done: true, value: undefined }),
        }),
      }),
      unstable_v2_createSession: (_options: Record<string, unknown>) => mockSession,
    };
    instrumentation.manuallyInstrument(mockModule);
    return mockModule.unstable_v2_createSession({ model: "claude-sonnet-4-20250514" });
  }

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    // register() installs an AsyncLocalStorage context manager so that
    // context.with() and context.active() propagate across awaits.
    provider.register();
    instrumentation = new ClaudeAgentSDKInstrumentation({
      tracerProvider: provider,
    });
  });

  afterEach(() => {
    instrumentation.disable();
    _resetPatchState();
    exporter.reset();
    context.disable();
    trace.disable();
  });

  it("should run the SDK stream generator with ClaudeAgent.turn as the active span", async () => {
    const sdkTracer = provider.getTracer("sdk-internal");
    const session = createInstrumentedSession(async function* () {
      // Stands in for SDK work (e.g. an HTTP/transport span) that parents
      // itself on the active context while the stream is being iterated.
      await tick();
      const nestedSpan = sdkTracer.startSpan("sdk.nested-operation");
      nestedSpan.end();
      yield resultMessage;
    });

    const applicationSpan = provider.getTracer("application").startSpan("application.request");
    await context.with(trace.setSpan(context.active(), applicationSpan), async () => {
      await session.send("Hello");
      for await (const _msg of session.stream()) {
        // consume messages
      }
    });
    applicationSpan.end();

    const spans = exporter.getFinishedSpans();
    const turnSpan = spans.find((span) => span.name === "ClaudeAgent.turn");
    const nestedSpan = spans.find((span) => span.name === "sdk.nested-operation");
    expect(turnSpan).toBeDefined();
    expect(nestedSpan).toBeDefined();
    expect(turnSpan!.parentSpanId).toBe(applicationSpan.spanContext().spanId);
    expect(nestedSpan!.parentSpanId).toBe(turnSpan!.spanContext().spanId);
  });

  it("should keep the turn context inside the SDK stream and the consumer context between yields", async () => {
    const sdkActiveSpanIds: (string | undefined)[] = [];
    const consumerActiveSpanIds: (string | undefined)[] = [];
    const recordSdkActiveSpan = () =>
      sdkActiveSpanIds.push(trace.getActiveSpan()?.spanContext().spanId);

    const session = createInstrumentedSession(async function* () {
      recordSdkActiveSpan();
      await tick();
      recordSdkActiveSpan();
      yield { type: "assistant", message: { stop_reason: null } };
      await tick();
      recordSdkActiveSpan();
      yield resultMessage;
    });

    const applicationSpan = provider.getTracer("application").startSpan("application.request");
    await context.with(trace.setSpan(context.active(), applicationSpan), async () => {
      await session.send("Hello");
      for await (const _msg of session.stream()) {
        consumerActiveSpanIds.push(trace.getActiveSpan()?.spanContext().spanId);
        await tick();
        consumerActiveSpanIds.push(trace.getActiveSpan()?.spanContext().spanId);
      }
    });
    applicationSpan.end();

    const turnSpan = exporter.getFinishedSpans().find((span) => span.name === "ClaudeAgent.turn");
    expect(turnSpan).toBeDefined();
    expect(sdkActiveSpanIds).toEqual(Array(3).fill(turnSpan!.spanContext().spanId));
    expect(consumerActiveSpanIds).toEqual(Array(4).fill(applicationSpan.spanContext().spanId));
  });

  it("should run SDK cleanup in the turn context and end the turn span on early break", async () => {
    const sdkTracer = provider.getTracer("sdk-internal");
    const session = createInstrumentedSession(async function* () {
      try {
        yield { type: "assistant", message: { stop_reason: null } };
        yield resultMessage;
      } finally {
        const cleanupSpan = sdkTracer.startSpan("sdk.cleanup");
        cleanupSpan.end();
      }
    });

    const applicationSpan = provider.getTracer("application").startSpan("application.request");
    await context.with(trace.setSpan(context.active(), applicationSpan), async () => {
      await session.send("Hello");
      for await (const _msg of session.stream()) {
        break;
      }
      expect(trace.getActiveSpan()).toBe(applicationSpan);
    });
    applicationSpan.end();

    const spans = exporter.getFinishedSpans();
    const turnSpan = spans.find((span) => span.name === "ClaudeAgent.turn");
    const cleanupSpan = spans.find((span) => span.name === "sdk.cleanup");
    expect(turnSpan).toBeDefined();
    expect(cleanupSpan).toBeDefined();
    expect(turnSpan!.status.code).toBe(SpanStatusCode.OK);
    expect(cleanupSpan!.parentSpanId).toBe(turnSpan!.spanContext().spanId);
  });

  it("should parent SDK work on the turn span and record the error when the stream throws", async () => {
    const sdkTracer = provider.getTracer("sdk-internal");
    const session = createInstrumentedSession(async function* () {
      yield { type: "assistant", message: { stop_reason: null } };
      await tick();
      const nestedSpan = sdkTracer.startSpan("sdk.nested-operation");
      nestedSpan.end();
      throw new Error("stream failed");
    });

    const applicationSpan = provider.getTracer("application").startSpan("application.request");
    await context.with(trace.setSpan(context.active(), applicationSpan), async () => {
      await session.send("Hello");
      await expect(async () => {
        for await (const _msg of session.stream()) {
          // consume messages
        }
      }).rejects.toThrow("stream failed");
      expect(trace.getActiveSpan()).toBe(applicationSpan);
    });
    applicationSpan.end();

    const spans = exporter.getFinishedSpans();
    const turnSpan = spans.find((span) => span.name === "ClaudeAgent.turn");
    const nestedSpan = spans.find((span) => span.name === "sdk.nested-operation");
    expect(turnSpan).toBeDefined();
    expect(nestedSpan).toBeDefined();
    expect(turnSpan!.status.code).toBe(SpanStatusCode.ERROR);
    expect(turnSpan!.status.message).toBe("stream failed");
    expect(nestedSpan!.parentSpanId).toBe(turnSpan!.spanContext().spanId);
  });
});
