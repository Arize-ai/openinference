import { SpanStatusCode } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { ClaudeAgentSDKInstrumentation, _resetPatchState } from "../src/instrumentation";

describe("V1 query() wrapper", () => {
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

  function createMockModule(messages: unknown[]) {
    return {
      query: function ({
        prompt: _prompt,
        options: _options,
      }: {
        prompt: string;
        options?: Record<string, unknown>;
      }) {
        return {
          [Symbol.asyncIterator]() {
            let index = 0;
            return {
              async next() {
                if (index < messages.length) {
                  return { done: false, value: messages[index++] };
                }
                return { done: true, value: undefined };
              },
            };
          },
        };
      },
    };
  }

  it("should create an AGENT span for query()", async () => {
    const mockModule = createMockModule([
      {
        type: "system",
        subtype: "init",
        session_id: "sess-123",
        model: "claude-sonnet-4-20250514",
        tools: ["Bash", "Read"],
      },
      {
        type: "result",
        subtype: "success",
        result: "Hello, world!",
        stop_reason: "end_turn",
        usage: { input_tokens: 100, output_tokens: 50 },
        total_cost_usd: 0.005,
        num_turns: 1,
        duration_ms: 1234,
        session_id: "sess-123",
      },
    ]);

    instrumentation.manuallyInstrument(mockModule);

    const iterable = mockModule.query({ prompt: "Say hello" });
    const collected: unknown[] = [];
    for await (const msg of iterable) {
      collected.push(msg);
    }

    expect(collected).toHaveLength(2);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("ClaudeAgent.query");
    expect(span.status.code).toBe(SpanStatusCode.OK);

    const attrs = span.attributes;
    expect(attrs[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(OpenInferenceSpanKind.AGENT);
    expect(attrs[SemanticConventions.INPUT_VALUE]).toBe("Say hello");
    expect(attrs[SemanticConventions.OUTPUT_VALUE]).toBe("Hello, world!");
    expect(attrs[SemanticConventions.SESSION_ID]).toBe("sess-123");
    expect(attrs[SemanticConventions.LLM_MODEL_NAME]).toBe("claude-sonnet-4-20250514");
    expect(attrs[SemanticConventions.LLM_FINISH_REASON]).toBe("end_turn");
    expect(attrs[SemanticConventions.LLM_TOKEN_COUNT_PROMPT]).toBe(100);
    expect(attrs[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]).toBe(50);
    expect(attrs[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBe(150);
    expect(attrs[SemanticConventions.LLM_COST_TOTAL]).toBe(0.005);
  });

  it("should handle error result messages", async () => {
    const mockModule = createMockModule([
      {
        type: "system",
        subtype: "init",
        session_id: "sess-456",
        model: "claude-sonnet-4-20250514",
        tools: [],
      },
      {
        type: "result",
        subtype: "error_max_turns",
        errors: ["Max turns reached"],
        stop_reason: "max_turns",
        usage: { input_tokens: 200, output_tokens: 100 },
        total_cost_usd: 0.01,
        num_turns: 5,
        duration_ms: 5000,
        session_id: "sess-456",
      },
    ]);

    instrumentation.manuallyInstrument(mockModule);

    const iterable = mockModule.query({ prompt: "Do something complex" });
    const collected: unknown[] = [];
    for await (const msg of iterable) {
      collected.push(msg);
    }

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    // The span ends with OK from the generator completing normally,
    // but the error result message sets ERROR status which takes precedence
    expect(span.attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBe(300);
    expect(span.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe("claude-sonnet-4-20250514");
    expect(span.attributes[SemanticConventions.LLM_FINISH_REASON]).toBe("max_turns");
  });

  it("should read finish reason from older assistant message shapes", async () => {
    const mockModule = createMockModule([
      {
        type: "assistant",
        message: { stop_reason: "end_turn" },
        parent_tool_use_id: null,
        uuid: "assistant-uuid",
        session_id: "sess-older-v1",
      },
      {
        type: "result",
        subtype: "success",
        result: "Done",
        usage: { input_tokens: 10, output_tokens: 5 },
        total_cost_usd: 0.001,
        num_turns: 1,
        duration_ms: 100,
        session_id: "sess-older-v1",
      },
    ]);

    instrumentation.manuallyInstrument(mockModule);

    for await (const _msg of mockModule.query({ prompt: "test" })) {
      // consume
    }

    const spans = exporter.getFinishedSpans();
    expect(spans[0].attributes[SemanticConventions.LLM_FINISH_REASON]).toBe("end_turn");
  });

  it("should handle generator errors", async () => {
    const mockModule = {
      query: function () {
        return {
          [Symbol.asyncIterator]() {
            return {
              async next() {
                throw new Error("Connection lost");
              },
            };
          },
        };
      },
    };

    instrumentation.manuallyInstrument(mockModule);

    const iterable = mockModule.query({ prompt: "test", options: {} });

    await expect(async () => {
      for await (const _msg of iterable) {
        // consume
      }
    }).rejects.toThrow("Connection lost");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.status.code).toBe(SpanStatusCode.ERROR);
    expect(span.status.message).toBe("Connection lost");
  });

  it("should handle early generator termination (break)", async () => {
    const mockModule = createMockModule([
      {
        type: "system",
        subtype: "init",
        session_id: "sess-789",
        model: "claude-sonnet-4-20250514",
        tools: [],
      },
      { type: "assistant", content: "working..." },
      { type: "assistant", content: "more work..." },
      {
        type: "result",
        subtype: "success",
        result: "Done",
        usage: { input_tokens: 50, output_tokens: 25 },
        total_cost_usd: 0.001,
        num_turns: 1,
        duration_ms: 500,
        session_id: "sess-789",
      },
    ]);

    instrumentation.manuallyInstrument(mockModule);

    const iterable = mockModule.query({ prompt: "test" });
    let count = 0;
    for await (const _msg of iterable) {
      count++;
      if (count === 2) break;
    }

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.OK);
  });

  it("should pass through non-string prompts as JSON", async () => {
    const mockModule = createMockModule([
      {
        type: "result",
        subtype: "success",
        result: "ok",
        usage: { input_tokens: 10, output_tokens: 5 },
        total_cost_usd: 0.001,
        num_turns: 1,
        duration_ms: 100,
        session_id: "sess-abc",
      },
    ]);

    instrumentation.manuallyInstrument(mockModule);

    // AsyncIterable prompt (simulated as non-string)
    const asyncPrompt = {
      async *[Symbol.asyncIterator]() {
        yield { type: "user", content: "hello" };
      },
    };

    const iterable = mockModule.query({ prompt: asyncPrompt as unknown as string });
    for await (const _msg of iterable) {
      // consume
    }

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    // Non-string prompts are JSON-stringified
    expect(spans[0].attributes[SemanticConventions.INPUT_MIME_TYPE]).toBe("application/json");
  });

  it("should omit finish reason when stop_reason is null", async () => {
    const mockModule = createMockModule([
      {
        type: "result",
        subtype: "success",
        result: "Done, no stop reason reported",
        stop_reason: null,
        usage: { input_tokens: 10, output_tokens: 5 },
        total_cost_usd: 0.001,
        num_turns: 1,
        duration_ms: 100,
        session_id: "sess-null-stop",
      },
    ]);

    instrumentation.manuallyInstrument(mockModule);

    const iterable = mockModule.query({ prompt: "test" });
    for await (const _msg of iterable) {
      // consume
    }

    const spans = exporter.getFinishedSpans();
    expect(spans[0].attributes).not.toHaveProperty(SemanticConventions.LLM_FINISH_REASON);
  });
  it("returns the SDK Query object with its control methods intact (#3774)", async () => {
    // The real SDK returns a Query: an AsyncGenerator carrying control methods
    // such as interrupt() and setPermissionMode(). Mirror that shape.
    const calls: string[] = [];
    const messages = [
      {
        type: "system",
        subtype: "init",
        session_id: "sess-q",
        model: "claude-sonnet-4-20250514",
        tools: [],
      },
      {
        type: "result",
        subtype: "success",
        result: "done",
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 1 },
        total_cost_usd: 0.001,
        num_turns: 1,
        duration_ms: 1,
        session_id: "sess-q",
      },
    ];
    const mockModule = {
      query: function (_params: { prompt: string; options?: Record<string, unknown> }) {
        let index = 0;
        const query = {
          async next() {
            return index < messages.length
              ? { done: false, value: messages[index++] }
              : { done: true, value: undefined };
          },
          async return() {
            return { done: true, value: undefined };
          },
          async throw(error: unknown) {
            throw error;
          },
          [Symbol.asyncIterator]() {
            return query;
          },
          async interrupt() {
            calls.push("interrupt");
          },
          async setPermissionMode(mode: string) {
            calls.push(`setPermissionMode:${mode}`);
          },
        };
        return query;
      },
    };

    instrumentation.manuallyInstrument(mockModule);

    const q = mockModule.query({ prompt: "Say hello" });
    expect(typeof q.interrupt).toBe("function");
    expect(typeof q.setPermissionMode).toBe("function");

    await q.setPermissionMode("plan");
    for await (const _msg of q) {
      // consume
    }
    await q.interrupt();
    expect(calls).toEqual(["setPermissionMode:plan", "interrupt"]);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe("ClaudeAgent.query");
    expect(spans[0].status.code).toBe(SpanStatusCode.OK);
    expect(spans[0].attributes[SemanticConventions.OUTPUT_VALUE]).toBe("done");
  });

  it("traces iteration driven through next()/return() on the returned object (#3774)", async () => {
    const mockModule = createMockModule([
      {
        type: "system",
        subtype: "init",
        session_id: "sess-123",
        model: "claude-sonnet-4-20250514",
        tools: [],
      },
      {
        type: "result",
        subtype: "success",
        result: "Hello, world!",
        usage: { input_tokens: 1, output_tokens: 1 },
        total_cost_usd: 0.001,
        num_turns: 1,
        duration_ms: 1,
        session_id: "sess-123",
      },
    ]);

    instrumentation.manuallyInstrument(mockModule);

    // Query extends AsyncGenerator, so callers may drive it by hand.
    const q = mockModule.query({ prompt: "Say hello" }) as unknown as AsyncGenerator<unknown, void>;
    const first = await q.next();
    expect(first.done).toBe(false);
    await q.return(undefined);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe("ClaudeAgent.query");
    expect(spans[0].status.code).toBe(SpanStatusCode.OK);
  });

  it("calls the SDK query() when the wrapper is called, not when iteration begins (#3774)", () => {
    // Unwrapped, query() spawns the Claude Code process at call time, and the
    // Query object it returns has to exist for the wrapper to hand it back.
    let calls = 0;
    const mockModule = {
      query: (_params: { prompt: string; options?: Record<string, unknown> }) => {
        calls++;
        return {
          [Symbol.asyncIterator]() {
            return { next: async () => ({ done: true as const, value: undefined }) };
          },
        };
      },
    };

    instrumentation.manuallyInstrument(mockModule);

    mockModule.query({ prompt: "Say hello" });
    expect(calls).toBe(1);
  });
});
