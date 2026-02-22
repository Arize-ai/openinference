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
});
