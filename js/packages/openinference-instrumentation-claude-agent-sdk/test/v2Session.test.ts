import { SpanStatusCode } from "@opentelemetry/api";
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
});
