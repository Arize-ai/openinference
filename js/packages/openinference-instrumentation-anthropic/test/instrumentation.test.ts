import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import Anthropic, { APIPromise } from "@anthropic-ai/sdk";
import { Stream } from "@anthropic-ai/sdk/streaming";
import type {
  Message,
  RawMessageStreamEvent,
} from "@anthropic-ai/sdk/resources/messages/messages";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { AnthropicInstrumentation } from "../src/instrumentation";

describe("AnthropicInstrumentation", () => {
  let instrumentation: AnthropicInstrumentation;
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    instrumentation = new AnthropicInstrumentation({
      tracerProvider: provider,
    });
  });

  afterEach(() => {
    if (instrumentation.isEnabled()) {
      instrumentation.disable();
    }
    exporter.reset();
  });

  it("should be enabled by default", () => {
    expect(instrumentation.isEnabled()).toBe(true);
  });

  it("should be able to enable and disable", () => {
    expect(instrumentation.isEnabled()).toBe(true);

    instrumentation.disable();
    expect(instrumentation.isEnabled()).toBe(false);

    instrumentation.enable();
    expect(instrumentation.isEnabled()).toBe(true);
  });

  it("should initialize without errors", () => {
    expect(() => {
      new AnthropicInstrumentation();
    }).not.toThrow();
  });

  it("should accept custom tracer provider", () => {
    const customProvider = new NodeTracerProvider();
    const customInstrumentation = new AnthropicInstrumentation({
      tracerProvider: customProvider,
    });

    expect(customInstrumentation.isEnabled()).toBe(true);
    customInstrumentation.disable();
  });

  it("should accept trace config options", () => {
    const customInstrumentation = new AnthropicInstrumentation({
      traceConfig: {
        hideInputs: true,
        hideOutputs: true,
      },
    });

    expect(customInstrumentation.isEnabled()).toBe(true);
    customInstrumentation.disable();
  });
});

/**
 * Regression coverage for https://github.com/Arize-ai/openinference/issues/<TBD>:
 *
 * The Anthropic SDK's `messages.create` returns an `APIPromise<T>` (a `Promise`
 * subclass with `.withResponse()`, `.asResponse()`, `_thenUnwrap(...)`).
 * `messages.stream(...)` internally relies on `messages.create(...).withResponse()`.
 *
 * The current instrumentation wraps the return value with `.then(...).catch(...)`,
 * which collapses the `APIPromise` to a plain `Promise` and removes those helpers.
 * These tests pin the SDK contract that the patched `create` must preserve.
 */
describe("AnthropicInstrumentation - APIPromise contract", () => {
  process.env.ANTHROPIC_API_KEY = "fake-api-key";

  const memoryExporter = new InMemorySpanExporter();
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();

  const instrumentation = new AnthropicInstrumentation();
  instrumentation.disable();
  instrumentation.setTracerProvider(tracerProvider);
  // @ts-expect-error _modules is private; needed to defeat auto-mocking
  instrumentation._modules[0].moduleExports = Anthropic;

  let client: Anthropic;

  beforeAll(() => {
    instrumentation.enable();
    client = new Anthropic({ apiKey: "fake-api-key" });
  });

  afterAll(() => {
    instrumentation.disable();
  });

  beforeEach(() => {
    memoryExporter.reset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  /**
   * Build a Response stub that satisfies the fields read by the SDK
   * (`headers.get`, `status`, `url`) without doing any real I/O.
   */
  function fakeResponse(contentType: string): Response {
    return {
      headers: new Headers({
        "content-type": contentType,
        "request-id": "req-test",
      }),
      status: 200,
      statusText: "OK",
      ok: true,
      url: "https://api.anthropic.com/v1/messages",
    } as unknown as Response;
  }

  /**
   * Build a synthetic non-streaming Anthropic message response wrapped in an
   * `APIPromise`. Uses a custom `parseResponse` so we skip `response.json()`.
   */
  function mockMessageResponse(message: Message): APIPromise<Message> {
    return new APIPromise(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      client as any,
      Promise.resolve({
        requestLogID: "log-test",
        retryOfRequestLogID: undefined,
        startTime: Date.now(),
        response: fakeResponse("application/json"),
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        options: { method: "post", path: "/v1/messages" } as any,
        controller: new AbortController(),
      }),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      async () => message as any,
    );
  }

  /**
   * Build a synthetic streaming Anthropic response: an `APIPromise` whose
   * parsed value is a `Stream<RawMessageStreamEvent>`. Uses a custom
   * `parseResponse` so we never touch SSE wire format.
   */
  function mockStreamResponse(events: RawMessageStreamEvent[]): APIPromise<
    Stream<RawMessageStreamEvent>
  > {
    const controller = new AbortController();
    const iterator = () =>
      (async function* () {
        for (const event of events) {
          yield event;
        }
      })();

    return new APIPromise(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      client as any,
      Promise.resolve({
        requestLogID: "log-test",
        retryOfRequestLogID: undefined,
        startTime: Date.now(),
        response: fakeResponse("text/event-stream"),
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        options: { method: "post", path: "/v1/messages", stream: true } as any,
        controller,
      }),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      async () => new Stream(iterator, controller) as any,
    );
  }

  const SAMPLE_MESSAGE: Message = {
    id: "msg_test",
    type: "message",
    role: "assistant",
    model: "claude-test",
    content: [{ type: "text", text: "Hello world", citations: null }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: {
      input_tokens: 5,
      output_tokens: 2,
      cache_creation_input_tokens: null,
      cache_read_input_tokens: null,
      server_tool_use: null,
      service_tier: null,
    },
  };

  const STREAM_EVENTS: RawMessageStreamEvent[] = [
    {
      type: "message_start",
      message: {
        id: "msg_test",
        type: "message",
        role: "assistant",
        model: "claude-test",
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 5,
          output_tokens: 0,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
          server_tool_use: null,
          service_tier: null,
        },
      },
    },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "", citations: null },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "Hello " },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "world" },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: {
        input_tokens: 5,
        output_tokens: 2,
        cache_creation_input_tokens: null,
        cache_read_input_tokens: null,
        server_tool_use: null,
        service_tier: null,
      },
    },
    { type: "message_stop" },
  ];

  it("preserves APIPromise helpers on the patched messages.create return value", async () => {
    vi.spyOn(client, "post").mockImplementation(
      () => mockMessageResponse(SAMPLE_MESSAGE) as ReturnType<typeof client.post>,
    );

    const result = client.messages.create({
      model: "claude-test",
      max_tokens: 16,
      messages: [{ role: "user", content: "hello" }],
    });

    // These assertions fail today because the wrapper returns a plain Promise.
    expect(result).toBeInstanceOf(APIPromise);
    expect(typeof (result as unknown as APIPromise<Message>).withResponse).toBe("function");
    expect(typeof (result as unknown as APIPromise<Message>).asResponse).toBe("function");

    // Drain the promise so the span ends and the test doesn't leak.
    await result;
  });

  it("supports messages.stream(...).finalMessage() under instrumentation", async () => {
    vi.spyOn(client, "post").mockImplementation(
      () => mockStreamResponse(STREAM_EVENTS) as ReturnType<typeof client.post>,
    );

    // Today this throws `messages.create(...).withResponse is not a function`
    // from MessageStream._createMessage.
    const stream = client.messages.stream({
      model: "claude-test",
      max_tokens: 16,
      messages: [{ role: "user", content: "hello" }],
    });

    const finalMessage = await stream.finalMessage();
    expect(finalMessage.content[0]).toMatchObject({ type: "text", text: "Hello world" });
  });

  it("still records output deltas when streaming via messages.create({ stream: true })", async () => {
    vi.spyOn(client, "post").mockImplementation(
      () => mockStreamResponse(STREAM_EVENTS) as ReturnType<typeof client.post>,
    );

    const stream = await client.messages.create({
      model: "claude-test",
      max_tokens: 16,
      stream: true,
      messages: [{ role: "user", content: "hello" }],
    });

    // Drain the user-facing branch of the tee so iteration completes.
    for await (const _event of stream) {
      void _event;
    }

    // The instrumentation consumes the other tee branch async; wait for it.
    await new Promise((resolve) => setImmediate(resolve));

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("Anthropic Messages");
    expect(span.attributes["output.value"]).toBe("Hello world");
    expect(span.attributes["llm.output_messages.0.message.content"]).toBe("Hello world");
    expect(span.attributes["llm.output_messages.0.message.role"]).toBe("assistant");
  });

  it("records expected attributes for non-streaming messages.create", async () => {
    vi.spyOn(client, "post").mockImplementation(
      () => mockMessageResponse(SAMPLE_MESSAGE) as ReturnType<typeof client.post>,
    );

    await client.messages.create({
      model: "claude-test",
      max_tokens: 16,
      messages: [{ role: "user", content: "hello" }],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("Anthropic Messages");
    expect(span.attributes["openinference.span.kind"]).toBe("LLM");
    expect(span.attributes["llm.model_name"]).toBe("claude-test");
    expect(span.attributes["llm.system"]).toBe("anthropic");
    expect(span.attributes["llm.provider"]).toBe("anthropic");
    expect(span.attributes["llm.input_messages.0.message.role"]).toBe("user");
    expect(span.attributes["llm.input_messages.0.message.content"]).toBe("hello");
    expect(span.attributes["llm.output_messages.0.message.role"]).toBe("assistant");
    expect(span.attributes["llm.output_messages.0.message.contents.0.message_content.type"]).toBe(
      "text",
    );
    expect(span.attributes["llm.output_messages.0.message.contents.0.message_content.text"]).toBe(
      "Hello world",
    );
    expect(span.attributes["llm.token_count.prompt"]).toBe(5);
    expect(span.attributes["llm.token_count.completion"]).toBe(2);
    expect(span.attributes["llm.token_count.total"]).toBe(7);
  });
});
