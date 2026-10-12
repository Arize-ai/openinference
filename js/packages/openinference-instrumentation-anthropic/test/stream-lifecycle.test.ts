import Anthropic from "@anthropic-ai/sdk";
import { SpanStatusCode } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";
import {
  failingSSEFetch,
  jsonFetch,
  messageDelta,
  messageStart,
  sseFetch,
  textStreamEvents,
  toSSE,
} from "./helpers/sse";
import { waitForSpans as waitForSpansOn } from "./helpers/waitForSpans";

const {
  INPUT_VALUE,
  LLM_INPUT_MESSAGES,
  LLM_OUTPUT_MESSAGES,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENT_TYPE,
  MESSAGE_CONTENTS,
  MESSAGE_TOOL_CALLS,
  TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
  TOOL_CALL_FUNCTION_NAME,
} = SemanticConventions;

const memoryExporter = new InMemorySpanExporter();
const waitForSpans = (count: number) => waitForSpansOn(memoryExporter, count);

const outputText = `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`;

const request = {
  model: "claude-sonnet-4-6",
  max_tokens: 100,
  messages: [{ role: "user" as const, content: "Hello" }],
};

function createClient(fetch: typeof globalThis.fetch) {
  return new Anthropic({ apiKey: "fake-api-key", maxRetries: 0, fetch });
}

describe("AnthropicInstrumentation - stream lifecycle", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new AnthropicInstrumentation({ tracerProvider });
  instrumentation.disable();
  instrumentation._modules[0].moduleExports = Anthropic;

  beforeAll(() => {
    instrumentation.enable();
  });

  beforeEach(() => {
    memoryExporter.reset();
  });

  afterEach(() => {
    instrumentation.disable();
    instrumentation.enable();
  });

  it("does not end the span before the caller reads the stream", async () => {
    const client = createClient(sseFetch(textStreamEvents()));
    const stream = await client.messages.create({ ...request, stream: true });

    await new Promise((resolve) => setTimeout(resolve, 20));
    // Nothing is read ahead of the caller.
    expect(memoryExporter.getFinishedSpans()).toHaveLength(0);

    for await (const _event of stream) {
      // drain
    }
    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    expect(span.status.code).toBe(SpanStatusCode.OK);
    expect(span.attributes[outputText]).toBe("Hello, world");
  });

  it("ends the span with partial output when the caller breaks, and aborts the request", async () => {
    const client = createClient(sseFetch(textStreamEvents(["one", " two", " three"])));
    const stream = await client.messages.create({ ...request, stream: true });

    let read = 0;
    for await (const _event of stream) {
      read++;
      if (read === 3) {
        // message_start, content_block_start, the first text delta
        break;
      }
    }

    expect(stream.controller.signal.aborted).toBe(true);
    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.UNSET);
    expect(spans[0].attributes[outputText]).toBe("one");
  });

  it("leaves the status unset when the caller aborts the stream", async () => {
    const client = createClient(sseFetch(textStreamEvents(["one", " two", " three"])));
    const stream = await client.messages.create({ ...request, stream: true });

    let read = 0;
    for await (const _event of stream) {
      read++;
      if (read === 3) {
        stream.controller.abort();
      }
    }

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.UNSET);
    // An in-memory body may still be decoded after the abort.
    expect(spans[0].attributes[outputText]).toEqual(expect.stringMatching(/^one/));
  });

  it("records a mid-stream error event on the span and still throws it to the caller", async () => {
    const events = textStreamEvents().slice(0, 3);
    const body =
      toSSE(events) +
      `event: error\ndata: ${JSON.stringify({
        type: "error",
        error: { type: "overloaded_error", message: "Overloaded" },
      })}\n\n`;
    const client = createClient(
      async () =>
        new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } }),
    );
    const stream = await client.messages.create({ ...request, stream: true });

    await expect(
      (async () => {
        for await (const _event of stream) {
          // drain
        }
      })(),
    ).rejects.toThrow();

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.ERROR);
    expect(spans[0].events.map((event) => event.name)).toContain("exception");
    expect(spans[0].attributes[outputText]).toBe("Hello");
  });

  it("records a dropped connection on the span without an unhandled rejection", async () => {
    const client = createClient(
      failingSSEFetch(textStreamEvents().slice(0, 3), new Error("socket hang up")),
    );
    const stream = await client.messages.create({ ...request, stream: true });

    await expect(
      (async () => {
        for await (const _event of stream) {
          // drain
        }
      })(),
    ).rejects.toThrow("socket hang up");

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.ERROR);
    expect(spans[0].status.message).toBe("socket hang up");
  });

  it("traces a stream read through toReadableStream()", async () => {
    const client = createClient(sseFetch(textStreamEvents()));
    const stream = await client.messages.create({ ...request, stream: true });

    const reader = stream.toReadableStream().getReader();
    while (!(await reader.read()).done) {
      // drain
    }

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.OK);
    expect(spans[0].attributes[outputText]).toBe("Hello, world");
  });

  it("traces a stream split with tee()", async () => {
    const client = createClient(sseFetch(textStreamEvents()));
    const stream = await client.messages.create({ ...request, stream: true });

    const [left, right] = stream.tee();
    const leftEvents = [];
    for await (const event of left) {
      leftEvents.push(event);
    }
    const rightEvents = [];
    for await (const event of right) {
      rightEvents.push(event);
    }
    expect(rightEvents).toEqual(leftEvents);

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes[outputText]).toBe("Hello, world");
  });

  it("ends the span when messages.stream() is aborted", async () => {
    const client = createClient(sseFetch(textStreamEvents(["one", " two", " three"])));
    const stream = client.messages.stream(request);

    let read = 0;
    await expect(
      (async () => {
        for await (const _event of stream) {
          read++;
          if (read === 3) {
            stream.abort();
          }
        }
      })(),
    ).rejects.toThrow("Request was aborted.");

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.UNSET);
  });

  it("appends tool arguments to the tool call of their own content block", async () => {
    const client = createClient(
      sseFetch([
        messageStart(),
        {
          type: "content_block_start",
          index: 0,
          content_block: { type: "tool_use", id: "toolu_1", name: "get_weather", input: {} },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '{"city":"Paris"}' },
        },
        { type: "content_block_stop", index: 0 },
        {
          type: "content_block_start",
          index: 1,
          content_block: {
            type: "server_tool_use",
            id: "srvtoolu_1",
            name: "web_search",
            input: {},
          },
        },
        {
          type: "content_block_delta",
          index: 1,
          delta: { type: "input_json_delta", partial_json: '{"query":"weather"}' },
        },
        { type: "content_block_stop", index: 1 },
        messageDelta({ stopReason: "tool_use" }),
        { type: "message_stop" },
      ]),
    );
    const stream = await client.messages.create({ ...request, stream: true });
    for await (const _event of stream) {
      // drain
    }

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    const toolCall = (index: number, postfix: string) =>
      span.attributes[`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_TOOL_CALLS}.${index}.${postfix}`];
    expect(toolCall(0, TOOL_CALL_FUNCTION_NAME)).toBe("get_weather");
    expect(toolCall(0, TOOL_CALL_FUNCTION_ARGUMENTS_JSON)).toBe('{"city":"Paris"}');
    expect(toolCall(-1, TOOL_CALL_FUNCTION_ARGUMENTS_JSON)).toBeUndefined();
  });

  it("never throws into the caller when a request cannot be turned into attributes", async () => {
    const client = createClient(
      jsonFetch({
        id: "msg_1",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-6",
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 1 },
      }),
    );

    const message = await client.beta.messages.create({
      ...request,
      messages: [
        // A fallback block echoed back without its `from` / `to` models.
        {
          role: "assistant",
          content: [{ type: "fallback" } as unknown as Anthropic.Beta.BetaFallbackBlockParam],
        },
      ],
    });
    expect(message.content).toHaveLength(1);

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    expect(span.status.code).toBe(SpanStatusCode.OK);
    expect(span.attributes[INPUT_VALUE]).toEqual(expect.any(String));
    expect(
      span.attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`],
    ).toBe("fallback");
  });

  it("parses a large streamed tool input once, not on every delta", async () => {
    const deltas = 20000;
    const client = createClient(
      sseFetch([
        messageStart(),
        {
          type: "content_block_start",
          index: 0,
          content_block: { type: "tool_use", id: "toolu_1", name: "write", input: {} },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '{"content": "' },
        },
        ...Array.from({ length: deltas }, () => ({
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: "abcdefghij" },
        })),
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '"}' },
        },
        { type: "content_block_stop", index: 0 },
        messageDelta({ stopReason: "tool_use" }),
        { type: "message_stop" },
      ]),
    );
    const started = Date.now();
    const stream = await client.messages.create({ ...request, stream: true });
    for await (const _event of stream) {
      // drain
    }
    // Parsing the input on every delta took seconds for this stream.
    expect(Date.now() - started).toBeLessThan(1000);

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    const toolArguments =
      span.attributes[
        `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_TOOL_CALLS}.0.${TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
      ];
    expect(toolArguments).toBe(JSON.stringify({ content: "abcdefghij".repeat(deltas) }));
  });

  it("keeps the raw input of a tool call the caller left before it stopped", async () => {
    const client = createClient(
      sseFetch([
        messageStart(),
        {
          type: "content_block_start",
          index: 0,
          content_block: { type: "tool_use", id: "toolu_1", name: "get_weather", input: {} },
        },
        {
          type: "content_block_delta",
          index: 0,
          delta: { type: "input_json_delta", partial_json: '{"city": "Par' },
        },
        { type: "content_block_stop", index: 0 },
        messageDelta({ stopReason: "tool_use" }),
        { type: "message_stop" },
      ]),
    );
    const stream = await client.messages.create({ ...request, stream: true });
    let read = 0;
    for await (const _event of stream) {
      if (++read === 3) {
        break;
      }
    }

    await waitForSpans(1);
    const [span] = memoryExporter.getFinishedSpans();
    expect(
      span.attributes[
        `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_TOOL_CALLS}.0.${TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
      ],
    ).toBe('{"city": "Par');
  });

  it("observes only the first iterator, so a second one's error does not end the span", async () => {
    const client = createClient(sseFetch(textStreamEvents()));
    const stream = await client.messages.create({ ...request, stream: true });

    const first = stream[Symbol.asyncIterator]();
    await first.next();
    await expect(
      (async () => {
        for await (const _event of stream) {
          // the response was already taken by the first iterator
        }
      })(),
    ).rejects.toThrow("Cannot iterate over a consumed stream");
    expect(memoryExporter.getFinishedSpans()).toHaveLength(0);

    while (!(await first.next()).done) {
      // drain
    }

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.OK);
    expect(spans[0].attributes[outputText]).toBe("Hello, world");
  });
});
