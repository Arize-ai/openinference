import type { Ai } from "@cloudflare/workers-types";
import { context, ROOT_CONTEXT } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  TracerProvider,
} from "@opentelemetry/sdk-trace";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { OITracer, setSession } from "@arizeai/openinference-core";

import { AsyncLocalStorageContextManager, instrumentAi } from "../src";

const model = "@cf/meta/llama-3.2-1b-instruct";
const input = { messages: [{ role: "user", content: "secret" }] };
const output = {
  response: "hello",
  usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
};
function setup(mask = false, limit = 65536) {
  const exporter = new InMemorySpanExporter();
  const provider = new TracerProvider({ spanProcessors: [new SimpleSpanProcessor({ exporter })] });
  const tracer = new OITracer({
    tracer: provider.getTracer("test"),
    traceConfig: { hideInputs: mask, hideOutputs: mask },
  });
  const pending: Promise<unknown>[] = [];
  const run = vi.fn<(...args: unknown[]) => Promise<unknown>>();
  const original: Ai["run"] = () => {
    throw new Error("Use fixture proxy");
  };
  const method = new Proxy(original, {
    apply(_target, receiver, args: unknown[]) {
      return Reflect.apply(run, receiver, args);
    },
  });
  const binding = {
    run: method,
    marker: "binding",
    receiver() {
      return this.marker;
    },
  };
  const flush = vi.fn(() => provider.forceFlush());
  const ai = instrumentAi({
    ai: binding,
    models: [model],
    tracer,
    execution: {
      waitUntil(p) {
        pending.push(p);
      },
    },
    flush,
    maxStreamCaptureLength: limit,
  });
  return { ai, binding, run, exporter, flush, pending, tracer };
}
function bytes(text: string) {
  return new TextEncoder().encode(text);
}
const sse = (value: unknown) => `data: ${JSON.stringify(value)}\n\n`;
function stream(text: string) {
  let sent = false;
  return new ReadableStream<Uint8Array>(
    {
      pull(c) {
        if (!sent) {
          sent = true;
          c.enqueue(bytes(text));
        } else c.close();
      },
    },
    { highWaterMark: 0 },
  );
}
function withResult(t: ReturnType<typeof setup>, result: unknown) {
  t.run.mockResolvedValue(result);
}

beforeEach(() => context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable()));
afterEach(() => {
  context.disable();
});

describe("Cloudflare AI binding", () => {
  it("preserves types, receivers, results, and LLM context", async () => {
    const t = setup();
    withResult(t, output);
    const result = await context.with(setSession(ROOT_CONTEXT, { sessionId: "session" }), () =>
      t.ai.run(model, input),
    );
    await Promise.all(t.pending);
    expect(result).toBe(output);
    expect(t.ai.receiver()).toBe("binding");
    expect(t.run.mock.instances[0]).toBe(t.binding);
    const [span] = t.exporter.getFinishedSpans();
    expect(span.attributes).toMatchObject({
      "openinference.span.kind": "LLM",
      "session.id": "session",
      "llm.model_name": model,
      "llm.input_messages.0.message.content": "secret",
      "llm.output_messages.0.message.content": "hello",
      "llm.token_count.total": 3,
    });
  });
  it("suppresses calls and bypasses unselected model tasks", async () => {
    const t = setup();
    withResult(t, output);
    await context.with(suppressTracing(ROOT_CONTEXT), () => t.ai.run(model, input));
    await t.ai.run("image-model", { prompt: "image" });
    expect(t.run).toHaveBeenCalledTimes(2);
    expect(t.exporter.getFinishedSpans()).toHaveLength(0);
    expect(t.pending).toHaveLength(0);
  });
  it("masks ordinary and streamed messages", async () => {
    for (const streaming of [false, true]) {
      const t = setup(true);
      withResult(t, streaming ? stream(sse(output) + "data: [DONE]\n\n") : output);
      const result = await t.ai.run(model, { ...input, stream: streaming });
      if (result instanceof ReadableStream) await new Response(result).text();
      await Promise.all(t.pending);
      const [span] = t.exporter.getFinishedSpans();
      expect(span.attributes["input.value"]).toBe("__REDACTED__");
      expect(span.attributes["output.value"]).toBe("__REDACTED__");
      expect(
        Object.keys(span.attributes).some(
          (k) => k.startsWith("llm.input_messages") || k.startsWith("llm.output_messages"),
        ),
      ).toBe(false);
    }
  });
  it("waits for consumption, preserves bytes, and uses final usage without transport padding", async () => {
    const t = setup();
    const payload =
      sse({
        response: "hel",
        p: "padding",
        usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
      }) +
      sse({ response: "lo", usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 } }) +
      "data: [DONE]\n\n";
    withResult(t, stream(payload));
    const result = await t.ai.run(model, { ...input, stream: true });
    expect(t.exporter.getFinishedSpans()).toHaveLength(0);
    expect(t.flush).not.toHaveBeenCalled();
    expect(await new Response(result).text()).toBe(payload);
    await Promise.all(t.pending);
    const [span] = t.exporter.getFinishedSpans();
    expect(span.attributes["llm.output_messages.0.message.content"]).toBe("hello");
    expect(span.attributes["llm.token_count.total"]).toBe(4);
    expect(span.attributes["output.value"]).not.toContain("padding");
    expect(t.flush).toHaveBeenCalledTimes(1);
  });
  it("records cancellation and flushes without reading ahead", async () => {
    const t = setup();
    const cancel = vi.fn();
    const pull = vi.fn();
    withResult(t, new ReadableStream({ pull, cancel }, { highWaterMark: 0 }));
    const result = await t.ai.run(model, { ...input, stream: true });
    expect(pull).not.toHaveBeenCalled();
    await result.cancel("private reason");
    await Promise.all(t.pending);
    expect(cancel).toHaveBeenCalledWith("private reason");
    expect(t.exporter.getFinishedSpans()[0].status.code).toBe(2);
    expect(JSON.stringify(t.exporter.getFinishedSpans()[0].events)).not.toContain("private reason");
  });
  it("bounds capture while forwarding the entire stream", async () => {
    const t = setup(false, 4);
    const payload = sse(output);
    withResult(t, stream(payload));
    const result = await t.ai.run(model, { ...input, stream: true });
    expect(await new Response(result).text()).toBe(payload);
    await Promise.all(t.pending);
    expect(t.exporter.getFinishedSpans()[0].attributes).toMatchObject({
      "cloudflare.ai.output_omitted": true,
    });
    expect(t.exporter.getFinishedSpans()[0].attributes["output.value"]).toBeUndefined();
  });
  it("preserves provider errors without recording their sensitive body", async () => {
    const t = setup(true);
    const error = new Error("private provider body");
    t.run.mockRejectedValue(error);
    await expect(t.ai.run(model, input)).rejects.toBe(error);
    await Promise.all(t.pending);
    const [span] = t.exporter.getFinishedSpans();
    expect(span.status.code).toBe(2);
    expect(JSON.stringify(span.events)).not.toContain(error.message);
  });
  it("contains export failures", async () => {
    const t = setup();
    withResult(t, output);
    t.flush.mockRejectedValue(new Error("export"));
    expect(await t.ai.run(model, input)).toBe(output);
    await expect(Promise.all(t.pending)).resolves.toBeDefined();
  });
  it("assembles fragmented tool-call deltas and UTF-8 without changing bytes", async () => {
    const t = setup();
    const payload =
      sse({
        choices: [
          {
            index: 0,
            delta: {
              content: "café",
              tool_calls: [
                { index: 0, id: "call-1", function: { name: "weather", arguments: '{"city":' } },
              ],
            },
          },
        ],
      }) +
      sse({
        choices: [
          { index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '"Boston"}' } }] } },
        ],
      }) +
      "data: [DONE]\n\n";
    const data = bytes(payload);
    let offset = 0;
    withResult(
      t,
      new ReadableStream<Uint8Array>(
        {
          pull(c) {
            if (offset === data.length) c.close();
            else c.enqueue(data.slice(offset, ++offset));
          },
        },
        { highWaterMark: 0 },
      ),
    );
    const result = await t.ai.run(model, { ...input, stream: true });
    expect(await new Response(result).text()).toBe(payload);
    await Promise.all(t.pending);
    expect(t.exporter.getFinishedSpans()[0].attributes).toMatchObject({
      "llm.output_messages.0.message.content": "café",
      "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call-1",
      "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "weather",
      "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments":
        '{"city":"Boston"}',
    });
  });
  it("preserves raw response identity and skips unsupported response modes", async () => {
    const t = setup();
    const response = new Response("raw");
    withResult(t, response);
    expect(await t.ai.run(model, input, { returnRawResponse: true })).toBe(response);
    expect(t.exporter.getFinishedSpans()).toHaveLength(0);
    expect(t.pending).toHaveLength(0);
  });
  it("records stream read failures and propagates the original error", async () => {
    const t = setup();
    const error = new Error("private stream failure");
    withResult(
      t,
      new ReadableStream(
        {
          pull(c) {
            c.error(error);
          },
        },
        { highWaterMark: 0 },
      ),
    );
    const result = await t.ai.run(model, { ...input, stream: true });
    await expect(new Response(result).text()).rejects.toBe(error);
    await Promise.all(t.pending);
    expect(t.exporter.getFinishedSpans()[0].status.code).toBe(2);
  });
  it("recognizes provider error events without recording their payload", async () => {
    const t = setup(true);
    const payload = sse({ error: "private error" });
    withResult(t, stream(payload));
    const result = await t.ai.run(model, { ...input, stream: true });
    expect(await new Response(result).text()).toBe(payload);
    await Promise.all(t.pending);
    const [span] = t.exporter.getFinishedSpans();
    expect(span.status.code).toBe(2);
    expect(span.attributes["output.value"]).toBeUndefined();
    expect(JSON.stringify(span.events)).not.toContain("private error");
  });
});
