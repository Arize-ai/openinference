import Anthropic from "@anthropic-ai/sdk";
import { SpanStatusCode } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import {
  LLMProvider,
  LLMSystem,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";
import { vcrFetch } from "./helpers/vcr";

const {
  OPENINFERENCE_SPAN_KIND,
  LLM_PROVIDER,
  LLM_SYSTEM,
  LLM_MODEL_NAME,
  OUTPUT_VALUE,
  OUTPUT_MIME_TYPE,
} = SemanticConventions;

const memoryExporter = new InMemorySpanExporter();

async function waitForSpans(count: number) {
  for (let i = 0; i < 50; i++) {
    if (memoryExporter.getFinishedSpans().length >= count) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

describe("AnthropicInstrumentation - APIPromise compatibility", () => {
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

  it("preserves APIPromise helpers on the patched messages.create", async () => {
    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY ?? "fake-api-key",
      fetch: vcrFetch("thinking-non-streaming"),
    });

    const apiPromise = client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 2048,
      thinking: { type: "enabled", budget_tokens: 1024 },
      messages: [
        {
          role: "user",
          content: "What is 27 * 453? Think it through step by step.",
        },
      ],
    });

    expect(typeof apiPromise.withResponse).toBe("function");
    expect(typeof apiPromise.asResponse).toBe("function");

    const { data, response } = await apiPromise.withResponse();
    expect(response.status).toBe(200);
    expect(data.role).toBe("assistant");

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const attributes = spans[0].attributes;
    expect(attributes[OPENINFERENCE_SPAN_KIND]).toBe(OpenInferenceSpanKind.LLM);
    expect(attributes[LLM_SYSTEM]).toBe(LLMSystem.ANTHROPIC);
    expect(attributes[LLM_PROVIDER]).toBe(LLMProvider.ANTHROPIC);
    expect(attributes[LLM_MODEL_NAME]).toBe("claude-sonnet-4-6");
    expect(attributes[OUTPUT_MIME_TYPE]).toBe(MimeType.JSON);
    expect(attributes[OUTPUT_VALUE]).toEqual(expect.any(String));
  });

  it("records errors on the span and still rejects the caller", async () => {
    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY ?? "fake-api-key",
      maxRetries: 0,
      fetch: (async () =>
        new Response(JSON.stringify({ type: "error", error: { type: "invalid_request_error" } }), {
          status: 400,
          headers: { "content-type": "application/json" },
        })) as typeof fetch,
    });

    await expect(
      client.messages.create({
        model: "claude-sonnet-4-6",
        max_tokens: 2048,
        messages: [{ role: "user", content: "hi" }],
      }),
    ).rejects.toThrow();

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    expect(spans[0].status.code).toBe(SpanStatusCode.ERROR);
    expect(spans[0].events.map((event) => event.name)).toContain("exception");
  });

  it("traces client.messages.stream(), which relies on withResponse()", async () => {
    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY ?? "fake-api-key",
      fetch: vcrFetch("thinking-streaming"),
    });

    const stream = client.messages.stream({
      model: "claude-sonnet-4-6",
      max_tokens: 2048,
      thinking: { type: "enabled", budget_tokens: 1024 },
      messages: [
        {
          role: "user",
          content: "What is 27 * 453? Think it through step by step.",
        },
      ],
    });

    for await (const _event of stream) {
      // drain the stream
    }
    const finalMessage = await stream.finalMessage();
    expect(finalMessage.role).toBe("assistant");

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    expect(spans[0].name).toBe("Anthropic Messages");
    const attributes = spans[0].attributes;
    expect(attributes[OPENINFERENCE_SPAN_KIND]).toBe(OpenInferenceSpanKind.LLM);
    expect(attributes[LLM_MODEL_NAME]).toBe("claude-sonnet-4-6");
    expect(attributes[OUTPUT_VALUE]).toEqual(expect.any(String));
  });
});
