import Anthropic from "@anthropic-ai/sdk";
import type { BetaRunnableTool } from "@anthropic-ai/sdk/lib/tools/BetaRunnableTool";
import { SpanStatusCode } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";
import type { StreamEvent } from "./helpers/sse";
import { messageDelta, messageStart, toSSE } from "./helpers/sse";
import { waitForSpans as waitForSpansOn } from "./helpers/waitForSpans";

const {
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  LLM_INPUT_MESSAGES,
  LLM_MODEL_NAME,
  LLM_OUTPUT_MESSAGES,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_TOTAL,
  MESSAGE_CONTENT,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENTS,
  MESSAGE_ROLE,
  OPENINFERENCE_SPAN_KIND,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  TOOL_DESCRIPTION,
  TOOL_NAME,
  TOOL_PARAMETERS,
} = SemanticConventions;

const memoryExporter = new InMemorySpanExporter();
const waitForSpans = (count: number) => waitForSpansOn(memoryExporter, count);

const request = {
  model: "claude-sonnet-4-6",
  max_tokens: 100,
  messages: [{ role: "user" as const, content: "What is the weather in Paris?" }],
};

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

/** A `fetch` that answers each request with the next of the given responses. */
function sequenceFetch(responses: Array<() => Response>): typeof fetch {
  let call = 0;
  return async () => responses[Math.min(call++, responses.length - 1)]();
}

function message(content: unknown[], stopReason: string) {
  return {
    id: `msg_${stopReason}`,
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-6",
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: { input_tokens: 10, output_tokens: 5 },
  };
}

const toolUseMessage = message(
  [{ type: "tool_use", id: "toolu_1", name: "get_weather", input: { city: "Paris" } }],
  "tool_use",
);
const finalMessage = message([{ type: "text", text: "It is sunny in Paris." }], "end_turn");

const inputSchema = {
  type: "object" as const,
  properties: { city: { type: "string" } },
  required: ["city"],
};

function weatherTool(run: (args: { city: string }) => string): BetaRunnableTool<{ city: string }> {
  return {
    name: "get_weather",
    description: "Get the weather for a city",
    input_schema: inputSchema,
    parse: (input: unknown) => input as { city: string },
    run,
  };
}

function byKind(spans: ReadableSpan[], kind: OpenInferenceSpanKind) {
  return spans.filter((span) => span.attributes[OPENINFERENCE_SPAN_KIND] === kind);
}

describe("AnthropicInstrumentation - other Messages methods", () => {
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

  describe("countTokens", () => {
    it.each([
      {
        name: "messages.countTokens",
        count: (client: Anthropic) => client.messages.countTokens(request),
      },
      {
        name: "beta.messages.countTokens",
        count: (client: Anthropic) => client.beta.messages.countTokens(request),
      },
    ])("traces $name as an LLM span with the count as output", async ({ count }) => {
      const client = new Anthropic({
        apiKey: "fake-api-key",
        fetch: async () => jsonResponse({ input_tokens: 12 }),
      });
      const result = await count(client);
      expect(result.input_tokens).toBe(12);

      await waitForSpans(1);
      const spans = memoryExporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      const [span] = spans;
      expect(span.name).toBe("Anthropic Count Tokens");
      expect(span.status.code).toBe(SpanStatusCode.OK);
      expect(span.attributes[OPENINFERENCE_SPAN_KIND]).toBe(OpenInferenceSpanKind.LLM);
      expect(span.attributes[LLM_MODEL_NAME]).toBe("claude-sonnet-4-6");
      expect(span.attributes[INPUT_MIME_TYPE]).toBe(MimeType.JSON);
      expect(span.attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`]).toBe("user");
      expect(span.attributes[`${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`]).toBe(
        "What is the weather in Paris?",
      );
      expect(span.attributes[OUTPUT_VALUE]).toBe(JSON.stringify({ input_tokens: 12 }));
      expect(span.attributes[OUTPUT_MIME_TYPE]).toBe(MimeType.JSON);
      // Counting is free, so it records no token usage to be priced.
      expect(span.attributes[LLM_TOKEN_COUNT_PROMPT]).toBeUndefined();
      expect(span.attributes[LLM_TOKEN_COUNT_TOTAL]).toBeUndefined();
    });

    it("records a failed count", async () => {
      const client = new Anthropic({
        apiKey: "fake-api-key",
        maxRetries: 0,
        fetch: async () =>
          new Response(
            JSON.stringify({ type: "error", error: { type: "invalid_request_error" } }),
            {
              status: 400,
              headers: { "content-type": "application/json" },
            },
          ),
      });
      await expect(client.messages.countTokens(request)).rejects.toThrow();

      await waitForSpans(1);
      const [span] = memoryExporter.getFinishedSpans();
      expect(span.status.code).toBe(SpanStatusCode.ERROR);
    });
  });

  describe("beta.messages.toolRunner", () => {
    it("parents the LLM and TOOL spans of a run with a CHAIN span", async () => {
      const client = new Anthropic({
        apiKey: "fake-api-key",
        fetch: sequenceFetch([
          () => jsonResponse(toolUseMessage),
          () => jsonResponse(finalMessage),
        ]),
      });
      const tool = weatherTool(() => "sunny");

      const result = await client.beta.messages.toolRunner({ ...request, tools: [tool] });
      expect(result.content).toEqual(finalMessage.content);

      await waitForSpans(4);
      const spans = memoryExporter.getFinishedSpans();
      expect(spans).toHaveLength(4);
      const [chain] = byKind(spans, OpenInferenceSpanKind.CHAIN);
      const llms = byKind(spans, OpenInferenceSpanKind.LLM);
      const [toolSpan] = byKind(spans, OpenInferenceSpanKind.TOOL);

      expect(chain.name).toBe("beta.messages.tool_runner");
      expect(chain.parentSpanId).toBeUndefined();
      expect(chain.status.code).toBe(SpanStatusCode.OK);
      expect(JSON.parse(chain.attributes[INPUT_VALUE] as string).messages).toEqual(
        request.messages,
      );
      expect(JSON.parse(chain.attributes[OUTPUT_VALUE] as string).id).toBe("msg_end_turn");
      expect(
        chain.attributes[`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`],
      ).toBe("It is sunny in Paris.");

      expect(llms).toHaveLength(2);
      for (const llm of llms) {
        expect(llm.parentSpanId).toBe(chain.spanContext().spanId);
      }

      expect(toolSpan.name).toBe("get_weather");
      expect(toolSpan.parentSpanId).toBe(chain.spanContext().spanId);
      expect(toolSpan.status.code).toBe(SpanStatusCode.OK);
      expect(toolSpan.attributes[TOOL_NAME]).toBe("get_weather");
      expect(toolSpan.attributes[TOOL_DESCRIPTION]).toBe("Get the weather for a city");
      expect(toolSpan.attributes[TOOL_PARAMETERS]).toBe(JSON.stringify(inputSchema));
      expect(toolSpan.attributes[INPUT_VALUE]).toBe(JSON.stringify({ city: "Paris" }));
      expect(toolSpan.attributes[OUTPUT_VALUE]).toBe("sunny");
      expect(toolSpan.attributes[OUTPUT_MIME_TYPE]).toBe(MimeType.TEXT);
    });

    it("records a failing tool, which the runner reports to the model", async () => {
      const client = new Anthropic({
        apiKey: "fake-api-key",
        fetch: sequenceFetch([
          () => jsonResponse(toolUseMessage),
          () => jsonResponse(finalMessage),
        ]),
      });
      const tool = weatherTool(() => {
        throw new Error("weather service down");
      });

      await client.beta.messages.toolRunner({ ...request, tools: [tool] }).runUntilDone();

      await waitForSpans(4);
      const spans = memoryExporter.getFinishedSpans();
      const [toolSpan] = byKind(spans, OpenInferenceSpanKind.TOOL);
      expect(toolSpan.status.code).toBe(SpanStatusCode.ERROR);
      expect(toolSpan.status.message).toBe("weather service down");
      const [chain] = byKind(spans, OpenInferenceSpanKind.CHAIN);
      expect(chain.status.code).toBe(SpanStatusCode.OK);
    });

    it("ends the CHAIN span when the caller leaves the run early", async () => {
      const client = new Anthropic({
        apiKey: "fake-api-key",
        fetch: sequenceFetch([
          () => jsonResponse(toolUseMessage),
          () => jsonResponse(finalMessage),
        ]),
      });
      const runner = client.beta.messages.toolRunner({
        ...request,
        tools: [weatherTool(() => "sunny")],
      });

      for await (const _message of runner) {
        break;
      }

      await waitForSpans(2);
      const spans = memoryExporter.getFinishedSpans();
      expect(spans).toHaveLength(2);
      const [chain] = byKind(spans, OpenInferenceSpanKind.CHAIN);
      expect(chain.status.code).toBe(SpanStatusCode.UNSET);
      expect(byKind(spans, OpenInferenceSpanKind.TOOL)).toHaveLength(0);
    });

    it("records a failed run on the CHAIN span", async () => {
      const client = new Anthropic({
        apiKey: "fake-api-key",
        maxRetries: 0,
        fetch: async () =>
          new Response(JSON.stringify({ type: "error", error: { type: "api_error" } }), {
            status: 500,
            headers: { "content-type": "application/json" },
          }),
      });

      await expect(
        client.beta.messages.toolRunner({ ...request, tools: [weatherTool(() => "sunny")] }),
      ).rejects.toThrow();

      await waitForSpans(2);
      const spans = memoryExporter.getFinishedSpans();
      const [chain] = byKind(spans, OpenInferenceSpanKind.CHAIN);
      expect(chain.status.code).toBe(SpanStatusCode.ERROR);
      const [llm] = byKind(spans, OpenInferenceSpanKind.LLM);
      expect(llm.parentSpanId).toBe(chain.spanContext().spanId);
    });

    it("parents the spans of a streaming run", async () => {
      const sse = (events: StreamEvent[]) => () =>
        new Response(toSSE(events), {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        });
      const client = new Anthropic({
        apiKey: "fake-api-key",
        fetch: sequenceFetch([
          sse([
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
            messageDelta({ stopReason: "tool_use" }),
            { type: "message_stop" },
          ]),
          sse([
            messageStart(),
            { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
            {
              type: "content_block_delta",
              index: 0,
              delta: { type: "text_delta", text: "It is sunny." },
            },
            { type: "content_block_stop", index: 0 },
            messageDelta(),
            { type: "message_stop" },
          ]),
        ]),
      });
      const runner = client.beta.messages.toolRunner({
        ...request,
        stream: true,
        tools: [weatherTool(() => "sunny")],
      });

      for await (const stream of runner) {
        for await (const _event of stream) {
          // drain
        }
      }

      await waitForSpans(4);
      const spans = memoryExporter.getFinishedSpans();
      expect(spans).toHaveLength(4);
      const [chain] = byKind(spans, OpenInferenceSpanKind.CHAIN);
      expect(chain.status.code).toBe(SpanStatusCode.OK);
      for (const span of spans.filter((span) => span !== chain)) {
        expect(span.parentSpanId, span.name).toBe(chain.spanContext().spanId);
      }
    });

    it("does not modify the caller's tools", async () => {
      const client = new Anthropic({
        apiKey: "fake-api-key",
        fetch: sequenceFetch([() => jsonResponse(finalMessage)]),
      });
      const tool = weatherTool(() => "sunny");
      const run = tool.run;
      const params = { ...request, tools: [tool] };

      await client.beta.messages.toolRunner(params);

      expect(params.tools[0]).toBe(tool);
      expect(tool.run).toBe(run);
    });
  });
});
