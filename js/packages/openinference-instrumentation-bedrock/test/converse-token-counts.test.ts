import type { ConverseResponse } from "@aws-sdk/client-bedrock-runtime";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { extractConverseResponseAttributes } from "../src/attributes/converse-response-attributes";
import { consumeConverseStreamChunks } from "../src/attributes/converse-streaming-response-attributes";

// Bedrock's inputTokens excludes cache reads and writes, while totalTokens includes them.
const CACHED_USAGE = {
  inputTokens: 4299,
  outputTokens: 631,
  totalTokens: 58291,
  cacheReadInputTokens: 35574,
  cacheWriteInputTokens: 17787,
};
const UNCACHED_USAGE = { inputTokens: 10, outputTokens: 5, totalTokens: 15 };

type Usage = typeof UNCACHED_USAGE & {
  cacheReadInputTokens?: number;
  cacheWriteInputTokens?: number;
};

const {
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_TOTAL,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
} = SemanticConventions;

describe("Converse token counts", () => {
  let exporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
  });

  afterEach(async () => {
    await provider.shutdown();
  });

  const runConverse = (usage: Usage) => {
    const span = provider.getTracer("test").startSpan("converse");
    const response = {
      output: { message: { role: "assistant", content: [{ text: "Answer." }] } },
      stopReason: "end_turn",
      usage,
    } as ConverseResponse;
    extractConverseResponseAttributes({ span, response });
    span.end();
    return exporter.getFinishedSpans()[0].attributes;
  };

  const runConverseStream = async (usage: Usage) => {
    const span = provider.getTracer("test").startSpan("converse");
    async function* stream() {
      yield { contentBlockDelta: { contentBlockIndex: 0, delta: { text: "Answer." } } };
      yield { messageStop: { stopReason: "end_turn" } };
      yield { metadata: { usage, metrics: { latencyMs: 10 } } };
    }
    await consumeConverseStreamChunks({ stream: stream(), span });
    span.end();
    return exporter.getFinishedSpans()[0].attributes;
  };

  it.each([
    ["Converse", runConverse],
    ["ConverseStream", runConverseStream],
  ])("%s folds cache tokens into the prompt count", async (_name, run) => {
    const attributes = await run(CACHED_USAGE);
    expect(attributes[LLM_TOKEN_COUNT_PROMPT]).toBe(4299 + 35574 + 17787);
    expect(attributes[LLM_TOKEN_COUNT_COMPLETION]).toBe(631);
    expect(attributes[LLM_TOKEN_COUNT_TOTAL]).toBe(58291);
    expect(attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(35574);
    expect(attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]).toBe(17787);
    expect(
      Number(attributes[LLM_TOKEN_COUNT_PROMPT]) + Number(attributes[LLM_TOKEN_COUNT_COMPLETION]),
    ).toBe(attributes[LLM_TOKEN_COUNT_TOTAL]);
  });

  it.each([
    ["Converse", runConverse],
    ["ConverseStream", runConverseStream],
  ])("%s keeps inputTokens when there are no cache tokens", async (_name, run) => {
    const attributes = await run(UNCACHED_USAGE);
    expect(attributes[LLM_TOKEN_COUNT_PROMPT]).toBe(10);
    expect(attributes[LLM_TOKEN_COUNT_COMPLETION]).toBe(5);
    expect(attributes[LLM_TOKEN_COUNT_TOTAL]).toBe(15);
    expect(attributes).not.toHaveProperty(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ);
    expect(attributes).not.toHaveProperty(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE);
  });
});
