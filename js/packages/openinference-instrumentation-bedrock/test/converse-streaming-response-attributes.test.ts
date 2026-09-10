import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { consumeConverseStreamChunks } from "../src/attributes/converse-streaming-response-attributes";
import { toNormalizedConverseStreamEvent } from "../src/types/bedrock-types";

describe("Converse streaming response attributes", () => {
  it("preserves one text block across raw-wire deltas", async () => {
    const exporter = new InMemorySpanExporter();
    const provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    const span = provider.getTracer("test").startSpan("converse");

    async function* stream() {
      yield {
        type: "content_block_delta",
        index: 4,
        delta: { type: "text_delta", text: "hel" },
      };
      yield {
        type: "content_block_delta",
        index: 4,
        delta: { type: "text_delta", text: "lo" },
      };
    }

    await consumeConverseStreamChunks({ stream: stream(), span });
    span.end();

    const [finishedSpan] = exporter.getFinishedSpans();
    expect(finishedSpan.attributes["llm.output_messages.0.message.content"]).toBe("hello");
    expect(
      finishedSpan.attributes["llm.output_messages.0.message.contents.0.message_content.text"],
    ).toBeUndefined();

    await provider.shutdown();
  });

  it("preserves raw-wire indices for tool events", () => {
    expect(
      toNormalizedConverseStreamEvent({
        type: "content_block_start",
        index: 7,
        content_block: { type: "tool_use", id: "tool-1", name: "weather" },
      }),
    ).toEqual({
      kind: "toolUseStart",
      id: "tool-1",
      name: "weather",
      contentBlockIndex: 7,
    });
    expect(
      toNormalizedConverseStreamEvent({
        type: "input_json_delta",
        index: 7,
        partial_json: '{"city":"Paris"}',
      }),
    ).toEqual({
      kind: "toolUseInputChunk",
      chunk: '{"city":"Paris"}',
      contentBlockIndex: 7,
    });
  });
});
