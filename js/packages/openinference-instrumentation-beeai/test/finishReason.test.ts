import {
  InMemorySpanExporter,
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { DummyChatModel } from "beeai-framework/adapters/dummy/backend/chat";
import type { ChatModelFinishReason } from "beeai-framework/backend/chat";
import { ChatModelOutput } from "beeai-framework/backend/chat";
import { AssistantMessage, UserMessage } from "beeai-framework/backend/message";

import { OITracer } from "@arizeai/openinference-core";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { createTelemetryMiddleware } from "../src/middleware";

class FinishReasonChatModel extends DummyChatModel {
  constructor(private readonly finishReason?: ChatModelFinishReason) {
    super();
  }

  protected async _create(): Promise<ChatModelOutput> {
    return new ChatModelOutput([new AssistantMessage("Hello")], undefined, this.finishReason);
  }

  protected async *_createStream(): AsyncGenerator<ChatModelOutput> {
    yield new ChatModelOutput([new AssistantMessage("Hello")]);
    yield new ChatModelOutput([], undefined, this.finishReason);
  }
}

describe.each([false, true])("LLM finish reason (stream=%s)", (stream) => {
  const exporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(exporter)],
  });
  const tracer = new OITracer({ tracer: provider.getTracer("beeai-test") });

  afterEach(() => exporter.reset());
  afterAll(() => provider.shutdown());

  const finishReasons: (ChatModelFinishReason | undefined)[] = [
    "stop",
    "length",
    "tool-calls",
    "content-filter",
    "error",
    "other",
    "unknown",
    undefined,
  ];

  it.each(finishReasons)("records finish reason %s on LLM spans", async (finishReason) => {
    const model = new FinishReasonChatModel(finishReason);
    try {
      await model
        .create({ messages: [new UserMessage("Hi")], stream })
        .middleware(createTelemetryMiddleware(tracer, OpenInferenceSpanKind.LLM));

      const spans = exporter.getFinishedSpans();
      const mainSpan = spans.find((span) => span.name === "beeai-framework-main");
      const successSpan = spans.find((span) => span.attributes.name === "success");
      expect(mainSpan).toBeDefined();
      expect(successSpan).toBeDefined();
      for (const span of [mainSpan!, successSpan!]) {
        expect(span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
          OpenInferenceSpanKind.LLM,
        );
        if (finishReason === undefined) {
          expect(span.attributes).not.toHaveProperty(SemanticConventions.LLM_FINISH_REASON);
        } else {
          expect(span.attributes[SemanticConventions.LLM_FINISH_REASON]).toBe(finishReason);
        }
      }

      const startSpan = spans.find((span) => span.attributes.name === "start");
      expect(startSpan).toBeDefined();
      expect(startSpan!.attributes).not.toHaveProperty(SemanticConventions.LLM_FINISH_REASON);
    } finally {
      model.destroy();
    }
  });
});
