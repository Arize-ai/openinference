import { context, trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { BasicTracerProvider, InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { isOpenInferenceSpan } from "@arizeai/openinference-vercel";

import { OpenInferenceBatchSpanProcessor, OpenInferenceSimpleSpanProcessor } from "../src";

type ProcessorConstructor =
  | typeof OpenInferenceSimpleSpanProcessor
  | typeof OpenInferenceBatchSpanProcessor;

const buildProvider = (
  ProcessorClass: ProcessorConstructor,
  spanFilter?: (span: ReadableSpan) => boolean,
) => {
  const exporter = new InMemorySpanExporter();
  const processor = new ProcessorClass({ exporter, spanFilter });
  const provider = new BasicTracerProvider({ spanProcessors: [processor] });
  const tracer = provider.getTracer("test");
  return { exporter, tracer, provider };
};

describe.each([
  ["OpenInferenceSimpleSpanProcessor", OpenInferenceSimpleSpanProcessor],
  ["OpenInferenceBatchSpanProcessor", OpenInferenceBatchSpanProcessor],
] as [string, ProcessorConstructor][])("%s — Eve attribute mapping", (_name, ProcessorClass) => {
  let exporter: InMemorySpanExporter;
  let tracer: ReturnType<typeof buildProvider>["tracer"];
  let provider: ReturnType<typeof buildProvider>["provider"];

  beforeEach(() => {
    ({ exporter, tracer, provider } = buildProvider(ProcessorClass));
  });

  afterEach(async () => {
    exporter.reset();
    await provider.shutdown();
  });

  it("sets AGENT span kind on ai.eve.turn spans", async () => {
    tracer
      .startSpan("ai.eve.turn", {
        attributes: {
          "operation.name": "ai.eve.turn",
          "eve.session.id": "sess-abc",
        },
      })
      .end();

    await provider.forceFlush();
    const [finished] = exporter.getFinishedSpans();
    expect(finished.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.AGENT,
    );
  });

  it("sets AGENT kind for ai.eve.turn with a functionId suffix", async () => {
    tracer
      .startSpan("ai.eve.turn my-agent", {
        attributes: {
          "operation.name": "ai.eve.turn my-agent",
          "eve.session.id": "sess-123",
        },
      })
      .end();

    await provider.forceFlush();
    const [finished] = exporter.getFinishedSpans();
    expect(finished.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.AGENT,
    );
  });

  it("maps eve.session.id to session.id", async () => {
    tracer
      .startSpan("ai.eve.turn", {
        attributes: {
          "operation.name": "ai.eve.turn",
          "eve.session.id": "session-xyz",
        },
      })
      .end();

    await provider.forceFlush();
    const [finished] = exporter.getFinishedSpans();
    expect(finished.attributes[SemanticConventions.SESSION_ID]).toBe("session-xyz");
  });

  it("maps eve.* attributes (excluding session.id) to metadata.*", async () => {
    tracer
      .startSpan("ai.eve.turn", {
        attributes: {
          "operation.name": "ai.eve.turn",
          "eve.session.id": "sess-1",
          "eve.version": "1.2.3",
          "eve.environment": "production",
          "eve.turn.id": "turn-42",
          "eve.turn.sequence": 3,
          "eve.step.index": 0,
          "eve.channel.kind": "http",
        },
      })
      .end();

    await provider.forceFlush();
    const [finished] = exporter.getFinishedSpans();
    expect(finished.attributes[`${SemanticConventions.METADATA}.eve.version`]).toBe("1.2.3");
    expect(finished.attributes[`${SemanticConventions.METADATA}.eve.environment`]).toBe(
      "production",
    );
    expect(finished.attributes[`${SemanticConventions.METADATA}.eve.turn.id`]).toBe("turn-42");
    expect(finished.attributes[`${SemanticConventions.METADATA}.eve.turn.sequence`]).toBe(3);
    expect(finished.attributes[`${SemanticConventions.METADATA}.eve.step.index`]).toBe(0);
    expect(finished.attributes[`${SemanticConventions.METADATA}.eve.channel.kind`]).toBe("http");
    // session.id should NOT appear as metadata — it's mapped to session.id directly
    expect(finished.attributes[`${SemanticConventions.METADATA}.eve.session.id`]).toBeUndefined();
  });

  it("propagates eve.session.id on child spans (ai.streamText.doStream)", async () => {
    const parentSpan = tracer.startSpan("ai.eve.turn", {
      attributes: {
        "operation.name": "ai.eve.turn",
        "eve.session.id": "sess-parent",
      },
    });

    tracer
      .startSpan(
        "ai.streamText.doStream",
        {
          attributes: {
            "operation.name": "ai.streamText.doStream",
            "eve.session.id": "sess-parent",
            "eve.step.index": 0,
          },
        },
        trace.setSpan(context.active(), parentSpan),
      )
      .end();
    parentSpan.end();

    await provider.forceFlush();
    const spans = exporter.getFinishedSpans();
    const child = spans.find((s) => s.name === "ai.streamText.doStream");
    expect(child?.attributes[SemanticConventions.SESSION_ID]).toBe("sess-parent");
  });

  it("does not modify spans without eve.* attributes", async () => {
    tracer.startSpan("custom-span", { attributes: { "some.attribute": "value" } }).end();

    await provider.forceFlush();
    const [finished] = exporter.getFinishedSpans();
    expect(finished.attributes[SemanticConventions.SESSION_ID]).toBeUndefined();
    expect(finished.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBeUndefined();
  });

  it("respects existing openinference.span.kind and does not override it", async () => {
    tracer
      .startSpan("ai.eve.turn", {
        attributes: {
          "operation.name": "ai.eve.turn",
          "eve.session.id": "sess-1",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
        },
      })
      .end();

    await provider.forceFlush();
    const [finished] = exporter.getFinishedSpans();
    expect(finished.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.CHAIN,
    );
  });

  it("exports only openinference spans when isOpenInferenceSpan filter is applied", async () => {
    // Tear down the default (unfiltered) provider from beforeEach
    await provider.shutdown();
    exporter.reset();

    const filtered = buildProvider(ProcessorClass, isOpenInferenceSpan);

    // Plain span with no eve/openinference attributes — should not be exported
    filtered.tracer.startSpan("plain-span").end();

    // Eve turn span gains AGENT kind → passes isOpenInferenceSpan
    filtered.tracer
      .startSpan("ai.eve.turn", {
        attributes: {
          "operation.name": "ai.eve.turn",
          "eve.session.id": "sess-filter",
        },
      })
      .end();

    await filtered.provider.forceFlush();
    const finished = filtered.exporter.getFinishedSpans();
    await filtered.provider.shutdown();

    expect(finished).toHaveLength(1);
    expect(finished[0].name).toBe("ai.eve.turn");
  });

  it("processes standard Vercel AI SDK child spans (ai.streamText.doStream → LLM)", async () => {
    const parentSpan = tracer.startSpan("ai.eve.turn", {
      attributes: {
        "operation.name": "ai.eve.turn",
        "eve.session.id": "sess-vercel",
      },
    });

    tracer
      .startSpan(
        "ai.streamText.doStream",
        {
          attributes: {
            "operation.name": "ai.streamText.doStream",
            "eve.session.id": "sess-vercel",
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "gpt-4o",
          },
        },
        trace.setSpan(context.active(), parentSpan),
      )
      .end();
    parentSpan.end();

    await provider.forceFlush();
    const spans = exporter.getFinishedSpans();
    const llm = spans.find((s) => s.name === "ai.streamText.doStream");
    expect(llm?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.LLM,
    );
    expect(llm?.attributes[SemanticConventions.SESSION_ID]).toBe("sess-vercel");
  });
});
