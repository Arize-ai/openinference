import { context, trace } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { BasicTracerProvider, InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { setSession } from "@arizeai/openinference-core";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import {
  isOpenInferenceSpan,
  OpenInferenceBatchSpanProcessor,
  OpenInferenceSimpleSpanProcessor,
} from "../src";

type ProcessorConstructor =
  | typeof OpenInferenceSimpleSpanProcessor
  | typeof OpenInferenceBatchSpanProcessor;

const buildProvider = (
  ProcessorClass: ProcessorConstructor,
  opts: {
    agentTraceMode?: boolean;
    spanFilter?: (span: ReadableSpan) => boolean;
  } = {},
) => {
  const exporter = new InMemorySpanExporter();
  const processor = new ProcessorClass({ exporter, ...opts });
  const provider = new BasicTracerProvider({ spanProcessors: [processor] });
  const tracer = provider.getTracer("test");
  return { exporter, tracer, provider };
};

/** Reads `parentSpanId` defensively across sdk-trace-base versions. */
const getParentId = (span: ReadableSpan): string | undefined => {
  const maybe = span as unknown as {
    parentSpanId?: string;
    parentSpanContext?: { spanId?: string };
  };
  return maybe.parentSpanId ?? maybe.parentSpanContext?.spanId;
};

describe.each([
  ["OpenInferenceSimpleSpanProcessor", OpenInferenceSimpleSpanProcessor],
  ["OpenInferenceBatchSpanProcessor", OpenInferenceBatchSpanProcessor],
] as [string, ProcessorConstructor][])("%s — agentTraceMode", (_name, ProcessorClass) => {
  let exporter: InMemorySpanExporter;
  let tracer: ReturnType<typeof buildProvider>["tracer"];
  let provider: ReturnType<typeof buildProvider>["provider"];

  afterEach(async () => {
    exporter.reset();
    await provider.shutdown();
  });

  describe("enabled", () => {
    beforeEach(() => {
      ({ exporter, tracer, provider } = buildProvider(ProcessorClass, {
        agentTraceMode: true,
        spanFilter: isOpenInferenceSpan,
      }));
    });

    it("promotes the first ai.* span in a trace to root", async () => {
      const httpSpan = tracer.startSpan("GET /chat");
      tracer
        .startSpan(
          "ai.streamText",
          { attributes: { "operation.name": "ai.streamText" } },
          trace.setSpan(context.active(), httpSpan),
        )
        .end();
      httpSpan.end();

      await provider.forceFlush();
      const aiSpan = exporter.getFinishedSpans().find((s) => s.name === "ai.streamText");
      expect(aiSpan).toBeDefined();
      expect(getParentId(aiSpan as ReadableSpan)).toBeUndefined();
      expect(aiSpan?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
        OpenInferenceSpanKind.AGENT,
      );
    });

    it("promotes a framework wrapper span and labels it AGENT so it survives the filter", async () => {
      // ai.eve.turn is unknown to the Vercel span-kind map and carries no I/O.
      const turn = tracer.startSpan("ai.eve.turn", {
        attributes: { "operation.name": "ai.eve.turn" },
      });
      tracer
        .startSpan(
          "ai.streamText",
          { attributes: { "operation.name": "ai.streamText" } },
          trace.setSpan(context.active(), turn),
        )
        .end();
      turn.end();

      await provider.forceFlush();
      const root = exporter.getFinishedSpans().find((s) => s.name === "ai.eve.turn");
      expect(root).toBeDefined();
      expect(getParentId(root as ReadableSpan)).toBeUndefined();
      expect(root?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
        OpenInferenceSpanKind.AGENT,
      );
    });

    it("only promotes the first ai.* span per trace", async () => {
      const turn = tracer.startSpan("ai.eve.turn", {
        attributes: { "operation.name": "ai.eve.turn" },
      });
      const turnId = turn.spanContext().spanId;
      tracer
        .startSpan(
          "ai.streamText",
          { attributes: { "operation.name": "ai.streamText" } },
          trace.setSpan(context.active(), turn),
        )
        .end();
      turn.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      expect(
        getParentId(spans.find((s) => s.name === "ai.eve.turn") as ReadableSpan),
      ).toBeUndefined();
      // The nested ai.streamText keeps its parent (the turn span).
      expect(getParentId(spans.find((s) => s.name === "ai.streamText") as ReadableSpan)).toBe(
        turnId,
      );
    });

    it("propagates session.id from the active context onto spans", async () => {
      const ctx = setSession(context.active(), { sessionId: "sess-xyz" });
      tracer
        .startSpan("ai.streamText", { attributes: { "operation.name": "ai.streamText" } }, ctx)
        .end();

      await provider.forceFlush();
      const aiSpan = exporter.getFinishedSpans().find((s) => s.name === "ai.streamText");
      expect(aiSpan?.attributes[SemanticConventions.SESSION_ID]).toBe("sess-xyz");
    });

    it("promotes a wrapper span whose name carries a functionId suffix", async () => {
      const httpSpan = tracer.startSpan("POST /agent");
      tracer
        .startSpan(
          "ai.eve.turn my-agent",
          { attributes: { "operation.name": "ai.eve.turn my-agent" } },
          trace.setSpan(context.active(), httpSpan),
        )
        .end();
      httpSpan.end();

      await provider.forceFlush();
      const root = exporter.getFinishedSpans().find((s) => s.name === "ai.eve.turn my-agent");
      expect(getParentId(root as ReadableSpan)).toBeUndefined();
      expect(root?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
        OpenInferenceSpanKind.AGENT,
      );
    });

    it("propagates session.id from context onto child spans too", async () => {
      const ctx = setSession(context.active(), { sessionId: "sess-parent" });
      const root = tracer.startSpan(
        "ai.eve.turn",
        { attributes: { "operation.name": "ai.eve.turn" } },
        ctx,
      );
      tracer
        .startSpan(
          "ai.streamText.doStream",
          {
            attributes: {
              "operation.name": "ai.streamText.doStream",
              "gen_ai.request.model": "gpt-4o",
            },
          },
          trace.setSpan(ctx, root),
        )
        .end();
      root.end();

      await provider.forceFlush();
      const child = exporter.getFinishedSpans().find((s) => s.name === "ai.streamText.doStream");
      expect(child?.attributes[SemanticConventions.SESSION_ID]).toBe("sess-parent");
    });

    it("does not override an existing span kind on the promoted root", async () => {
      tracer
        .startSpan("ai.eve.turn", {
          attributes: {
            "operation.name": "ai.eve.turn",
            [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
          },
        })
        .end();

      await provider.forceFlush();
      const root = exporter.getFinishedSpans().find((s) => s.name === "ai.eve.turn");
      expect(root?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
        OpenInferenceSpanKind.CHAIN,
      );
    });

    it("stamps earliest input and latest output onto a wrapper root with none of its own", async () => {
      const turn = tracer.startSpan("ai.eve.turn", {
        attributes: { "operation.name": "ai.eve.turn" },
        startTime: 1_000,
      });
      const turnCtx = trace.setSpan(context.active(), turn);

      tracer
        .startSpan(
          "ai.streamText",
          {
            attributes: {
              "operation.name": "ai.streamText",
              "ai.prompt": JSON.stringify({ messages: [{ role: "user", content: "first" }] }),
              "ai.response.text": "step 0 output",
            },
            startTime: 1_010,
          },
          turnCtx,
        )
        .end(1_020);

      tracer
        .startSpan(
          "ai.streamText",
          {
            attributes: {
              "operation.name": "ai.streamText",
              "ai.prompt": JSON.stringify({ messages: [{ role: "user", content: "second" }] }),
              "ai.response.text": "final answer",
            },
            startTime: 1_030,
          },
          turnCtx,
        )
        .end(1_040);

      turn.end(1_050);

      await provider.forceFlush();
      const root = exporter.getFinishedSpans().find((s) => s.name === "ai.eve.turn");
      // input comes from the earliest-started child, output from the latest-ended child
      expect(root?.attributes[SemanticConventions.INPUT_VALUE]).toBe(
        JSON.stringify({ messages: [{ role: "user", content: "first" }] }),
      );
      expect(root?.attributes[SemanticConventions.OUTPUT_VALUE]).toBe("final answer");
    });

    it("does not overwrite input/output the root already has", async () => {
      // A top-level ai.streamText root already carries its own ai.prompt/response.
      const root = tracer.startSpan("ai.streamText", {
        attributes: {
          "operation.name": "ai.streamText",
          "ai.prompt": JSON.stringify({ messages: [{ role: "user", content: "root prompt" }] }),
          "ai.response.text": "root answer",
        },
      });
      tracer
        .startSpan(
          "ai.streamText.doStream",
          {
            attributes: {
              "operation.name": "ai.streamText.doStream",
              "ai.response.text": "child answer",
            },
          },
          trace.setSpan(context.active(), root),
        )
        .end();
      root.end();

      await provider.forceFlush();
      const rootSpan = exporter.getFinishedSpans().find((s) => s.name === "ai.streamText");
      expect(rootSpan?.attributes[SemanticConventions.OUTPUT_VALUE]).toBe("root answer");
    });

    it("defers the root and stamps a late child's output even when the root ends first", async () => {
      // A framework wrapper (ai.eve.turn) commonly ends before its children. The
      // root must be held back and stamped with the final child's output.
      const turn = tracer.startSpan("ai.eve.turn", {
        attributes: { "operation.name": "ai.eve.turn" },
      });
      const child = tracer.startSpan(
        "ai.streamText",
        {
          attributes: {
            "operation.name": "ai.streamText",
            "ai.prompt": JSON.stringify({ messages: [{ role: "user", content: "hi" }] }),
            "ai.response.text": "the final answer",
          },
        },
        trace.setSpan(context.active(), turn),
      );
      // Root ends BEFORE the child.
      turn.end();
      child.end();

      await provider.forceFlush();
      const root = exporter.getFinishedSpans().find((s) => s.name === "ai.eve.turn");
      expect(root).toBeDefined();
      expect(getParentId(root as ReadableSpan)).toBeUndefined();
      expect(root?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
        OpenInferenceSpanKind.AGENT,
      );
      expect(root?.attributes[SemanticConventions.INPUT_VALUE]).toBeTruthy();
      expect(root?.attributes[SemanticConventions.OUTPUT_VALUE]).toBe("the final answer");
    });

    it("stamps llm message attributes from GenAI-style children onto a wrapper root", async () => {
      // Eve's children are GenAI-semconv spans carrying I/O as flattened
      // llm.input_messages.* / llm.output_messages.* rather than input/output.value.
      const turn = tracer.startSpan("ai.eve.turn", {
        attributes: { "operation.name": "ai.eve.turn" },
      });
      const child = tracer.startSpan(
        "gen_ai.client",
        {
          attributes: {
            "gen_ai.operation.name": "chat",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "what is the weather",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "sunny",
          },
        },
        trace.setSpan(context.active(), turn),
      );
      turn.end();
      child.end();

      await provider.forceFlush();
      const root = exporter.getFinishedSpans().find((s) => s.name === "ai.eve.turn");
      // messages copied onto the root
      expect(root?.attributes["llm.input_messages.0.message.content"]).toBe("what is the weather");
      expect(root?.attributes["llm.output_messages.0.message.content"]).toBe("sunny");
      // and a scalar input.value/output.value derived from them so tools that key
      // off input.value/output.value (Phoenix/AX panels) render the I/O
      const inputValue = root?.attributes[SemanticConventions.INPUT_VALUE];
      const outputValue = root?.attributes[SemanticConventions.OUTPUT_VALUE];
      expect(typeof inputValue).toBe("string");
      expect(inputValue as string).toContain("what is the weather");
      expect(outputValue as string).toContain("sunny");
      expect(root?.attributes[SemanticConventions.OUTPUT_MIME_TYPE]).toBe(MimeType.JSON);
    });

    it("propagates input/output messages to a deferred wrapper root (real Eve shape)", async () => {
      // The exact shape Eve produces: ai.eve.turn ends BEFORE its children, and
      // the children are multi-step GenAI spans carrying I/O as flattened
      // llm.*_messages. The root's input must come from the earliest step and its
      // output from the latest step, with scalar input.value/output.value derived.
      const turn = tracer.startSpan("ai.eve.turn", {
        attributes: { "operation.name": "ai.eve.turn" },
        startTime: 1_000,
      });
      const ctx = trace.setSpan(context.active(), turn);

      // Earliest step (carries the user's turn input + an interim response).
      const step0 = tracer.startSpan(
        "gen_ai.client",
        {
          attributes: {
            "gen_ai.operation.name": "chat",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "You are a helpful assistant.",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "Show me dinosaur toys",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Let me search.",
          },
          startTime: 1_010,
        },
        ctx,
      );
      // Latest step (carries the final answer).
      const step1 = tracer.startSpan(
        "gen_ai.client",
        {
          attributes: {
            "gen_ai.operation.name": "chat",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "(tool results)",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Here are some great dinosaur toys!",
          },
          startTime: 1_030,
        },
        ctx,
      );

      // Root ends FIRST (before the steps complete) — exercises the deferral path.
      turn.end(1_015);
      step0.end(1_020);
      step1.end(1_040);

      await provider.forceFlush();
      const root = exporter.getFinishedSpans().find((s) => s.name === "ai.eve.turn");
      expect(root).toBeDefined();
      expect(getParentId(root as ReadableSpan)).toBeUndefined();
      expect(root?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
        OpenInferenceSpanKind.AGENT,
      );

      // messages: input from the earliest step, output from the latest step
      expect(root?.attributes["llm.input_messages.1.message.content"]).toBe(
        "Show me dinosaur toys",
      );
      expect(root?.attributes["llm.output_messages.0.message.content"]).toBe(
        "Here are some great dinosaur toys!",
      );

      // scalar values derived from those messages, as parseable JSON
      const input = JSON.parse(root?.attributes[SemanticConventions.INPUT_VALUE] as string);
      const output = JSON.parse(root?.attributes[SemanticConventions.OUTPUT_VALUE] as string);
      expect(JSON.stringify(input)).toContain("Show me dinosaur toys");
      expect(JSON.stringify(output)).toContain("Here are some great dinosaur toys!");
      // the interim "Let me search." output must NOT leak in as the final output
      expect(JSON.stringify(output)).not.toContain("Let me search.");
      expect(root?.attributes[SemanticConventions.INPUT_MIME_TYPE]).toBe(MimeType.JSON);
      expect(root?.attributes[SemanticConventions.OUTPUT_MIME_TYPE]).toBe(MimeType.JSON);
    });
  });

  describe("disabled (default)", () => {
    beforeEach(() => {
      ({ exporter, tracer, provider } = buildProvider(ProcessorClass));
    });

    it("does not change trace topology, span kind, or session", async () => {
      const turn = tracer.startSpan("ai.eve.turn", {
        attributes: { "operation.name": "ai.eve.turn" },
      });
      const turnId = turn.spanContext().spanId;
      const ctx = setSession(context.active(), { sessionId: "sess-1" });
      tracer
        .startSpan(
          "ai.streamText",
          { attributes: { "operation.name": "ai.streamText" } },
          trace.setSpan(ctx, turn),
        )
        .end();
      turn.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      const child = spans.find((s) => s.name === "ai.streamText");
      const root = spans.find((s) => s.name === "ai.eve.turn");
      // child keeps its real parent; wrapper gets no AGENT kind; no session propagated
      expect(getParentId(child as ReadableSpan)).toBe(turnId);
      expect(root?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBeUndefined();
      expect(child?.attributes[SemanticConventions.SESSION_ID]).toBeUndefined();
    });
  });
});
