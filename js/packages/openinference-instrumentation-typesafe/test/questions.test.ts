import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import * as TypeSafe from "@typesafe-ai/sdk";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { TypeSafeInstrumentation } from "../src";

const questions = {
  category: TypeSafe.choice(
    { task: "classify", focus: ["subject", "body"] },
    { billing: { description: "Payments" }, other: null },
  ),
  urgent: TypeSafe.noul(["Does the ticket require immediate attention?"], {
    true: { priority: "high" },
    false: null,
  }),
  severity: TypeSafe.score("How severe?", [null, { level: "minor" }, ["critical", "urgent"]]),
} satisfies TypeSafe.Questions;

const plainQuestions = {
  category: { type: "choice", criteria: { billing: null, other: "Other topics" } },
  urgent: { type: "noul" },
  severity: {
    type: "score",
    instructions: ["Assess severity"],
    criteria: ["low", "medium", "high"],
  },
} satisfies TypeSafe.Questions;

const answers = {
  category: {
    type: "choice",
    choice: "billing",
    confidence: 0,
    probabilities: { billing: 0.5, other: 0.5 },
  },
  urgent: { type: "noul", noul: 0.87 },
  severity: {
    type: "score",
    score: 1.7,
    confidence: 0.9,
    probabilities: { 0: 0.1, 1: 0.1, 2: 0.8 },
    legend: { 0: null, 1: { level: "minor" }, 2: ["critical", "urgent"] },
  },
} satisfies TypeSafe.SystemOneResult<typeof questions>["answers"];

describe("question and state extraction", () => {
  let exporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;
  let instrumentation: TypeSafeInstrumentation;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] });
    instrumentation = new TypeSafeInstrumentation({ tracerProvider: provider });
    instrumentation.manuallyInstrument(TypeSafe);
  });
  afterEach(async () => {
    instrumentation.disable();
    await provider.shutdown();
  });

  it.each(["category", "urgent", "severity"] as const)(
    "supports the %s builder and plain objects",
    async (name) => {
      for (const question of [questions[name], plainQuestions[name]]) {
        const request = { state: "test", questions: { [name]: question } };
        const result = {
          model: "jev",
          answers: { [name]: answers[name] },
          usage: { input_tokens: 10, output_tokens: 2 },
        };
        const client = new TypeSafe.TypeSafeClient({
          apiKey: "test",
          fetch: async () => Response.json(result),
        });
        expect(await client.systemOne(request)).toEqual(result);
        const span = exporter.getFinishedSpans().at(-1)!;
        expect(JSON.parse(String(span.attributes["input.value"])).questions[name]).toEqual(
          JSON.parse(JSON.stringify(question)),
        );
        expect(JSON.parse(String(span.attributes["output.value"]))).toEqual(result);
        const metadata = JSON.parse(String(span.attributes.metadata));
        expect(metadata.typesafe.questions[name].type).toBe(question.type);
        if (name === "urgent") {
          expect(metadata.typesafe.questions[name]).not.toHaveProperty("confidence");
        } else {
          expect(metadata.typesafe.questions[name].confidence).toBe(answers[name].confidence);
        }
      }
      expect(exporter.getFinishedSpans()).toHaveLength(2);
    },
  );

  it.each<TypeSafe.EntryType>([
    "text",
    { document: "ticket", fields: [1, null] },
    ["passage", { index: 1 }],
    [{ role: "user", content: "Please refund the duplicate charge." }],
    null,
  ])("serializes mixed questions and structured state (%j)", async (state) => {
    const result = { model: "jev", answers, usage: { input_tokens: 10, output_tokens: 3 } };
    const client = new TypeSafe.TypeSafeClient({
      apiKey: "test",
      fetch: async () => Response.json(result),
    });
    await client.systemOne({ state, questions });
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    const attributes = spans[0].attributes;
    expect(JSON.parse(String(attributes["input.value"]))).toEqual({
      state,
      questions: JSON.parse(JSON.stringify(questions)),
      model: "jev-latest",
    });
    expect(JSON.parse(String(attributes["output.value"]))).toEqual(result);
    expect(
      Object.keys(attributes).some(
        (key) => key.startsWith("llm.input_messages") || key.startsWith("llm.output_messages"),
      ),
    ).toBe(false);
    expect(JSON.parse(String(attributes.metadata)).typesafe.questions).toEqual({
      category: { type: "choice", confidence: 0 },
      urgent: { type: "noul" },
      severity: { type: "score", confidence: 0.9 },
    });
    expect(attributes["llm.token_count.total"]).toBe(13);
  });
});
