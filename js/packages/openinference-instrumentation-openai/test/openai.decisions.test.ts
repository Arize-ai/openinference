import { context, SpanStatusCode } from "@opentelemetry/api";
import { suppressTracing } from "@opentelemetry/core";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import OpenAI, { APIPromise } from "openai";
import type { Decision, DecisionCreateParams } from "openai/resources/decisions";
import { vi } from "vitest";

import { setMetadata, setSession, setTags, setUser } from "@arizeai/openinference-core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { OpenAIInstrumentation } from "../src";
import { mockAPIPromise } from "./mockAPIPromise";

const memoryExporter = new InMemorySpanExporter();

const usage = {
  input_tokens: 42,
  input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
  output_tokens: 3,
  output_tokens_details: { reasoning_tokens: 0 },
  total_tokens: 45,
} satisfies Decision["usage"];

const predicateRequest = {
  model: "gpt-6-luna",
  input: "I have asked for a refund twice now and nobody has replied.",
  questions: [
    {
      type: "predicate",
      name: "wants_escalation",
      instructions: "Does the customer want their case escalated?",
    },
  ],
} satisfies DecisionCreateParams;

const predicateResponse = {
  model: "gpt-6-luna-2026-10-01",
  answers: [{ type: "predicate", name: "wants_escalation", probability: 0.87 }],
  usage,
} satisfies Decision;

describe("OpenAIInstrumentation - Decisions", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new OpenAIInstrumentation();
  instrumentation.disable();
  let openai: OpenAI;

  instrumentation.setTracerProvider(tracerProvider);
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = OpenAI;

  beforeAll(() => {
    instrumentation.enable();
    openai = new OpenAI({
      apiKey: "fake-api-key",
    });
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("creates a DECISION span for a predicate question", async () => {
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => predicateResponse));

    const result = await openai.decisions.create(predicateRequest);

    expect(result).toEqual(predicateResponse);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Decisions");
    expect(span.status.code).toBe(SpanStatusCode.OK);
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "decision.model_name": "gpt-6-luna-2026-10-01",
        "decision.provider": "openai",
        "decision.request.model_name": "gpt-6-luna",
        "decision.response.model_name": "gpt-6-luna-2026-10-01",
        "decision.system": "openai",
        "decision.token_count.input": 42,
        "decision.token_count.output": 3,
        "input.mime_type": "application/json",
        "input.value": "{"model":"gpt-6-luna","input":"I have asked for a refund twice now and nobody has replied.","questions":[{"type":"predicate","name":"wants_escalation","instructions":"Does the customer want their case escalated?"}]}",
        "openinference.span.kind": "DECISION",
        "output.mime_type": "application/json",
        "output.value": "{"model":"gpt-6-luna-2026-10-01","answers":[{"type":"predicate","name":"wants_escalation","probability":0.87}],"usage":{"input_tokens":42,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":45}}",
      }
    `);
  });

  it("does not emit llm.* attributes on decision spans", async () => {
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => predicateResponse));

    await openai.decisions.create(predicateRequest);

    const [span] = memoryExporter.getFinishedSpans();
    const llmKeys = Object.keys(span.attributes).filter((key) => key.startsWith("llm."));
    expect(llmKeys).toEqual([]);
  });

  it("creates a DECISION span for a choice question", async () => {
    const response = {
      model: "gpt-6-luna",
      answers: [
        {
          type: "choice",
          name: "route",
          choice: "billing",
          confidence: 0.91,
          probabilities: [
            { value: "billing", probability: 0.91 },
            { value: "technical", probability: 0.07 },
            { value: false, probability: 0.02 },
          ],
        },
      ],
      usage,
    } satisfies Decision;
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => response));

    await openai.decisions.create({
      model: "gpt-6-luna",
      input: "My invoice shows a charge I do not recognise.",
      questions: [
        {
          type: "choice",
          name: "route",
          instructions: "Which team should handle this message?",
          choices: [
            { value: "billing", description: "Invoices, charges and refunds" },
            { value: "technical", description: "Bugs and outages" },
            { value: false, description: "No team needs to act" },
          ],
        },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Decisions");
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "decision.model_name": "gpt-6-luna",
        "decision.provider": "openai",
        "decision.request.model_name": "gpt-6-luna",
        "decision.response.model_name": "gpt-6-luna",
        "decision.system": "openai",
        "decision.token_count.input": 42,
        "decision.token_count.output": 3,
        "input.mime_type": "application/json",
        "input.value": "{"model":"gpt-6-luna","input":"My invoice shows a charge I do not recognise.","questions":[{"type":"choice","name":"route","instructions":"Which team should handle this message?","choices":[{"value":"billing","description":"Invoices, charges and refunds"},{"value":"technical","description":"Bugs and outages"},{"value":false,"description":"No team needs to act"}]}]}",
        "openinference.span.kind": "DECISION",
        "output.mime_type": "application/json",
        "output.value": "{"model":"gpt-6-luna","answers":[{"type":"choice","name":"route","choice":"billing","confidence":0.91,"probabilities":[{"value":"billing","probability":0.91},{"value":"technical","probability":0.07},{"value":false,"probability":0.02}]}],"usage":{"input_tokens":42,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":45}}",
      }
    `);
  });

  it("creates a DECISION span for a score question", async () => {
    const response = {
      model: "gpt-6-luna",
      answers: [
        {
          type: "score",
          name: "severity",
          score: 2,
          confidence: 0.64,
          probabilities: [
            { value: 0, label: "low", probability: 0.1 },
            { value: 1, label: "medium", probability: 0.26 },
            { value: 2, label: "high", probability: 0.64 },
          ],
        },
      ],
      usage,
    } satisfies Decision;
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => response));

    await openai.decisions.create({
      model: "gpt-6-luna",
      input: "Production checkout has been returning 500s for an hour.",
      questions: [
        {
          type: "score",
          name: "severity",
          instructions: "How severe is the reported incident?",
          levels: [
            { label: "low", description: "Cosmetic or single-user" },
            { label: "medium", description: "Degraded for some users" },
            { label: "high", description: "Core flow down for everyone" },
          ],
        },
      ],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "decision.model_name": "gpt-6-luna",
        "decision.provider": "openai",
        "decision.request.model_name": "gpt-6-luna",
        "decision.response.model_name": "gpt-6-luna",
        "decision.system": "openai",
        "decision.token_count.input": 42,
        "decision.token_count.output": 3,
        "input.mime_type": "application/json",
        "input.value": "{"model":"gpt-6-luna","input":"Production checkout has been returning 500s for an hour.","questions":[{"type":"score","name":"severity","instructions":"How severe is the reported incident?","levels":[{"label":"low","description":"Cosmetic or single-user"},{"label":"medium","description":"Degraded for some users"},{"label":"high","description":"Core flow down for everyone"}]}]}",
        "openinference.span.kind": "DECISION",
        "output.mime_type": "application/json",
        "output.value": "{"model":"gpt-6-luna","answers":[{"type":"score","name":"severity","score":2,"confidence":0.64,"probabilities":[{"value":0,"label":"low","probability":0.1},{"value":1,"label":"medium","probability":0.26},{"value":2,"label":"high","probability":0.64}]}],"usage":{"input_tokens":42,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":45}}",
      }
    `);
  });

  it("records every answer of a multi-question request, including refusals", async () => {
    const response = {
      model: "gpt-6-luna",
      answers: [
        { type: "predicate", name: "is_spam", probability: 0.02 },
        {
          type: "choice",
          name: "language",
          choice: "en",
          confidence: 0.99,
          probabilities: [
            { value: "en", probability: 0.99 },
            { value: "de", probability: 0.01 },
          ],
        },
        { type: "refusal", name: "contains_pii" },
        { type: "refusal", name: null },
      ],
      usage: { ...usage, output_tokens: 7, total_tokens: 49 },
    } satisfies Decision;
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => response));

    await openai.decisions.create({
      model: "gpt-6-luna",
      input: [
        { role: "user", content: "Hi, my account number is 12345 and I need help." },
        { role: "user", content: "It is urgent." },
      ],
      questions: [
        { type: "predicate", name: "is_spam", instructions: "Is this message spam?" },
        {
          type: "choice",
          name: "language",
          instructions: "Which language is the message written in?",
          choices: [{ value: "en" }, { value: "de" }],
        },
        {
          type: "predicate",
          name: "contains_pii",
          instructions: "Does the message contain personal data?",
        },
        { type: "predicate", instructions: "Is the message urgent?" },
      ],
      safety_identifier: "end-user-7",
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.status.code).toBe(SpanStatusCode.OK);
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "decision.model_name": "gpt-6-luna",
        "decision.provider": "openai",
        "decision.request.model_name": "gpt-6-luna",
        "decision.response.model_name": "gpt-6-luna",
        "decision.system": "openai",
        "decision.token_count.input": 42,
        "decision.token_count.output": 7,
        "input.mime_type": "application/json",
        "input.value": "{"model":"gpt-6-luna","input":[{"role":"user","content":"Hi, my account number is 12345 and I need help."},{"role":"user","content":"It is urgent."}],"questions":[{"type":"predicate","name":"is_spam","instructions":"Is this message spam?"},{"type":"choice","name":"language","instructions":"Which language is the message written in?","choices":[{"value":"en"},{"value":"de"}]},{"type":"predicate","name":"contains_pii","instructions":"Does the message contain personal data?"},{"type":"predicate","instructions":"Is the message urgent?"}],"safety_identifier":"end-user-7"}",
        "openinference.span.kind": "DECISION",
        "output.mime_type": "application/json",
        "output.value": "{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"is_spam","probability":0.02},{"type":"choice","name":"language","choice":"en","confidence":0.99,"probabilities":[{"value":"en","probability":0.99},{"value":"de","probability":0.01}]},{"type":"refusal","name":"contains_pii"},{"type":"refusal","name":null}],"usage":{"input_tokens":42,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":7,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":49}}",
      }
    `);
  });

  it("captures inline image input", async () => {
    const response = {
      model: "gpt-6-luna",
      answers: [{ type: "predicate", name: "is_cat", probability: 0.98 }],
      usage,
    } satisfies Decision;
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => response));

    await openai.decisions.create({
      model: "gpt-6-luna",
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: "What animal is in this picture?" },
            {
              type: "input_image",
              image_url:
                "data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==",
              detail: "low",
            },
          ],
        },
      ],
      questions: [{ type: "predicate", name: "is_cat", instructions: "Is the animal a cat?" }],
    });

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "decision.model_name": "gpt-6-luna",
        "decision.provider": "openai",
        "decision.request.model_name": "gpt-6-luna",
        "decision.response.model_name": "gpt-6-luna",
        "decision.system": "openai",
        "decision.token_count.input": 42,
        "decision.token_count.output": 3,
        "input.mime_type": "application/json",
        "input.value": "{"model":"gpt-6-luna","input":[{"role":"user","content":[{"type":"input_text","text":"What animal is in this picture?"},{"type":"input_image","image_url":"data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==","detail":"low"}]}],"questions":[{"type":"predicate","name":"is_cat","instructions":"Is the animal a cat?"}]}",
        "openinference.span.kind": "DECISION",
        "output.mime_type": "application/json",
        "output.value": "{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"is_cat","probability":0.98}],"usage":{"input_tokens":42,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":45}}",
      }
    `);
  });

  it("records the error and ends the span when the request fails", async () => {
    vi.spyOn(openai, "post").mockImplementation(
      () =>
        new APIPromise(
          openai,
          Promise.reject(new Error("decisions endpoint unavailable")),
          () => undefined as never,
        ),
    );

    await expect(openai.decisions.create(predicateRequest)).rejects.toThrow(
      "decisions endpoint unavailable",
    );

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Decisions");
    expect(span.status).toEqual({
      code: SpanStatusCode.ERROR,
      message: "decisions endpoint unavailable",
    });
    expect(span.events.map((event) => event.name)).toEqual(["exception"]);
    // The request attributes are still recorded, the response ones never arrive.
    expect(span.attributes[SemanticConventions.DECISION_REQUEST_MODEL_NAME]).toBe("gpt-6-luna");
    expect(span.attributes[SemanticConventions.INPUT_VALUE]).toBe(JSON.stringify(predicateRequest));
    expect(span.attributes[SemanticConventions.OUTPUT_VALUE]).toBeUndefined();
    expect(span.attributes[SemanticConventions.DECISION_RESPONSE_MODEL_NAME]).toBeUndefined();
    expect(span.attributes[SemanticConventions.DECISION_TOKEN_COUNT_INPUT]).toBeUndefined();
  });

  it("does not emit a span if tracing is suppressed", async () => {
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => predicateResponse));

    const result = await context.with(suppressTracing(context.active()), () =>
      openai.decisions.create(predicateRequest),
    );

    expect(result).toEqual(predicateResponse);
    expect(memoryExporter.getFinishedSpans().length).toBe(0);
  });

  it("propagates context attributes onto decision spans", async () => {
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => predicateResponse));

    await context.with(
      setTags(
        setMetadata(
          setUser(setSession(context.active(), { sessionId: "session-id" }), {
            userId: "user-id",
          }),
          { tenant: "acme", attempt: 2 },
        ),
        ["support", "triage"],
      ),
      () => openai.decisions.create(predicateRequest),
    );

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const { attributes } = spans[0];
    expect(attributes[SemanticConventions.SESSION_ID]).toBe("session-id");
    expect(attributes[SemanticConventions.USER_ID]).toBe("user-id");
    expect(attributes[SemanticConventions.METADATA]).toBe(
      JSON.stringify({ tenant: "acme", attempt: 2 }),
    );
    expect(attributes[SemanticConventions.TAG_TAGS]).toBe(JSON.stringify(["support", "triage"]));
    expect(attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe("DECISION");
  });
});

describe("OpenAIInstrumentation - Decisions with TraceConfig", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new OpenAIInstrumentation({
    traceConfig: { hideInputs: true, hideOutputs: true },
  });
  instrumentation.disable();
  let openai: OpenAI;

  instrumentation.setTracerProvider(tracerProvider);
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = OpenAI;

  beforeAll(() => {
    instrumentation.enable();
    openai = new OpenAI({
      apiKey: "fake-api-key",
    });
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("masks input.value and output.value but keeps the decision attributes", async () => {
    vi.spyOn(openai, "post").mockImplementation(mockAPIPromise(openai, () => predicateResponse));

    await openai.decisions.create(predicateRequest);

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Decisions");
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "decision.model_name": "gpt-6-luna-2026-10-01",
        "decision.provider": "openai",
        "decision.request.model_name": "gpt-6-luna",
        "decision.response.model_name": "gpt-6-luna-2026-10-01",
        "decision.system": "openai",
        "decision.token_count.input": 42,
        "decision.token_count.output": 3,
        "input.value": "__REDACTED__",
        "openinference.span.kind": "DECISION",
        "output.value": "__REDACTED__",
      }
    `);
  });
});
