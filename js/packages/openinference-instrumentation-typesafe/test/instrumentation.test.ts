// Prototype methods are compared by identity, never called without a receiver.
/* oxlint-disable typescript/unbound-method */
import { context, ROOT_CONTEXT, SpanKind, SpanStatusCode, trace } from "@opentelemetry/api";
import { AsyncLocalStorageContextManager } from "@opentelemetry/context-async-hooks";
import { suppressTracing } from "@opentelemetry/core";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import * as TypeSafe from "@typesafe-ai/sdk";
import { afterEach, beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest";

import type { TraceConfigOptions } from "@arizeai/openinference-core";
import {
  REDACTED_VALUE,
  setMetadata,
  setSession,
  setTags,
  setUser,
} from "@arizeai/openinference-core";

import { TypeSafeInstrumentation } from "../src";

const request = {
  state: "I was charged twice. Please fix this ASAP.",
  questions: {
    category: TypeSafe.choice("What is this ticket about?", { billing: null, technical: null }),
  },
};
const result: TypeSafe.SystemOneResult<typeof request.questions> = {
  model: "jev-resolved",
  answers: {
    category: {
      type: "choice",
      choice: "billing",
      confidence: 0.95,
      probabilities: { billing: 0.95, technical: 0.05 },
    },
  },
  usage: { input_tokens: 42, output_tokens: 7 },
};

describe("TypeSafeInstrumentation", () => {
  let exporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;
  let instrumentation: TypeSafeInstrumentation;

  function configure(traceConfig?: TraceConfigOptions) {
    instrumentation?.disable();
    instrumentation = new TypeSafeInstrumentation({ tracerProvider: provider, traceConfig });
    instrumentation.manuallyInstrument(TypeSafe);
  }

  function makeClient(body: unknown = result, init?: ResponseInit) {
    const fetch = vi.fn<TypeSafe.Fetch>(async () =>
      Response.json(body, {
        headers: { "x-typesafe-request-id": "req_test" },
        ...init,
      }),
    );
    return {
      client: new TypeSafe.TypeSafeClient({
        apiKey: "test-key",
        fetch,
        defaultModel: "jev-default",
        retry: { maxRetries: 0 },
        logLevel: "off",
      }),
      fetch,
    };
  }

  beforeEach(() => {
    context.setGlobalContextManager(new AsyncLocalStorageContextManager().enable());
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({ spanProcessors: [new SimpleSpanProcessor(exporter)] });
    configure();
  });

  afterEach(async () => {
    instrumentation.disable();
    await provider.shutdown();
    context.disable();
    vi.restoreAllMocks();
    vi.unstubAllEnvs();
  });

  it("records one complete LLM span and leaves the request and response intact", async () => {
    const { client, fetch } = makeClient();
    const options = { timeout: 5000, retry: { maxRetries: 1, httpStatuses: new Set([429]) } };
    const response = await client.systemOne(request, options);
    expectTypeOf(response.answers.category.choice).toEqualTypeOf<"billing" | "technical">();
    expect(response).toEqual(result);
    expect(fetch).toHaveBeenCalledOnce();
    expect(fetch.mock.calls[0][0]).toBe("https://api.typesafe.ai/v1/systemone");
    const sentBody = fetch.mock.calls[0][1]?.body;
    expect(typeof sentBody).toBe("string");
    expect(JSON.parse(sentBody as string)).toEqual({ ...request, model: "jev-default" });
    expect(request).not.toHaveProperty("model");
    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].name).toBe("TypeSafeClient.systemOne");
    expect(spans[0].kind).toBe(SpanKind.CLIENT);
    expect(spans[0].status).toEqual({ code: SpanStatusCode.OK });
    expect(spans[0].attributes).toEqual({
      "openinference.span.kind": "LLM",
      "llm.provider": "typesafe",
      "llm.system": "typesafe",
      "llm.model_name": "jev-resolved",
      "llm.invocation_parameters": JSON.stringify({
        model: "jev-default",
        timeout: 5000,
        retry: { maxRetries: 1, httpStatuses: [429] },
      }),
      "input.value": JSON.stringify({ ...request, model: "jev-default" }),
      "input.mime_type": "application/json",
      "output.value": JSON.stringify(result),
      "output.mime_type": "application/json",
      "llm.input_messages.0.message.role": "user",
      "llm.input_messages.0.message.content": request.state,
      "llm.output_messages.0.message.role": "assistant",
      "llm.output_messages.0.message.content": JSON.stringify(result.answers),
      "llm.token_count.prompt": 42,
      "llm.token_count.completion": 7,
      "llm.token_count.total": 49,
      metadata: JSON.stringify({
        typesafe: {
          request_id: "req_test",
          questions: { category: { type: "choice", confidence: 0.95 } },
        },
      }),
    });
  });

  it.each([undefined, "jev-override"])(
    "falls back to request/client model when the response omits it (%s)",
    async (model) => {
      const { client } = makeClient({ ...result, model: undefined });
      await client.systemOne({ ...request, model });
      expect(exporter.getFinishedSpans()[0].attributes["llm.model_name"]).toBe(
        model ?? "jev-default",
      );
    },
  );

  it.each([
    [undefined, undefined, undefined, undefined],
    [{}, undefined, undefined, undefined],
    [{ input_tokens: 0 }, 0, undefined, undefined],
    [{ output_tokens: 5 }, undefined, 5, undefined],
    [{ input_tokens: null, output_tokens: 5 }, undefined, 5, undefined],
    [{ input_tokens: 0, output_tokens: 0 }, 0, 0, 0],
  ])(
    "handles partial usage %j without inventing token counts",
    async (usage, prompt, completion, total) => {
      await makeClient({ ...result, usage }).client.systemOne(request);
      const attributes = exporter.getFinishedSpans()[0].attributes;
      expect(attributes["llm.token_count.prompt"]).toBe(prompt);
      expect(attributes["llm.token_count.completion"]).toBe(completion);
      expect(attributes["llm.token_count.total"]).toBe(total);
    },
  );

  it("preserves APIPromise, withResponse, map, and a single cached SDK parse", async () => {
    const { client, fetch } = makeClient();
    const promise = client.systemOne(request);
    expect(promise).toBeInstanceOf(TypeSafe.APIPromise);
    expect(promise).toBeInstanceOf(Promise);
    const mapped = promise.map((data) => data.answers.category.choice);
    expect(mapped).toBeInstanceOf(TypeSafe.APIPromise);
    const [first, second, withResponse, mappedResponse] = await Promise.all([
      promise,
      promise.then((data) => data),
      promise.withResponse(),
      mapped.withResponse(),
    ]);
    expect(first).toBe(second);
    expect(withResponse.data).toBe(first);
    expect(withResponse.requestId).toBe("req_test");
    expect(withResponse.response.bodyUsed).toBe(true);
    expect(mappedResponse.data).toBe("billing");
    expect(mappedResponse.response).toBe(withResponse.response);
    expect(mappedResponse.requestId).toBe("req_test");
    expect(fetch).toHaveBeenCalledOnce();
    expect(exporter.getFinishedSpans()).toHaveLength(1);
    expect(exporter.getFinishedSpans()[0].attributes["output.value"]).toBe(JSON.stringify(result));
  });

  it.each([false, true])(
    "preserves an unread raw response, including after map (%s)",
    async (map) => {
      const { client } = makeClient();
      const promise = client.systemOne(request);
      const response = await (map ? promise.map((data) => data.answers) : promise).asResponse();
      expect(response.bodyUsed).toBe(false);
      expect(response.headers.get("x-typesafe-request-id")).toBe("req_test");
      expect(await response.json()).toEqual(result);
      expect(exporter.getFinishedSpans()).toHaveLength(1);
      expect(exporter.getFinishedSpans()[0].attributes["llm.token_count.total"]).toBe(49);
    },
  );

  it("supports then/catch/finally without marking consumer errors as SDK failures", async () => {
    const onFinally = vi.fn();
    const value = await makeClient()
      .client.systemOne(request)
      .then(() => {
        throw new Error("consumer failure");
      })
      .catch(() => "recovered")
      .finally(onFinally);
    expect(value).toBe("recovered");
    expect(onFinally).toHaveBeenCalledOnce();
    expect(exporter.getFinishedSpans()[0].status.code).toBe(SpanStatusCode.OK);
  });

  it("ends a span even if the caller never consumes a successful APIPromise", async () => {
    void makeClient().client.systemOne(request);
    await vi.waitFor(() => expect(exporter.getFinishedSpans()).toHaveLength(1));
  });

  it.each(["await", "withResponse", "asResponse", "map", "catch"] as const)(
    "records API errors via %s",
    async (mode) => {
      const { client } = makeClient(
        { message: "not allowed" },
        { status: 401, headers: { "x-typesafe-request-id": "req_error" } },
      );
      const promise = client.systemOne(request);
      const consumed =
        mode === "withResponse"
          ? promise.withResponse()
          : mode === "asResponse"
            ? promise.asResponse()
            : mode === "map"
              ? promise.map((data) => data.answers)
              : promise;
      const error =
        mode === "catch"
          ? await promise.catch((e: unknown) => e)
          : await consumed.then(
              () => undefined,
              (e: unknown) => e,
            );
      expect(error).toBeInstanceOf(TypeSafe.AuthenticationError);
      const spans = exporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      expect(spans[0].status).toEqual({ code: SpanStatusCode.ERROR, message: "401 not allowed" });
      expect(spans[0].events).toEqual([
        expect.objectContaining({
          name: "exception",
          attributes: expect.objectContaining({ "exception.message": "401 not allowed" }),
        }),
      ]);
      expect(spans[0].attributes["http.response.status_code"]).toBe(401);
      expect(JSON.parse(String(spans[0].attributes.metadata)).typesafe.request_id).toBe(
        "req_error",
      );
    },
  );

  it("preserves synchronous validation errors and ends their spans", () => {
    const { client, fetch } = makeClient();
    expect(() => client.systemOne({ state: "text", questions: {} })).toThrow(
      TypeSafe.TypeSafeError,
    );
    expect(fetch).not.toHaveBeenCalled();
    expect(exporter.getFinishedSpans()).toHaveLength(1);
    expect(exporter.getFinishedSpans()[0].status.code).toBe(SpanStatusCode.ERROR);
    expect(exporter.getFinishedSpans()[0].events[0].name).toBe("exception");
  });

  it.each(["abort", "timeout", "connection"] as const)(
    "ends spans for %s errors",
    async (failure) => {
      const controller = new AbortController();
      const fetch: TypeSafe.Fetch = async (_url, init) => {
        if (failure === "connection") throw new Error("offline");
        return new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), {
            once: true,
          });
        });
      };
      const client = new TypeSafe.TypeSafeClient({
        apiKey: "test",
        fetch,
        retry: { maxRetries: 0 },
        logLevel: "off",
      });
      const promise = client.systemOne(request, { signal: controller.signal, timeout: 10 });
      if (failure === "abort") controller.abort();
      await expect(promise).rejects.toBeInstanceOf(
        failure === "abort"
          ? TypeSafe.APIUserAbortError
          : failure === "timeout"
            ? TypeSafe.APITimeoutError
            : TypeSafe.APIConnectionError,
      );
      expect(exporter.getFinishedSpans()).toHaveLength(1);
      expect(exporter.getFinishedSpans()[0].status.code).toBe(SpanStatusCode.ERROR);
    },
  );

  it("includes SDK retries in one span", async () => {
    const { client, fetch } = makeClient();
    fetch.mockResolvedValueOnce(Response.json({ message: "retry" }, { status: 429 }));
    await client.systemOne(request, { retry: { maxRetries: 1, backoffInitialMs: 0 } });
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(exporter.getFinishedSpans()).toHaveLength(1);
    expect(exporter.getFinishedSpans()[0].status.code).toBe(SpanStatusCode.OK);
  });

  it("suppresses tracing without changing APIPromise behavior", async () => {
    const { client } = makeClient();
    const promise = context.with(suppressTracing(context.active()), () =>
      client.systemOne(request),
    );
    expect(promise).toBeInstanceOf(TypeSafe.APIPromise);
    expect((await promise.withResponse()).data).toEqual(result);
    expect(exporter.getFinishedSpans()).toHaveLength(0);
  });

  it("propagates caller context, merges metadata, and activates the span during fetch", async () => {
    const { client, fetch } = makeClient();
    let activeSpanId: string | undefined;
    fetch.mockImplementation(async () => {
      activeSpanId = trace.getSpan(context.active())?.spanContext().spanId;
      return Response.json(result);
    });
    const parent = provider.getTracer("test").startSpan("parent");
    let ctx = trace.setSpan(ROOT_CONTEXT, parent);
    ctx = setSession(ctx, { sessionId: "session-1" });
    ctx = setUser(ctx, { userId: "user-1" });
    ctx = setTags(ctx, ["routing"]);
    ctx = setMetadata(ctx, { application: "support", nested: { keep: true } });
    await context.with(ctx, async () => {
      await client.systemOne(request);
      expect(trace.getSpan(context.active())).toBe(parent);
    });
    const span = exporter.getFinishedSpans()[0];
    expect(span.parentSpanContext?.spanId).toBe(parent.spanContext().spanId);
    expect(span.spanContext().spanId).toBe(activeSpanId);
    expect(span.attributes).toMatchObject({
      "session.id": "session-1",
      "user.id": "user-1",
      "tag.tags": JSON.stringify(["routing"]),
    });
    expect(JSON.parse(String(span.attributes.metadata))).toMatchObject({
      application: "support",
      nested: { keep: true },
      typesafe: { questions: { category: { confidence: 0.95 } } },
    });
    parent.end();
  });

  it("keeps concurrent calls in their own parent contexts", async () => {
    const { client } = makeClient();
    const parents = [
      provider.getTracer("test").startSpan("a"),
      provider.getTracer("test").startSpan("b"),
    ];
    await Promise.all(
      parents.map((parent) =>
        context.with(trace.setSpan(ROOT_CONTEXT, parent), () => client.systemOne(request)),
      ),
    );
    expect(
      new Set(exporter.getFinishedSpans().map((span) => span.parentSpanContext?.spanId)),
    ).toEqual(new Set(parents.map((span) => span.spanContext().spanId)));
    parents.forEach((span) => span.end());
  });

  it("redacts inputs without copying state, instructions, or question names into other attributes", async () => {
    configure({ hideInputs: true });
    const secretRequest = {
      state: "secret-state",
      questions: { "secret-question": TypeSafe.noul("secret-instructions") },
    };
    await makeClient({ model: "m", answers: {}, usage: {} }).client.systemOne(secretRequest);
    const attributes = exporter.getFinishedSpans()[0].attributes;
    expect(attributes["input.value"]).toBe(REDACTED_VALUE);
    expect(attributes["input.mime_type"]).toBeUndefined();
    expect(attributes["llm.input_messages.0.message.content"]).toBeUndefined();
    expect(JSON.stringify(attributes)).not.toContain("secret-");
  });

  it("respects environment-based masking", async () => {
    vi.stubEnv("OPENINFERENCE_HIDE_INPUTS", "true");
    configure();
    await makeClient().client.systemOne(request);
    expect(exporter.getFinishedSpans()[0].attributes["input.value"]).toBe(REDACTED_VALUE);
    expect(
      JSON.parse(String(exporter.getFinishedSpans()[0].attributes.metadata)).typesafe.questions,
    ).toBeUndefined();
  });

  it("hides output values, messages, and confidence metadata", async () => {
    configure({ hideOutputs: true });
    await makeClient().client.systemOne(request);
    const attributes = exporter.getFinishedSpans()[0].attributes;
    expect(attributes["output.value"]).toBe(REDACTED_VALUE);
    expect(attributes["llm.output_messages.0.message.content"]).toBeUndefined();
    expect(JSON.stringify(attributes)).not.toContain("confidence");
    expect(attributes["llm.token_count.total"]).toBe(49);
  });

  it("supports message-only masking independently from full input/output", async () => {
    configure({ hideInputMessages: true, hideOutputMessages: true });
    await makeClient().client.systemOne(request);
    const attributes = exporter.getFinishedSpans()[0].attributes;
    expect(
      Object.keys(attributes).some(
        (key) => key.startsWith("llm.input_messages") || key.startsWith("llm.output_messages"),
      ),
    ).toBe(false);
    expect(JSON.parse(String(attributes["input.value"])).state).toBe(request.state);
    expect(attributes["output.value"]).toBe(JSON.stringify(result));
  });

  it("does not collect headers or signals", async () => {
    const { client, fetch } = makeClient();
    await client.systemOne(request, {
      headers: { "x-custom-secret": "credential-value" },
      signal: new AbortController().signal,
    });
    expect(new Headers(fetch.mock.calls[0][1]?.headers).get("x-custom-secret")).toBe(
      "credential-value",
    );
    const serialized = JSON.stringify(exporter.getFinishedSpans()[0].attributes);
    expect(serialized).not.toContain("credential-value");
    expect(serialized).not.toContain("test-key");
    expect(serialized).not.toContain("signal");
  });

  it("restores the original method on disable and reapplies manual instrumentation on enable", async () => {
    instrumentation.disable();
    const original = TypeSafe.TypeSafeClient.prototype.systemOne;
    instrumentation.enable();
    const patched = TypeSafe.TypeSafeClient.prototype.systemOne;
    expect(patched).not.toBe(original);
    instrumentation.manuallyInstrument(TypeSafe);
    expect(TypeSafe.TypeSafeClient.prototype.systemOne).toBe(patched);
    await makeClient().client.systemOne(request);
    instrumentation.disable();
    expect(TypeSafe.TypeSafeClient.prototype.systemOne).toBe(original);
    await makeClient().client.systemOne(request);
    expect(exporter.getFinishedSpans()).toHaveLength(1);
    instrumentation.enable();
    await makeClient().client.systemOne(request);
    expect(exporter.getFinishedSpans()).toHaveLength(2);
  });

  it("does not patch a manually registered module until enabled", async () => {
    instrumentation.disable();
    instrumentation = new TypeSafeInstrumentation({
      tracerProvider: provider,
      instrumentationConfig: { enabled: false },
    });
    instrumentation.manuallyInstrument(TypeSafe);
    await makeClient().client.systemOne(request);
    expect(exporter.getFinishedSpans()).toHaveLength(0);
    instrumentation.enable();
    await makeClient().client.systemOne(request);
    expect(exporter.getFinishedSpans()).toHaveLength(1);
  });

  it("does not let a second instrumentation double-wrap or unpatch the owner", async () => {
    const second = new TypeSafeInstrumentation({ tracerProvider: provider });
    try {
      second.manuallyInstrument(TypeSafe);
      second.disable();
      await makeClient().client.systemOne(request);
      expect(exporter.getFinishedSpans()).toHaveLength(1);
    } finally {
      second.disable();
    }
  });

  it("uses a tracer provider assigned after construction", async () => {
    const secondExporter = new InMemorySpanExporter();
    const secondProvider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(secondExporter)],
    });
    try {
      instrumentation.setTracerProvider(secondProvider);
      await makeClient().client.systemOne(request);
      expect(exporter.getFinishedSpans()).toHaveLength(0);
      expect(secondExporter.getFinishedSpans()).toHaveLength(1);
    } finally {
      await secondProvider.shutdown();
    }
  });

  it("leaves models.list uninstrumented", async () => {
    const { client } = makeClient({ models: [] });
    await expect(client.models.list()).resolves.toEqual([]);
    expect(exporter.getFinishedSpans()).toHaveLength(0);
  });

  it("does not change non-JSON success responses", async () => {
    const { client, fetch } = makeClient();
    fetch.mockResolvedValueOnce(new Response("plain text", { status: 200 }));
    await expect(client.systemOne(request)).resolves.toBe("plain text");
    expect(exporter.getFinishedSpans()).toHaveLength(1);
    expect(exporter.getFinishedSpans()[0].status.code).toBe(SpanStatusCode.OK);
  });

  it("does not turn telemetry extraction failures into SDK failures", async () => {
    // The SDK parses with text(); only our cloned response calls json().
    vi.spyOn(Response.prototype, "json").mockRejectedValueOnce(new Error("telemetry failure"));
    await expect(makeClient().client.systemOne(request)).resolves.toEqual(result);
    expect(exporter.getFinishedSpans()).toHaveLength(1);
    expect(exporter.getFinishedSpans()[0].status.code).toBe(SpanStatusCode.OK);
  });

  it("preserves SDK serialization errors for circular inputs", async () => {
    const state: Record<string, TypeSafe.JsonValue> = {};
    state.self = state;
    await expect(makeClient().client.systemOne({ ...request, state })).rejects.toBeInstanceOf(
      TypeError,
    );
    expect(exporter.getFinishedSpans()).toHaveLength(1);
    expect(exporter.getFinishedSpans()[0].status.code).toBe(SpanStatusCode.ERROR);
  });
});
