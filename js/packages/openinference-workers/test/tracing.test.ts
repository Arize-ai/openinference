import { AsyncLocalStorage } from "node:async_hooks";

import {
  context,
  createContextKey,
  propagation,
  ROOT_CONTEXT,
  SpanStatusCode,
  trace,
} from "@opentelemetry/api";
import {
  ExportResultCode,
  isTracingSuppressed,
  suppressTracing,
  W3CTraceContextPropagator,
  type ExportResult,
} from "@opentelemetry/core";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  TracerProvider,
} from "@opentelemetry/sdk-trace";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getInputAttributes,
  getOutputAttributes,
  setMetadata,
  setSession,
  setTags,
  setUser,
} from "@arizeai/openinference-core";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

import {
  AsyncLocalStorageContextManager,
  createWorkersTracing,
  extractTraceContext,
  FetchTraceExporter,
  injectTraceHeaders,
  withRequestSpan,
  type RequestSpanOptions,
  type WorkersTracing,
} from "../src";

const instances: WorkersTracing[] = [];
function setup() {
  const memory = new InMemorySpanExporter();
  const tracing = createWorkersTracing({
    endpoint: "https://collector.example/",
    projectName: "workers-regression",
    serviceName: "workers-test",
    register: false,
    fetch: async () => new Response(null, { status: 200 }),
    traceConfig: { hideInputs: true, hideOutputs: true },
    spanProcessors: [new SimpleSpanProcessor({ exporter: memory })],
  });
  instances.push(tracing);
  return { ...tracing, memory };
}
function readableSpans() {
  const exporter = new InMemorySpanExporter();
  const provider = new TracerProvider({ spanProcessors: [new SimpleSpanProcessor({ exporter })] });
  provider.getTracer("fixture").startSpan("fixture").end();
  return exporter.getFinishedSpans();
}
const requestOptions = (t: WorkersTracing): RequestSpanOptions => ({
  tracer: t.tracer,
  request: new Request("https://worker.example/"),
  name: "request",
  kind: OpenInferenceSpanKind.CHAIN,
  flush: t.flush,
});

beforeEach(() => {
  context.disable();
  propagation.disable();
  trace.disable();
  context.setGlobalContextManager(new AsyncLocalStorageContextManager());
  propagation.setGlobalPropagator(new W3CTraceContextPropagator());
});
afterEach(async () => {
  await Promise.all(instances.splice(0).map((t) => t.shutdown()));
  context.disable();
  propagation.disable();
  trace.disable();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("workerd context regressions", () => {
  it("preserves a bound method's receiver, arguments, and async context", async () => {
    const manager = new AsyncLocalStorageContextManager();
    const key = createContextKey("bound");
    const method = manager.bind(
      ROOT_CONTEXT.setValue(key, "bound"),
      async function (this: { value: number }, amount: number) {
        await Promise.resolve();
        return [this.value + amount, manager.active().getValue(key)];
      },
    );
    expect(await method.call({ value: 7 }, 2)).toEqual([9, "bound"]);
    expect(manager.active()).toBe(ROOT_CONTEXT);
  });

  it("disables and re-enables without calling workerd's unsupported disable method", () => {
    const unsupported = vi.spyOn(AsyncLocalStorage.prototype, "disable").mockImplementation(() => {
      throw new Error("not implemented");
    });
    const manager = new AsyncLocalStorageContextManager();
    const key = createContextKey("lifecycle");
    manager.with(ROOT_CONTEXT.setValue(key, 1), () => {
      manager.disable();
      expect(manager.active()).toBe(ROOT_CONTEXT);
      manager.enable();
      expect(manager.active()).toBe(ROOT_CONTEXT);
    });
    expect(manager.with(ROOT_CONTEXT.setValue(key, 2), () => manager.active().getValue(key))).toBe(
      2,
    );
    expect(unsupported).not.toHaveBeenCalled();
  });

  it.each([{}, { TraceParent: "malformed" }])(
    "does not inherit an ambient span for invalid or absent inbound headers: %j",
    async (headers) => {
      const t = setup();
      await t.tracer.startActiveSpan("ambient", async (span) => {
        try {
          const extracted = extractTraceContext({ headers });
          expect(trace.getSpanContext(extracted)).toBeUndefined();
        } finally {
          span.end();
        }
      });
    },
  );

  it("extracts case-insensitive headers and retains local attributes and suppression", () => {
    const ctx = suppressTracing(setSession(ROOT_CONTEXT, { sessionId: "session" }));
    context.with(ctx, () => {
      const extracted = extractTraceContext({
        headers: {
          TraceParent: "00-11111111111111111111111111111111-2222222222222222-00",
          TraceState: "vendor=value",
          unused: undefined,
        },
      });
      expect(trace.getSpanContext(extracted)).toMatchObject({
        traceId: "11111111111111111111111111111111",
        spanId: "2222222222222222",
        traceFlags: 0,
        isRemote: true,
      });
      expect(trace.getSpanContext(extracted)?.traceState?.serialize()).toBe("vendor=value");
      expect(isTracingSuppressed(extracted)).toBe(true);
    });
  });

  it("injects into a copy without duplicate case variants", async () => {
    const t = setup();
    await t.tracer.startActiveSpan("outbound", async (span) => {
      try {
        const original = new Headers({ TraceParent: "stale", "Content-Type": "application/json" });
        const result = injectTraceHeaders({ headers: original });
        expect(result.traceparent).toContain(span.spanContext().traceId);
        expect(result["content-type"]).toBe("application/json");
        expect(original.get("traceparent")).toBe("stale");
      } finally {
        span.end();
      }
    });
  });
});

describe("request lifecycle regressions", () => {
  for (const background of [false, true]) {
    for (const synchronous of [false, true]) {
      it(`preserves results and errors when flush fails (background=${background}, synchronous=${synchronous})`, async () => {
        const t = setup();
        const pending: Promise<unknown>[] = [];
        const flush = synchronous
          ? () => {
              throw new Error("flush");
            }
          : async () => {
              throw new Error("flush");
            };
        const options = {
          ...requestOptions(t),
          flush,
          execution: background
            ? {
                waitUntil: (p: Promise<unknown>) => {
                  pending.push(p);
                },
              }
            : undefined,
        };
        await expect(withRequestSpan(options, async () => "result")).resolves.toBe("result");
        const applicationError = new Error("application");
        await expect(
          withRequestSpan(options, async () => {
            throw applicationError;
          }),
        ).rejects.toBe(applicationError);
        await expect(Promise.all(pending)).resolves.toBeDefined();
        expect(t.memory.getFinishedSpans()).toHaveLength(2);
        expect(t.memory.getFinishedSpans()[1].status).toMatchObject({
          code: SpanStatusCode.ERROR,
          message: "application",
        });
        expect(t.memory.getFinishedSpans()[1].events[0].name).toBe("exception");
      });
    }
  }

  it("preserves an explicit ERROR when the handler returns normally", async () => {
    const t = setup();
    const response = await withRequestSpan(requestOptions(t), async (span) => {
      span.setStatus({ code: SpanStatusCode.ERROR, message: "upstream refused" });
      return new Response(null, { status: 500 });
    });
    expect(response.status).toBe(500);
    expect(t.memory.getFinishedSpans()[0].status).toEqual({
      code: SpanStatusCode.ERROR,
      message: "upstream refused",
    });
  });

  it("does not block a response on waitUntil export work", async () => {
    const t = setup();
    let finishFlush = () => {};
    const flushing = new Promise<void>((resolve) => {
      finishFlush = resolve;
    });
    const pending: Promise<unknown>[] = [];
    await expect(
      withRequestSpan(
        {
          ...requestOptions(t),
          flush: () => flushing,
          execution: {
            waitUntil: (p) => {
              pending.push(p);
            },
          },
        },
        async () => "response",
      ),
    ).resolves.toBe("response");
    expect(pending).toHaveLength(1);
    finishFlush();
    await Promise.all(pending);
  });

  it("suppresses the request and its nested spans without losing the result", async () => {
    const t = setup();
    const flush = vi.fn(t.flush);
    const result = await context.with(suppressTracing(context.active()), () =>
      withRequestSpan({ ...requestOptions(t), flush }, async (span) => {
        expect(span.isRecording()).toBe(false);
        t.tracer.startSpan("nested").end();
        return "result";
      }),
    );
    expect(result).toBe("result");
    expect(t.memory.getFinishedSpans()).toHaveLength(0);
    expect(flush).not.toHaveBeenCalled();
  });

  it("isolates concurrent sessions and masks initial and later attributes", async () => {
    const t = setup();
    await Promise.all(
      ["one", "two"].map((sessionId) => {
        let ctx = setSession(ROOT_CONTEXT, { sessionId });
        ctx = setUser(ctx, { userId: sessionId });
        ctx = setMetadata(ctx, { sessionId });
        ctx = setTags(ctx, [sessionId]);
        return context.with(ctx, () =>
          withRequestSpan(
            { ...requestOptions(t), attributes: getInputAttributes("secret input") },
            async (span) => {
              await Promise.resolve();
              span.setAttributes(getOutputAttributes("secret output"));
              t.tracer.startSpan(`child-${sessionId}`).end();
            },
          ),
        );
      }),
    );
    const spans = t.memory.getFinishedSpans();
    expect(spans).toHaveLength(4);
    for (const sessionId of ["one", "two"]) {
      const parent = spans.find(
        (s) => s.name === "request" && s.attributes["session.id"] === sessionId,
      )!;
      const child = spans.find((s) => s.name === `child-${sessionId}`)!;
      expect(parent.attributes).toMatchObject({
        "input.value": "__REDACTED__",
        "output.value": "__REDACTED__",
        "session.id": sessionId,
        "user.id": sessionId,
        metadata: JSON.stringify({ sessionId }),
        "tag.tags": JSON.stringify([sessionId]),
      });
      expect(child.parentSpanContext?.spanId).toBe(parent.spanContext().spanId);
      expect(child.attributes["session.id"]).toBe(sessionId);
    }
  });
});

describe("fetch exporter regressions", () => {
  it("calls native fetch with the global receiver and suppresses export tracing", async () => {
    const native = vi.fn(function (
      this: unknown,
      _input: Parameters<typeof fetch>[0],
      init?: RequestInit,
    ) {
      expect(this).toBe(globalThis);
      expect(isTracingSuppressed(context.active())).toBe(true);
      expect(init?.body).toBeInstanceOf(Uint8Array);
      expect(new Headers(init?.headers).get("content-type")).toBe("application/x-protobuf");
      return Promise.resolve(new Response(null));
    });
    vi.stubGlobal("fetch", native);
    const exporter = new FetchTraceExporter({ url: "https://collector.example/v1/traces" });
    const callback = vi.fn();
    exporter.export(readableSpans(), callback);
    await exporter.forceFlush();
    expect(callback).toHaveBeenCalledExactlyOnceWith({ code: ExportResultCode.SUCCESS });
    expect(native).toHaveBeenCalledTimes(1);
    await exporter.shutdown();
  });

  it("completes once even when the error and result callbacks throw", async () => {
    const onError = vi.fn(() => {
      throw new Error("error hook");
    });
    const exporter = new FetchTraceExporter({
      url: "https://collector.example/v1/traces",
      fetch: async () => new Response(null, { status: 503 }),
      onError,
    });
    const callback = vi.fn((result: ExportResult) => {
      expect(result.code).toBe(ExportResultCode.FAILED);
      throw new Error("result hook");
    });
    exporter.export(readableSpans(), callback);
    await expect(exporter.forceFlush()).resolves.toBeUndefined();
    expect(onError).toHaveBeenCalledTimes(1);
    expect(callback).toHaveBeenCalledTimes(1);
    await exporter.shutdown();
  });

  it("aborts a hanging collector request within the configured timeout", async () => {
    vi.useFakeTimers();
    const exporter = new FetchTraceExporter({
      url: "https://collector.example/v1/traces",
      timeoutMillis: 200,
      fetch: (_input, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("aborted")), {
            once: true,
          });
        }),
    });
    const callback = vi.fn();
    exporter.export(readableSpans(), callback);
    await vi.advanceTimersByTimeAsync(200);
    await exporter.forceFlush();
    expect(callback).toHaveBeenCalledExactlyOnceWith({
      code: ExportResultCode.FAILED,
      error: expect.objectContaining({ message: "aborted" }),
    });
    await exporter.shutdown();
  });

  it("flushes in-flight exports before shutdown and rejects later exports", async () => {
    let finishResponse: (value: Response) => void = () => {};
    const response = new Promise<Response>((resolve) => {
      finishResponse = resolve;
    });
    const exporter = new FetchTraceExporter({
      url: "https://collector.example/v1/traces",
      fetch: () => response,
    });
    const callback = vi.fn();
    exporter.export(readableSpans(), callback);
    let stopped = false;
    const shutdown = exporter.shutdown().then(() => {
      stopped = true;
    });
    await Promise.resolve();
    expect(stopped).toBe(false);
    finishResponse(new Response(null));
    await shutdown;
    expect(callback).toHaveBeenCalledExactlyOnceWith({ code: ExportResultCode.SUCCESS });
    const later = vi.fn();
    exporter.export(readableSpans(), later);
    await exporter.forceFlush();
    expect(later).toHaveBeenCalledExactlyOnceWith({
      code: ExportResultCode.FAILED,
      error: expect.objectContaining({ message: "exporter is shut down" }),
    });
  });

  it("preserves an existing global provider while returning an independent tracer", async () => {
    const first = setup();
    expect(trace.setGlobalTracerProvider(first.provider)).toBe(true);
    const memory = new InMemorySpanExporter();
    const second = createWorkersTracing({
      endpoint: "https://collector.example",
      projectName: "second",
      serviceName: "second",
      spanProcessors: [new SimpleSpanProcessor({ exporter: memory })],
      fetch: async () => new Response(null),
    });
    instances.push(second);
    trace.getTracer("global").startSpan("first").end();
    second.tracer.startSpan("second").end();
    expect(first.memory.getFinishedSpans().map((s) => s.name)).toEqual(["first"]);
    expect(memory.getFinishedSpans()[0].resource.attributes["openinference.project.name"]).toBe(
      "second",
    );
    expect(memory.getFinishedSpans()[0].instrumentationScope.name).toBe(
      "@arizeai/openinference-workers",
    );
  });
});
