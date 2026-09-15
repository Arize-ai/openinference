import { SpanStatusCode } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { OITracer } from "@arizeai/openinference-core";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { ToolSpanTracker, mergeHooks } from "../src/hookInjector";

describe("ToolSpanTracker", () => {
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;
  let oiTracer: OITracer;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    oiTracer = new OITracer({
      tracer: provider.getTracer("test"),
    });
  });

  afterEach(() => {
    exporter.reset();
  });

  it("should create and end a tool span successfully", () => {
    const tracker = new ToolSpanTracker(oiTracer);

    tracker.startToolSpan("Bash", { command: "ls" }, "tool-1");
    tracker.endToolSpan("tool-1", { output: "file1.txt\nfile2.txt" });

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.name).toBe("Bash");
    expect(span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.TOOL,
    );
    expect(span.attributes[SemanticConventions.TOOL_NAME]).toBe("Bash");
    expect(span.attributes[SemanticConventions.INPUT_VALUE]).toBe(
      JSON.stringify({ command: "ls" }),
    );
    expect(span.attributes[SemanticConventions.OUTPUT_VALUE]).toBe(
      JSON.stringify({ output: "file1.txt\nfile2.txt" }),
    );
    expect(span.status.code).toBe(SpanStatusCode.OK);
  });

  it("should end a tool span with error", () => {
    const tracker = new ToolSpanTracker(oiTracer);

    tracker.startToolSpan("Bash", { command: "rm -rf /" }, "tool-2");
    tracker.endToolSpanWithError("tool-2", "Permission denied");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0];
    expect(span.status.code).toBe(SpanStatusCode.ERROR);
    expect(span.status.message).toBe("Permission denied");
    expect(span.events).toHaveLength(1);
    expect(span.events[0].name).toBe("exception");
  });

  it("should handle endToolSpan for unknown tool_use_id gracefully", () => {
    const tracker = new ToolSpanTracker(oiTracer);

    // Should not throw
    tracker.endToolSpan("nonexistent");
    tracker.endToolSpanWithError("nonexistent", "error");

    expect(exporter.getFinishedSpans()).toHaveLength(0);
  });

  it("should end all in-flight spans on cleanup", () => {
    const tracker = new ToolSpanTracker(oiTracer);

    tracker.startToolSpan("Read", { path: "/a" }, "tool-a");
    tracker.startToolSpan("Write", { path: "/b" }, "tool-b");

    tracker.endAllInFlight();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(2);
    for (const span of spans) {
      expect(span.status.code).toBe(SpanStatusCode.ERROR);
      expect(span.status.message).toBe("Abandoned");
    }
  });
});

describe("mergeHooks", () => {
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;
  let oiTracer: OITracer;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    oiTracer = new OITracer({
      tracer: provider.getTracer("test"),
    });
  });

  afterEach(() => {
    exporter.reset();
  });

  it("should create hooks when options is undefined", () => {
    const tracker = new ToolSpanTracker(oiTracer);
    const parentSpan = oiTracer.startSpan("parent");

    const result = mergeHooks({ options: undefined, toolTracker: tracker, parentSpan });

    expect(result).toBeDefined();
    const hooks = result.hooks as Record<string, unknown[]>;
    expect(hooks.PreToolUse).toHaveLength(1);
    expect(hooks.PostToolUse).toHaveLength(1);
    expect(hooks.PostToolUseFailure).toHaveLength(1);

    parentSpan.end();
  });

  it("should preserve existing user hooks", () => {
    const tracker = new ToolSpanTracker(oiTracer);
    const parentSpan = oiTracer.startSpan("parent");

    const userHook = {
      hooks: [async () => ({})],
    };

    const options = {
      hooks: {
        PreToolUse: [userHook],
      },
    };

    const result = mergeHooks({ options, toolTracker: tracker, parentSpan });

    const hooks = result.hooks as Record<string, unknown[]>;
    // User hook + our hook
    expect(hooks.PreToolUse).toHaveLength(2);
    expect(hooks.PreToolUse[0]).toBe(userHook);

    parentSpan.end();
  });

  it("should create tool spans via hook callbacks", async () => {
    const tracker = new ToolSpanTracker(oiTracer);
    const parentSpan = oiTracer.startSpan("parent");

    const result = mergeHooks({ options: {}, toolTracker: tracker, parentSpan });

    const hooks = result.hooks as Record<
      string,
      Array<{
        hooks: Array<
          (
            input: Record<string, unknown>,
            toolUseID: string | undefined,
            options: { signal: AbortSignal },
          ) => Promise<Record<string, unknown>>
        >;
      }>
    >;

    // Simulate PreToolUse
    const preToolUseCallback = hooks.PreToolUse[0].hooks[0];
    await preToolUseCallback(
      {
        hook_event_name: "PreToolUse",
        tool_name: "Bash",
        tool_input: { command: "echo hi" },
        tool_use_id: "tu-1",
        session_id: "s1",
        transcript_path: "/tmp",
        cwd: "/",
      },
      "tu-1",
      { signal: new AbortController().signal },
    );

    // Simulate PostToolUse
    const postToolUseCallback = hooks.PostToolUse[0].hooks[0];
    await postToolUseCallback(
      {
        hook_event_name: "PostToolUse",
        tool_name: "Bash",
        tool_input: { command: "echo hi" },
        tool_response: "hi\n",
        tool_use_id: "tu-1",
        session_id: "s1",
        transcript_path: "/tmp",
        cwd: "/",
      },
      "tu-1",
      { signal: new AbortController().signal },
    );

    parentSpan.end();

    const spans = exporter.getFinishedSpans();
    // Tool span + parent span
    const toolSpans = spans.filter(
      (s) =>
        s.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] === OpenInferenceSpanKind.TOOL,
    );
    expect(toolSpans).toHaveLength(1);
    expect(toolSpans[0].name).toBe("Bash");
    expect(toolSpans[0].status.code).toBe(SpanStatusCode.OK);
  });
});

describe("ToolSpanTracker non-object toolInput", () => {
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;
  let oiTracer: OITracer;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    oiTracer = new OITracer({
      tracer: provider.getTracer("test"),
    });
  });

  afterEach(() => {
    exporter.reset();
  });

  it("should handle string toolInput", () => {
    const tracker = new ToolSpanTracker(oiTracer);

    tracker.startToolSpan("Bash", "echo hello", "tool-str");
    tracker.endToolSpan("tool-str");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes[SemanticConventions.INPUT_VALUE]).toBe('"echo hello"');
  });

  it("should handle array toolInput", () => {
    const tracker = new ToolSpanTracker(oiTracer);

    tracker.startToolSpan("Multi", ["a", "b"], "tool-arr");
    tracker.endToolSpan("tool-arr");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes[SemanticConventions.INPUT_VALUE]).toBe('["a","b"]');
  });

  it("should handle null toolInput", () => {
    const tracker = new ToolSpanTracker(oiTracer);

    tracker.startToolSpan("Empty", null, "tool-null");
    tracker.endToolSpan("tool-null");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes[SemanticConventions.INPUT_VALUE]).toBe("null");
  });

  it("should handle undefined toolInput", () => {
    const tracker = new ToolSpanTracker(oiTracer);

    tracker.startToolSpan("Empty", undefined, "tool-undef");
    tracker.endToolSpan("tool-undef");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
  });
});

describe("Hook callbacks error handling", () => {
  let provider: NodeTracerProvider;
  let exporter: InMemorySpanExporter;
  let oiTracer: OITracer;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
    oiTracer = new OITracer({
      tracer: provider.getTracer("test"),
    });
  });

  afterEach(() => {
    exporter.reset();
  });

  it("should not throw when startToolSpan fails", async () => {
    const tracker = new ToolSpanTracker(oiTracer);
    vi.spyOn(tracker, "startToolSpan").mockImplementation(() => {
      throw new Error("Span creation failed");
    });

    const parentSpan = oiTracer.startSpan("parent");
    const result = mergeHooks({ options: {}, toolTracker: tracker, parentSpan });

    const hooks = result.hooks as Record<
      string,
      Array<{
        hooks: Array<
          (
            input: Record<string, unknown>,
            toolUseID: string | undefined,
            options: { signal: AbortSignal },
          ) => Promise<Record<string, unknown>>
        >;
      }>
    >;

    // Should not throw
    const preToolUseCallback = hooks.PreToolUse[0].hooks[0];
    const output = await preToolUseCallback(
      {
        hook_event_name: "PreToolUse",
        tool_name: "Bash",
        tool_input: {},
        tool_use_id: "tu-err",
        session_id: "s1",
        transcript_path: "/tmp",
        cwd: "/",
      },
      "tu-err",
      { signal: new AbortController().signal },
    );

    expect(output).toEqual({});
    parentSpan.end();
  });

  it("should not throw when endToolSpan fails", async () => {
    const tracker = new ToolSpanTracker(oiTracer);
    vi.spyOn(tracker, "endToolSpan").mockImplementation(() => {
      throw new Error("Span end failed");
    });

    const parentSpan = oiTracer.startSpan("parent");
    const result = mergeHooks({ options: {}, toolTracker: tracker, parentSpan });

    const hooks = result.hooks as Record<
      string,
      Array<{
        hooks: Array<
          (
            input: Record<string, unknown>,
            toolUseID: string | undefined,
            options: { signal: AbortSignal },
          ) => Promise<Record<string, unknown>>
        >;
      }>
    >;

    const postToolUseCallback = hooks.PostToolUse[0].hooks[0];
    const output = await postToolUseCallback(
      {
        hook_event_name: "PostToolUse",
        tool_name: "Bash",
        tool_input: {},
        tool_response: "ok",
        tool_use_id: "tu-err",
        session_id: "s1",
        transcript_path: "/tmp",
        cwd: "/",
      },
      "tu-err",
      { signal: new AbortController().signal },
    );

    expect(output).toEqual({});
    parentSpan.end();
  });

  it("should not throw when endToolSpanWithError fails", async () => {
    const tracker = new ToolSpanTracker(oiTracer);
    vi.spyOn(tracker, "endToolSpanWithError").mockImplementation(() => {
      throw new Error("Span error end failed");
    });

    const parentSpan = oiTracer.startSpan("parent");
    const result = mergeHooks({ options: {}, toolTracker: tracker, parentSpan });

    const hooks = result.hooks as Record<
      string,
      Array<{
        hooks: Array<
          (
            input: Record<string, unknown>,
            toolUseID: string | undefined,
            options: { signal: AbortSignal },
          ) => Promise<Record<string, unknown>>
        >;
      }>
    >;

    const postToolUseFailureCallback = hooks.PostToolUseFailure[0].hooks[0];
    const output = await postToolUseFailureCallback(
      {
        hook_event_name: "PostToolUseFailure",
        tool_name: "Bash",
        tool_input: {},
        tool_use_id: "tu-err",
        error: "tool failed",
        session_id: "s1",
        transcript_path: "/tmp",
        cwd: "/",
      },
      "tu-err",
      { signal: new AbortController().signal },
    );

    expect(output).toEqual({});
    parentSpan.end();
  });
});
