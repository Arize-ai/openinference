import { context } from "@opentelemetry/api";
import { getSessionId, setSessionId } from "../src/trace/session";
// import {
//   InMemorySpanExporter,
//   SimpleSpanProcessor,
// } from "@opentelemetry/sdk-trace-base";
// import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
// import { suppressTracing } from "@opentelemetry/core";
// import { context } from "@opentelemetry/api";

describe("session", () => {
  it("should set session id in the context", () => {
    context.with(setSessionId(context.active(), "session-id"), () => {
      expect(getSessionId(context.active())).toBe("session-id");
    });
  });
});
