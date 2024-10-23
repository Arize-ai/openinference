import { addTracerToHandlers } from "../src/instrumentationUtils";
import { LangChainTracer } from "../src/tracer";
import { CallbackManager } from "@langchain/core/callbacks/manager";
import { OITracer } from "@arizeai/openinference-core";

describe("addTracerToHandlers", () => {
  it("should add a tracer if there are no handlers", () => {
    const tracer = {} as OITracer;

    const result = addTracerToHandlers(tracer);

    expect(Array.isArray(result)).toBe(true);
    expect(result).toHaveLength(1);
    if (Array.isArray(result)) {
      expect(result[0]).toBeInstanceOf(LangChainTracer);
    }
  });
  it("should add a handler to a pre-existing array of handlers", () => {
    const tracer = {} as OITracer;
    const handlers = [new CallbackManager()];

    const result = addTracerToHandlers(tracer, handlers);

    expect(result).toBe(handlers);
    expect(result).toHaveLength(2);
    if (Array.isArray(result)) {
      expect(result[1]).toBeInstanceOf(LangChainTracer);
    }
  });
  it("should add a handler to a callback handler class' handlers", () => {
    const tracer = {} as OITracer;
    const handlers = new CallbackManager();

    const result = addTracerToHandlers(tracer, handlers);

    expect(result).toBe(handlers);
    if ("handlers" in result) {
      expect(result.handlers).toHaveLength(1);
      expect(result.inheritableHandlers).toHaveLength(1);
      expect(result.handlers[0]).toBeInstanceOf(LangChainTracer);
      expect(result.inheritableHandlers[0]).toBeInstanceOf(LangChainTracer);
    }
  });

  it("should not add a handler if it already exists in an array of handlers", () => {
    const tracer = {} as OITracer;
    const handlers = [new LangChainTracer(tracer)];

    const result = addTracerToHandlers(tracer, handlers);

    expect(result).toBe(handlers);
    expect(result).toHaveLength(1);
    if (Array.isArray(result)) {
      expect(result[0]).toBeInstanceOf(LangChainTracer);
    }
  });

  it("should not add a handler if it already exists in a callback handler class' handlers", () => {
    const tracer = {} as OITracer;
    const handlers = new CallbackManager();
    handlers.addHandler(new LangChainTracer(tracer));

    const result = addTracerToHandlers(tracer, handlers);

    expect(result).toBe(handlers);
    if ("handlers" in result) {
      expect(result.handlers).toHaveLength(1);
      expect(result.inheritableHandlers).toHaveLength(1);
      expect(result.handlers[0]).toBeInstanceOf(LangChainTracer);
      expect(result.inheritableHandlers[0]).toBeInstanceOf(LangChainTracer);
    }
  });
});
