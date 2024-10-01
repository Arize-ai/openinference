import { SESSION_ID } from "@arizeai/openinference-semantic-conventions";
import { OITracer, REDACTED_VALUE, setSession } from "../../src";
import { OISpan } from "../../src/trace/trace-config/OISpan";
import {
  context,
  ContextManager,
  Span,
  SpanKind,
  SpanOptions,
  Tracer,
} from "@opentelemetry/api";
import { AsyncHooksContextManager } from "@opentelemetry/context-async-hooks";

describe("OITracer", () => {
  let mockTracer: jest.Mocked<Tracer>;
  let mockSpan: jest.Mocked<Span>;
  let contextManager: ContextManager;

  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
    mockSpan = {
      setAttribute: jest.fn(),
      setAttributes: jest.fn(),
    } as unknown as jest.Mocked<Span>;

    mockTracer = {
      startSpan: jest.fn(() => mockSpan),
      startActiveSpan: jest.fn((name, options, context, fn) => {
        return fn(mockSpan);
      }),
    } as unknown as jest.Mocked<Tracer>;
  });

  afterEach(() => {
    context.disable();
  });

  describe("startSpan", () => {
    it("should create an OISpan with start span and set attributes according to the trace config", () => {
      const oiTracer = new OITracer({
        tracer: mockTracer,
        traceConfig: {
          hideInputs: true,
        },
      });
      const name = "test-span";
      const options = {
        attributes: { key1: "value1", "input.value": "sensitiveValue" },
      };

      const span = oiTracer.startSpan(name, options);

      expect(mockTracer.startSpan).toHaveBeenCalledWith(
        name,
        { attributes: undefined },
        context.active(),
      );
      expect(mockSpan.setAttributes).toHaveBeenCalledWith({
        key1: "value1",
        "input.value": REDACTED_VALUE,
      });

      expect(span).toBeInstanceOf(OISpan);
    });

    it("should add OpenInference context attributes to spans", () => {
      const oiTracer = new OITracer({
        tracer: mockTracer,
        traceConfig: {
          hideInputs: true,
        },
      });
      const name = "test-span";
      const options = {
        attributes: { key1: "value1", "input.value": "sensitiveValue" },
      };
      context.with(
        setSession(context.active(), { sessionId: "my-session-id" }),
        () => {
          const span = oiTracer.startSpan(name, options, context.active());

          expect(mockTracer.startSpan).toHaveBeenCalledWith(
            name,
            { attributes: undefined },
            context.active(),
          );
          expect(mockSpan.setAttributes).toHaveBeenCalledWith({
            key1: "value1",
            [SESSION_ID]: "my-session-id",
            "input.value": REDACTED_VALUE,
          });

          expect(span).toBeInstanceOf(OISpan);
        },
      );
    });
    it("should correctly nest spans", () => {});
  });

  describe("startActiveSpan", () => {
    it("should create an OISpan with startActiveSpan and set attributes according to the trace config", () => {
      const oiTracer = new OITracer({
        tracer: mockTracer,
        traceConfig: {
          hideInputs: true,
        },
      });
      const name = "test-span";
      const options = {
        attributes: { key1: "value1", "input.value": "sensitiveValue" },
      };

      const span = oiTracer.startActiveSpan(name, options, (span) => {
        return span;
      });

      expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
        name,
        {
          attributes: undefined,
        },
        expect.any(Object),
        expect.any(Function),
      );
      expect(mockSpan.setAttributes).toHaveBeenCalledWith({
        key1: "value1",
        "input.value": REDACTED_VALUE,
      });

      expect(span).toBeInstanceOf(OISpan);
    });

    it("should handle overloads correctly", () => {
      const oiTracer = new OITracer({
        tracer: mockTracer,
        traceConfig: {
          hideInputs: true,
        },
      });
      const name = "test-span";
      const mockFn = jest.fn();

      oiTracer.startActiveSpan(name, mockFn);
      expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
        name,
        { attributes: undefined },
        context.active(),
        expect.any(Function),
      );

      const options: SpanOptions = {
        kind: SpanKind.INTERNAL,
        attributes: { key: "value" },
      };
      oiTracer.startActiveSpan(name, options, mockFn);
      expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
        name,
        { kind: SpanKind.INTERNAL, attributes: undefined },
        context.active(),
        expect.any(Function),
      );

      const newContext = context.active().setValue(Symbol("test"), "test");

      oiTracer.startActiveSpan(name, options, newContext, mockFn);
      expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
        name,
        { kind: SpanKind.INTERNAL, attributes: undefined },
        newContext,
        expect.any(Function),
      );
    });

    it("should add OpenInference context attributes to spans", () => {
      const oiTracer = new OITracer({
        tracer: mockTracer,
        traceConfig: {
          hideInputs: true,
        },
      });
      const name = "test-span";
      const options = {
        attributes: { key1: "value1", "input.value": "sensitiveValue" },
      };
      context.with(
        setSession(context.active(), { sessionId: "my-session-id" }),
        () => {
          const span = oiTracer.startActiveSpan(name, options, (span) => span);

          expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
            name,
            {
              attributes: undefined,
            },
            context.active(),
            expect.any(Function),
          );
          expect(mockSpan.setAttributes).toHaveBeenCalledWith({
            key1: "value1",
            [SESSION_ID]: "my-session-id",
            "input.value": REDACTED_VALUE,
          });

          expect(span).toBeInstanceOf(OISpan);
        },
      );
    });
  });
});
