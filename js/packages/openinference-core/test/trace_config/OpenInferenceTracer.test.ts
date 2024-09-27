import { OpenInferenceTracer, REDACTED_VALUE } from "@core/trace";
import { OpenInferenceSpan } from "@core/trace/trace_config/OpenInferenceSpan";
import {
  context,
  Context,
  ContextAPI,
  Span,
  SpanKind,
  SpanOptions,
  Tracer,
} from "@opentelemetry/api";

describe("OpenInferenceTracer", () => {
  let mockTracer: jest.Mocked<Tracer>;
  let mockSpan: jest.Mocked<Span>;

  beforeEach(() => {
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

  describe("startSpan", () => {
    it("should create an OpenInferenceSpan with start span and set attributes according to the trace config", () => {
      const openInferenceTracer = new OpenInferenceTracer({
        tracer: mockTracer,
        traceConfig: {
          hideInputs: true,
        },
      });
      const name = "test-span";
      const options = {
        attributes: { key1: "value1", "input.value": "sensitiveValue" },
      };

      const span = openInferenceTracer.startSpan(name, options);

      expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
        name,
        { attributes: undefined },
        undefined,
      );
      expect(mockSpan.setAttributes).toHaveBeenCalledWith({
        key1: "value1",
        "input.value": REDACTED_VALUE,
      });

      expect(span).toBeInstanceOf(OpenInferenceSpan);
    });
  });

  describe("startActiveSpan", () => {
    it("should create an OpenInferenceSpan with startActiveSpan and set attributes according to the trace config", () => {
      const openInferenceTracer = new OpenInferenceTracer({
        tracer: mockTracer,
        traceConfig: {
          hideInputs: true,
        },
      });
      const name = "test-span";
      const options = {
        attributes: { key1: "value1", "input.value": "sensitiveValue" },
      };

      const span = openInferenceTracer.startActiveSpan(
        name,
        options,
        (span) => {
          return span;
        },
      );

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

      expect(span).toBeInstanceOf(OpenInferenceSpan);
    });
  });
  it("should handle overloads correctly", () => {
    const openInferenceTracer = new OpenInferenceTracer({
      tracer: mockTracer,
      traceConfig: {
        hideInputs: true,
      },
    });
    const name = "test-span";
    const mockFn = jest.fn();

    openInferenceTracer.startActiveSpan(name, mockFn);
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
    openInferenceTracer.startActiveSpan(name, options, mockFn);
    expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
      name,
      { kind: SpanKind.INTERNAL, attributes: undefined },
      context.active(),
      expect.any(Function),
    );

    const newContext = context.active().setValue(Symbol("test"), "test");

    openInferenceTracer.startActiveSpan(name, options, newContext, mockFn);
    expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
      name,
      { kind: SpanKind.INTERNAL, attributes: undefined },
      newContext,
      expect.any(Function),
    );
  });
});

//   describe("startActiveSpan", () => {
//     it("should create an OpenInferenceSpan and set attributes", () => {
//       const name = "test-active-span";
//       const options: SpanOptions = {
//         attributes: { key1: "value1", sensitiveKey: "sensitiveValue" },
//       };
//       const mockContext = {} as Context;
//       const mockFn = jest.fn();

//       openInferenceTracer.startActiveSpan(name, options, mockContext, mockFn);

//       expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
//         name,
//         { ...options, attributes: undefined },
//         mockContext,
//         expect.any(Function),
//       );
//       expect(OpenInferenceSpan).toHaveBeenCalledWith({
//         span: mockSpan,
//         config: expect.any(Object),
//       });
//       expect(mockFn).toHaveBeenCalled();
//     });

//     it("should handle overloads correctly", () => {
//       const name = "test-overload";
//       const mockFn = jest.fn();

//       openInferenceTracer.startActiveSpan(name, mockFn);
//       expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
//         name,
//         {},
//         expect.any(Object),
//         expect.any(Function),
//       );

//       const options: SpanOptions = { attributes: { key: "value" } };
//       openInferenceTracer.startActiveSpan(name, options, mockFn);
//       expect(mockTracer.startActiveSpan).toHaveBeenCalledWith(
//         name,
//         { ...options, attributes: undefined },
//         expect.any(Object),
//         expect.any(Function),
//       );
//     });
//   });
