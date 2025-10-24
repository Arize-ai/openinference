import {
  SemanticConventions,
  OpenInferenceSpanKind,
  MimeType,
  OUTPUT_MIME_TYPE,
} from "@arizeai/openinference-semantic-conventions";
import { Attributes, type Tracer, trace } from "@opentelemetry/api";
import { OITracer } from "../trace";
import { safelyJSONStringify } from "../utils";

const DEFAULT_TRACER_NAME = "openinference-core";

const { OPENINFERENCE_SPAN_KIND, INPUT_VALUE, OUTPUT_VALUE, INPUT_MIME_TYPE } =
  SemanticConventions;

export interface TraceDecoratorOptions {
  /**
   * An optional name for the span. If not provided, the name if the decorated function will be used
   */
  name?: string;
  /**
   * An optional tracer to be used for tracing the chain operation. If not provided, the global tracer will be used.
   */
  tracer?: Tracer;
  /**
   * A callback function that will be used to set the input attribute of the span from the arguments of the function.
   */
  processInput?:
    | ((args: unknown) => string)
    | ((args: unknown[]) => SpanInput | undefined);
  /**
   * A callback function that will be used to set the output attribute of the span from the result of the function.
   */
  processOutput?: (result: unknown) => string;
}

export type SpanInput = {
  /**
   * the textual representation of the input
   */
  value: string;
  /**
   * the MIME type of the input
   */
  mimeType: MimeType;
};

export type SpanOutput = {
  /**
   * the textual representation of the output
   */
  value: string;
  /**
   * the MIME type of the output
   */
  mimeType: MimeType;
};

/**
 * A decorator factory for tracing chain operations in an LLM application.
 * @experimental
 */
export function chain<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Fn extends (...args: any[]) => any,
>(options: TraceDecoratorOptions = {}) {
  const {
    tracer: _tracer,
    name = "chain",
    processInput: _processInput,
    processOutput: _processOutput,
  } = options;
  const tracer: OITracer = _tracer ? wrapTracer(_tracer) : getTracer();
  const processInput = _processInput ?? defaultProcessInput;
  const processOutput = _processOutput ?? defaultProcessOutput;
  // TODO: infer the name from the target
  return function (originalMethod: Fn, _ctx: ClassMethodDecoratorContext) {
    const wrappedFn = function (this: unknown, ...args: unknown[]) {
      const input = processInput(args);
      return tracer.startActiveSpan(
        name,
        {
          attributes: {
            [OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
            ...toInputAttributes(input),
          },
        },
        (span) => {
          const result = originalMethod.apply(this, args);
          span.setAttributes({
            ...toOutputAttributes(processOutput(result)),
          });
          // TODO: set the status of the span based on the result
          span.end();
          return result;
        },
      );
    };
    return wrappedFn;
  };
}

/**
 * The default input processor that safely JSON stringifies the arguments.
 * @param args The arguments to process
 * @returns The safely JSON stringified arguments
 */
function defaultProcessInput(args: unknown[]): SpanInput | undefined {
  if (args.length === 1) {
    const value = args[0];
    if (typeof value === "string") {
      return {
        value,
        mimeType: MimeType.TEXT,
      };
    }
    return {
      value: safelyJSONStringify(value) ?? "{}",
      mimeType: MimeType.JSON,
    };
  }
  const value = safelyJSONStringify(args);
  if (value == null) {
    return undefined;
  }
  return {
    value,
    mimeType: MimeType.JSON,
  };
}

function defaultProcessOutput(result: unknown): SpanOutput | undefined {
  if (result == null) {
    return undefined;
  }
  if (typeof result === "string") {
    return {
      value: result,
      mimeType: MimeType.TEXT,
    };
  }
  return {
    value: safelyJSONStringify(result) ?? "{}",
    mimeType: MimeType.JSON,
  };
}
/**
 * A helper function to convert a SpanOutput to OpenTelemetry attributes.
 * @param output The SpanOutput to convert
 * @returns The OpenTelemetry attributes
 */
function toOutputAttributes(
  output: SpanOutput | string | undefined,
): Attributes {
  if (output == null) {
    return {};
  }
  if (typeof output === "string") {
    return {
      [OUTPUT_VALUE]: output,
      [OUTPUT_MIME_TYPE]: MimeType.TEXT,
    };
  }
  return {
    [OUTPUT_VALUE]: output.value,
    [OUTPUT_MIME_TYPE]: output.mimeType,
  };
}
function toInputAttributes(input: SpanInput | string | undefined): Attributes {
  if (input == null) {
    return {};
  }
  if (typeof input === "string") {
    return {
      [INPUT_VALUE]: input,
      [INPUT_MIME_TYPE]: MimeType.TEXT,
    };
  }
  return {
    [INPUT_VALUE]: input.value,
    [INPUT_MIME_TYPE]: input.mimeType,
  };
}
/**
 * A function that ensures the tracer is wrapped in an OITracer if necessary.
 * @param tracer The tracer to wrap if necessary
 * @returns
 */
function wrapTracer(tracer: Tracer) {
  if (tracer instanceof OITracer) {
    return tracer;
  }
  return new OITracer({
    tracer,
  });
}
/**
 * A helper function to get a tracer for the decorators.
 */
function getTracer() {
  return new OITracer({
    tracer: trace.getTracer(DEFAULT_TRACER_NAME),
  });
}
