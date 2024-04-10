import { BaseTracer, Run } from "@langchain/core/tracers/base";
import {
  Tracer,
  SpanKind,
  Span,
  context,
  trace,
  SpanStatusCode,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import {
  safelyFlattenAttributes,
  safelyFormatIO,
  safelyFormatInputMessages,
  safelyFormatOutputMessages,
  safelyGetOpenInferenceSpanKindFromRunType,
} from "./utils";

type RunWithSpan = {
  run: Run;
  span: Span;
};

export class LangChainTracer extends BaseTracer {
  private tracer: Tracer;
  private runs: Record<string, RunWithSpan | undefined> = {};
  constructor(tracer: Tracer) {
    super();
    this.tracer = tracer;
  }
  name: string = "OpenInferenceLangChainTracer";
  protected persistRun(_run: Run): Promise<void> {
    return Promise.resolve();
  }

  protected async _startTrace(run: Run) {
    await super._startTrace(run);
    if (isTracingSuppressed(context.active())) {
      return;
    }

    let activeContext = context.active();
    const parentCtx = this.getParentSpanContext(run);
    if (parentCtx != null) {
      activeContext = trace.setSpanContext(context.active(), parentCtx);
    }

    const span = this.tracer.startSpan(
      run.name,
      {
        kind: SpanKind.INTERNAL,
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            safelyGetOpenInferenceSpanKindFromRunType(run.run_type) ??
            undefined,
        },
      },
      activeContext,
    );

    this.runs[run.id] = { run, span };
  }
  protected async _endTrace(run: Run) {
    await super._endTrace(run);
    if (isTracingSuppressed(context.active())) {
      return;
    }
    const runWithSpan = this.runs[run.id];
    if (!runWithSpan) {
      return;
    }
    const { span } = runWithSpan;
    if (run.error != null) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: run.error,
      });
    } else {
      span.setStatus({ code: SpanStatusCode.OK });
    }

    const attributes = safelyFlattenAttributes({
      ...safelyFormatIO({ io: run.inputs, type: "input" }),
      ...safelyFormatIO({ io: run.outputs, type: "output" }),
      ...safelyFormatInputMessages(run.inputs),
      ...safelyFormatOutputMessages(run.outputs),
    });
    if (attributes != null) {
      span.setAttributes(attributes);
    }

    runWithSpan.span.end();
  }

  private getParentSpanContext(run: Run) {
    if (run.parent_run_id == null) {
      return;
    }
    const maybeParent = this.runs[run.parent_run_id];
    if (maybeParent == null) {
      return;
    }

    return maybeParent.span.spanContext();
  }
}
