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
  safelyFormatFunctionCalls,
  safelyFormatIO,
  safelyFormatInputMessages,
  safelyFormatLLMParams,
  safelyFormatMetadata,
  safelyFormatOutputMessages,
  safelyFormatPromptTemplate,
  safelyFormatRetrievalDocuments,
  safelyFormatTokenCounts,
  safelyFormatToolCalls,
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

  /**
   * Called when a new run is created on v0.1.0 of langchain see {@link BaseTracer}
   * @param run the langchain {@link Run} object
   *
   * This method is only available on langchain ^0.1.0 BaseTracer and has been replaced in 0.2 by onRunCreate
   * we support both 0.1 and 0.2 so we need to check if the method exists on the super class before calling it
   */
  protected async _startTrace(run: Run) {
    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-ignore
    if (typeof super._startTrace === "function") {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore
      await super._startTrace(run);
    }
    await this.startTracing(run);
  }

  /**
   * Called when a new run is created on v0.2.0 of langchain see {@link BaseTracer}
   * @param run the langchain {@link Run} object
   *
   * This method is only available on the langchain ^0.2.0 {@link BaseTracer}
   */
  async onRunCreate(run: Run) {
    if (typeof super.onRunCreate === "function") {
      await super.onRunCreate(run);
    }
    await this.startTracing(run);
  }

  async startTracing(run: Run) {
    if (isTracingSuppressed(context.active())) {
      return;
    }

    /**
     * If the parent span context is available, use it as the active context for the new span.
     * This will allow the new span to be a child of the parent span.
     */
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
      ...safelyFormatIO({ io: run.inputs, ioType: "input" }),
      ...safelyFormatIO({ io: run.outputs, ioType: "output" }),
      ...safelyFormatInputMessages(run.inputs),
      ...safelyFormatOutputMessages(run.outputs),
      ...safelyFormatRetrievalDocuments(run),
      ...safelyFormatLLMParams(run.extra),
      ...safelyFormatPromptTemplate(run),
      ...safelyFormatTokenCounts(run.outputs),
      ...safelyFormatFunctionCalls(run.outputs),
      ...safelyFormatToolCalls(run),
      ...safelyFormatMetadata(run),
    });
    if (attributes != null) {
      span.setAttributes(attributes);
    }

    runWithSpan.span.end();
    delete this.runs[run.id];
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
