/* eslint-disable no-console */
import { BaseTracer, Run } from "@langchain/core/tracers/base";
import { Tracer, SpanKind, Span } from "@opentelemetry/api";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

type RunWithSpan = {
  run: Run;
  span: Span;
};

export class LangChainTracer extends BaseTracer {
  tracer: Tracer;
  runs: Record<string, RunWithSpan | undefined> = {};
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
    const span = this.tracer.startSpan(run.name, {
      kind: SpanKind.INTERNAL,
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.LLM,
      },
    });
    this.runs[run.id] = { run, span };
  }
  protected async _endTrace(run: Run) {
    await super._endTrace(run);
    const runWithSpan = this.runs[run.id];
    console.log("test--", runWithSpan);
    if (runWithSpan) {
      runWithSpan.span.end();
    }
  }
}
