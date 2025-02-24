import { OITracer } from "@arizeai/openinference-core";
import { FrameworkSpan, GeneratedResponse } from "../types.js";
import { SpanStatusCode, TimeInput } from "@opentelemetry/api";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

interface BuiltTraceTreeProps {
  tracer: OITracer;
  data: {
    prompt?: string | null;
    history: GeneratedResponse[] | undefined;
    generatedMessage: GeneratedResponse | undefined;
    spans: FrameworkSpan[];
    traceId: string;
    version: string;
    runErrorSpanKey: string;
    startTime: TimeInput;
    endTime: TimeInput;
    source: string;
  };
}

interface BuildSpansForParentProps {
  tracer: OITracer;
  data: {
    spans: FrameworkSpan[];
    traceId: string;
    parentId: string | undefined;
  };
}

function buildSpansForParent({ tracer, data }: BuildSpansForParentProps) {
  data.spans
    .filter((fwSpan) => fwSpan.parent_id === data.parentId)
    .forEach((fwSpan) => {
      tracer.startActiveSpan(
        fwSpan.context.span_id,
        {
          // custom start time
          startTime: fwSpan.start_time,
          // set span important attributes
          attributes: {
            target: fwSpan.attributes.target,
            name: fwSpan.name,
            traceId: data.traceId,
            ...(fwSpan.attributes.metadata && {
              metadata: JSON.stringify(fwSpan.attributes.metadata),
            }),
            ...(fwSpan.attributes.data && { ...fwSpan.attributes.data }),
          },
        },
        (activeSpan) => {
          // set status
          activeSpan.setStatus(fwSpan.status);

          // set nested spans
          buildSpansForParent({
            tracer,
            data: {
              spans: data.spans,
              traceId: data.traceId,
              parentId: fwSpan.context.span_id,
            },
          });

          // finish the span
          activeSpan.end(fwSpan.end_time);
        },
      );
    });
}

export function buildTraceTree({ tracer, data }: BuiltTraceTreeProps) {
  tracer.startActiveSpan(
    `beeai-framework-${data.source}-${data.traceId}`,
    {
      // custom start time
      startTime: data.startTime,
      // set main span important attributes
      attributes: {
        traceId: data.traceId,
        version: data.version,
        ...(data.prompt && { [SemanticConventions.INPUT_VALUE]: data.prompt }),
        ...(data.generatedMessage !== undefined && {
          [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(
            data.generatedMessage,
          ),
        }),
        ...(data.history && { history: JSON.stringify(data.history) }),
      },
    },
    (activeSpan) => {
      // set status
      const runErrorSpan = data.spans.find(
        (span) => span.attributes.target === data.runErrorSpanKey,
      );
      if (runErrorSpan) {
        activeSpan.setStatus(runErrorSpan.status);
      } else {
        activeSpan.setStatus({ code: SpanStatusCode.OK });
      }

      // set nested spans
      buildSpansForParent({
        tracer,
        data: { spans: data.spans, traceId: data.traceId, parentId: undefined },
      });

      // finish the main span with custom end time
      activeSpan.end(data.endTime);
    },
  );
}
