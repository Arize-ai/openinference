import { OITracer } from "@arizeai/openinference-core";
import { FrameworkSpan, GeneratedResponse } from "../types";
import { SpanStatusCode, TimeInput } from "@opentelemetry/api";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { startEventName, successEventName } from "../config";

interface BuiltTraceTreeProps {
  tracer: OITracer;
  mainSpanKind: OpenInferenceSpanKind;
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
    .filter((beeaiSpan) => beeaiSpan.parent_id === data.parentId)
    .forEach((beeaiSpan) => {
      tracer.startActiveSpan(
        beeaiSpan.context.span_id,
        {
          // custom start time
          startTime: beeaiSpan.start_time,
          // set span important attributes
          attributes: {
            target: beeaiSpan.attributes.target,
            name: beeaiSpan.name,
            traceId: data.traceId,
            ...(beeaiSpan.attributes.metadata && {
              metadata: JSON.stringify(beeaiSpan.attributes.metadata),
            }),
            ...(beeaiSpan.attributes.data && { ...beeaiSpan.attributes.data }),
          },
        },
        (activeSpan) => {
          // set status
          activeSpan.setStatus(beeaiSpan.status);

          // set nested spans
          buildSpansForParent({
            tracer,
            data: {
              spans: data.spans,
              traceId: data.traceId,
              parentId: beeaiSpan.context.span_id,
            },
          });

          // finish the span
          activeSpan.end(beeaiSpan.end_time);
        },
      );
    });
}

function buildAgentMainSpanData(data: BuiltTraceTreeProps["data"]) {
  return {
    ...(data.prompt && { [SemanticConventions.INPUT_VALUE]: data.prompt }),
    ...(data.generatedMessage !== undefined && {
      [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(data.generatedMessage),
    }),
    ...(data.history && { history: JSON.stringify(data.history) }),
  };
}

function buildToolMainSpanData(data: BuiltTraceTreeProps["data"]) {
  const startBeeaiSpan = data.spans.find(
    (span) => span.name === startEventName,
  );
  const successBeeeaiSpan = data.spans.find(
    (span) => span.name === successEventName,
  );

  return {
    ...(startBeeaiSpan && {
      [SemanticConventions.INPUT_VALUE]:
        startBeeaiSpan.attributes.data?.[SemanticConventions.TOOL_PARAMETERS],
      [SemanticConventions.TOOL_PARAMETERS]:
        startBeeaiSpan.attributes.data?.[SemanticConventions.TOOL_PARAMETERS],
    }),
    ...(successBeeeaiSpan && {
      [SemanticConventions.OUTPUT_VALUE]:
        successBeeeaiSpan.attributes.data?.[SemanticConventions.OUTPUT_VALUE],
    }),
  };
}

function buildLLMMainSpanData(data: BuiltTraceTreeProps["data"]) {
  const startBeeaiSpan = data.spans.find(
    (span) => span.name === startEventName,
  );
  const successBeeeaiSpan = data.spans.find(
    (span) => span.name === successEventName,
  );

  if (!startBeeaiSpan && !successBeeeaiSpan) return {};

  const provider =
    startBeeaiSpan?.attributes.data?.[SemanticConventions.LLM_PROVIDER] ||
    successBeeeaiSpan?.attributes.data?.[SemanticConventions.LLM_PROVIDER];

  const modelName =
    startBeeaiSpan?.attributes.data?.[SemanticConventions.LLM_MODEL_NAME] ||
    successBeeeaiSpan?.attributes.data?.[SemanticConventions.LLM_MODEL_NAME];

  return {
    ...(startBeeaiSpan && {
      [SemanticConventions.INPUT_VALUE]:
        startBeeaiSpan?.attributes.data?.[SemanticConventions.INPUT_VALUE],
      [SemanticConventions.INPUT_MIME_TYPE]:
        startBeeaiSpan?.attributes.data?.[SemanticConventions.INPUT_MIME_TYPE],
    }),
    ...(successBeeeaiSpan && {
      [SemanticConventions.OUTPUT_MIME_TYPE]:
        successBeeeaiSpan.attributes.data?.[
          SemanticConventions.OUTPUT_MIME_TYPE
        ],
      [SemanticConventions.OUTPUT_VALUE]:
        successBeeeaiSpan.attributes.data?.[SemanticConventions.OUTPUT_VALUE],
    }),
    ...(provider && { [SemanticConventions.LLM_PROVIDER]: provider }),
    ...(modelName && { [SemanticConventions.LLM_MODEL_NAME]: modelName }),
  };
}

export function buildTraceTree({
  tracer,
  data,
  mainSpanKind,
}: BuiltTraceTreeProps) {
  const computedData =
    mainSpanKind === OpenInferenceSpanKind.AGENT
      ? buildAgentMainSpanData(data)
      : mainSpanKind === OpenInferenceSpanKind.TOOL
        ? buildToolMainSpanData(data)
        : mainSpanKind === OpenInferenceSpanKind.LLM
          ? buildLLMMainSpanData(data)
          : {};

  tracer.startActiveSpan(
    `beeai-framework-main`,
    {
      // custom start time
      startTime: data.startTime,
      // set main span important attributes
      attributes: {
        source: data.source,
        traceId: data.traceId,
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: mainSpanKind,
        ["beeai.version"]: data.version,
        ...computedData,
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
