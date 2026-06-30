import type { Tracer } from "@opentelemetry/api";
import type { StreamChunk } from "@tanstack/ai";

import type {
  OISpan,
  ToolCall as OpenInferenceToolCall,
  TraceConfigOptions,
} from "@arizeai/openinference-core";

/**
 * Configuration for the TanStack AI middleware.
 *
 * `tracer` lets consumers opt into a specific OTel tracer while still using
 * the shared OpenInference span shaping in this package.
 */
export type OpenInferenceTanStackAIMiddlewareOptions = {
  tracer?: Tracer;
  traceConfig?: TraceConfigOptions;
};

export type UsageInfo = NonNullable<Extract<StreamChunk, { type: "RUN_FINISHED" }>["usage"]>;

export type CurrentLLMSpanState = {
  span: OISpan;
  outputText: string;
  outputToolCalls: Map<string, OpenInferenceToolCall>;
};

export type RequestState = {
  chatSpan: OISpan;
  toolSpans: Map<string, OISpan>;
  currentLLMSpan?: CurrentLLMSpanState;
};
