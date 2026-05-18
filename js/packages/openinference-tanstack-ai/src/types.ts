import type { StreamChunk } from "@tanstack/ai";

import type { OISpan, ToolCall as OpenInferenceToolCall } from "@arizeai/openinference-core";

export type { OpenInferenceTanStackAIMiddlewareOptions } from "./index";

export type UsageInfo = NonNullable<Extract<StreamChunk, { type: "RUN_FINISHED" }>["usage"]>;

export type CurrentLLMSpanState = {
  span: OISpan;
  outputText: string;
  outputToolCalls: Map<string, OpenInferenceToolCall>;
};
