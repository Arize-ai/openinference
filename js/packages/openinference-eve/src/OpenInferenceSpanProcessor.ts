import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

import {
  OpenInferenceBatchSpanProcessor as VercelBatchSpanProcessor,
  OpenInferenceSimpleSpanProcessor as VercelSimpleSpanProcessor,
} from "@arizeai/openinference-vercel";

import { addEveAttributesToSpan } from "./utils";

/**
 * Extends {@link VercelSimpleSpanProcessor} to support Eve AI framework spans.
 *
 * Processes `ai.eve.turn` root spans and propagates Eve session/context
 * attributes into OpenInference conventions before the standard Vercel/GenAI
 * attribute conversion runs.
 *
 * @example
 * ```typescript
 * import { OpenInferenceSimpleSpanProcessor, isOpenInferenceSpan } from "@arizeai/openinference-eve";
 * import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
 *
 * const processor = new OpenInferenceSimpleSpanProcessor({
 *   exporter: new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" }),
 *   spanFilter: isOpenInferenceSpan,
 * });
 * ```
 */
export class OpenInferenceSimpleSpanProcessor extends VercelSimpleSpanProcessor {
  override onEnd(span: ReadableSpan): void {
    addEveAttributesToSpan(span);
    super.onEnd(span);
  }
}

/**
 * Extends {@link VercelBatchSpanProcessor} to support Eve AI framework spans.
 *
 * Batches spans before exporting. Processes `ai.eve.turn` root spans and
 * propagates Eve session/context attributes into OpenInference conventions
 * before the standard Vercel/GenAI attribute conversion runs.
 *
 * @example
 * ```typescript
 * import { OpenInferenceBatchSpanProcessor, isOpenInferenceSpan } from "@arizeai/openinference-eve";
 * import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
 *
 * const processor = new OpenInferenceBatchSpanProcessor({
 *   exporter: new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" }),
 *   spanFilter: isOpenInferenceSpan,
 *   config: { maxQueueSize: 2048, scheduledDelayMillis: 5000 },
 * });
 * ```
 */
export class OpenInferenceBatchSpanProcessor extends VercelBatchSpanProcessor {
  override onEnd(span: ReadableSpan): void {
    addEveAttributesToSpan(span);
    super.onEnd(span);
  }
}
