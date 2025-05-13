import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import type { ExportResult } from "@opentelemetry/core";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { addOpenInferenceAttributesToSpan } from "@arizeai/openinference-vercel/utils";

type ConstructorArgs = {
  /**
   * The API key to use for the OpenInference Trace Exporter.
   * If provided, the `Authorization` header will be added to the request with the value `Bearer ${apiKey}`.
   */
  apiKey?: string;
  /**
   * The endpoint to send the traces to.
   */
  collectorEndpoint: string;
  /**
   * A function that filters the spans to be exported.
   * If provided, the span will be exported if the function returns `true`.
   *
   * @example
   * ```ts
   * import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
   * import { isOpenInferenceSpan, OpenInferenceOTLPTraceExporter } from "@arizeai/openinference-vercel";
   * const spanFilter = (span: ReadableSpan) => {
   *   // add more span filtering logic here if desired
   *   // or just use the default isOpenInferenceSpan filter directly
   *   return isOpenInferenceSpan(span);
   * };
   * const exporter = new OpenInferenceOTLPTraceExporter({
   *   apiKey: "...",
   *   collectorEndpoint: "...",
   *   spanFilter,
   * });
   * ```
   */
  spanFilter?: (span: ReadableSpan) => boolean;
} & Omit<
  NonNullable<ConstructorParameters<typeof OTLPTraceExporter>[0]>,
  "url"
>;

export class OpenInferenceOTLPTraceExporter extends OTLPTraceExporter {
  private readonly spanFilter?: (span: ReadableSpan) => boolean;
  constructor({
    apiKey,
    collectorEndpoint,
    headers,
    spanFilter,
    ...rest
  }: ConstructorArgs) {
    super({
      headers: {
        ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
        ...headers,
      },
      url: collectorEndpoint,
      ...rest,
    });
    this.spanFilter = spanFilter;
  }
  export(
    items: ReadableSpan[],
    resultCallback: (result: ExportResult) => void,
  ) {
    let filteredItems = items.map((i) => {
      addOpenInferenceAttributesToSpan(i);
      return i;
    });
    if (this.spanFilter) {
      filteredItems = filteredItems.filter(this.spanFilter);
    }
    super.export(filteredItems, resultCallback);
  }
}
