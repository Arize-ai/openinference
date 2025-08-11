import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

/**
 * Container for span attributes and metadata.
 * Stores various attributes related to a span, including name, type, request/output attributes, metadata, and span kind.
 */
export class Attributes {
  /** Optional name for the span */
  public name?: string;

  /** Type of the span (e.g., LLM, CHAIN, AGENT) */
  public spanType: string = "";

  /** Request attributes for the span */
  public requestAttributes: Record<string, unknown> = {};

  /** Output attributes for the span */
  public outputAttributes: Record<string, unknown> = {};

  /** Metadata for the span */
  public metadata: Record<string, unknown> = {};

  /** Kind of span (LLM, CHAIN, AGENT, etc.) */
  public spanKind: OpenInferenceSpanKind;

  /** Whether an agent span is required */
  public requiresAgentSpan: boolean = false;

  /**
   * Initialize an Attributes instance.
   * @param params Optional parameters for initialization.
   */
  constructor(params?: { name?: string; spanKind?: OpenInferenceSpanKind }) {
    this.name = params?.name;
    this.spanKind = params?.spanKind ?? OpenInferenceSpanKind.LLM;
  }
}
