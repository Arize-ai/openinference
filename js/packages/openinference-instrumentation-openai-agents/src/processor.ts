import type {
  AgentSpanData,
  CustomSpanData,
  FunctionSpanData,
  GenerationSpanData,
  GuardrailSpanData,
  MCPListToolsSpanData,
  ResponseSpanData,
  Span as SDKSpan,
  SpanData,
  Trace as SDKTrace,
  TracingProcessor,
} from "@openai/agents";
import { context, SpanStatusCode, trace } from "@opentelemetry/api";
import type { Attributes, Context, Span as OTelSpan, Tracer } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";

import {
  getInputAttributes,
  getOutputAttributes,
  getToolAttributes,
  isObjectWithStringKeys,
  OITracer,
  safelyJSONParse,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import type { TraceConfigOptions } from "@arizeai/openinference-core";
import {
  LLMProvider,
  LLMSystem,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

// `TracingProcessor.onSpan{Start,End}` uses `Span<any>` in the SDK, so we
// match that signature and narrow `span.spanData` to `SpanData` below.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnySDKSpan = SDKSpan<any>;

export interface OpenInferenceTracingProcessorConfig {
  /**
   * The tracer to use for creating spans. Accepts either a raw OpenTelemetry
   * {@link Tracer} or a pre-built {@link OITracer}. A raw `Tracer` is wrapped
   * internally; pass `traceConfig` alongside to configure masking.
   */
  tracer?: Tracer | OITracer;
  /**
   * Optional trace configuration for masking sensitive data. Ignored if the
   * provided `tracer` is already an {@link OITracer} (in that case, configure
   * the trace config on the OITracer directly).
   */
  traceConfig?: TraceConfigOptions;
}

const MAX_HANDOFFS_IN_FLIGHT = 1000;

/**
 * Request-side parameters from the Responses API that map to
 * {@link SemanticConventions.LLM_INVOCATION_PARAMETERS}. Everything else on the
 * response object is response-side metadata and is intentionally excluded.
 */
const RESPONSE_INVOCATION_PARAM_KEYS = [
  "temperature",
  "top_p",
  "top_logprobs",
  "max_output_tokens",
  "max_tool_calls",
  "tool_choice",
  "parallel_tool_calls",
  "truncation",
  "reasoning",
  "instructions",
  "text",
  "previous_response_id",
  "prompt",
  "user",
] as const;

/**
 * A TracingProcessor implementation that converts OpenAI Agents SDK spans
 * to OpenTelemetry spans with OpenInference semantic conventions.
 *
 * This processor implements the SDK's TracingProcessor interface and can be
 * registered with the global trace provider.
 */
export class OpenInferenceTracingProcessor implements TracingProcessor {
  private oiTracer: OITracer | null;
  private traceConfig?: TraceConfigOptions;

  // Maps SDK span/trace IDs to OTel spans
  private rootSpans: Map<string, OTelSpan> = new Map();
  private otelSpans: Map<string, OTelSpan> = new Map();

  // Track handoffs for graph visualization
  // Key: "{to_agent}:{trace_id}" -> Value: from_agent
  private reverseHandoffsDict: Map<string, string> = new Map();

  private _shutdown = false;

  constructor(config: OpenInferenceTracingProcessorConfig = {}) {
    this.traceConfig = config.traceConfig;
    this.oiTracer = config.tracer ? this.toOITracer(config.tracer) : null;
  }

  private toOITracer(tracer: Tracer | OITracer): OITracer {
    return tracer instanceof OITracer
      ? tracer
      : new OITracer({ tracer, traceConfig: this.traceConfig });
  }

  /**
   * Set the tracer to use for creating spans
   */
  setTracer(tracer: Tracer | OITracer): void {
    this.oiTracer = this.toOITracer(tracer);
  }

  /**
   * Called when a trace is started
   */
  async onTraceStart(sdkTrace: SDKTrace): Promise<void> {
    if (this._shutdown || !this.oiTracer) {
      return;
    }
    if (isTracingSuppressed(context.active())) {
      return;
    }

    const otelSpan = this.oiTracer.startSpan(sdkTrace.name, {
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
      },
    });

    this.rootSpans.set(sdkTrace.traceId, otelSpan);
  }

  /**
   * Called when a trace is ended
   */
  async onTraceEnd(sdkTrace: SDKTrace): Promise<void> {
    if (this._shutdown) return;

    const rootSpan = this.rootSpans.get(sdkTrace.traceId);
    if (rootSpan) {
      rootSpan.setStatus({ code: SpanStatusCode.OK });
      rootSpan.end();
      this.rootSpans.delete(sdkTrace.traceId);
    }
  }

  /**
   * Called when a span is started
   */
  async onSpanStart(span: AnySDKSpan): Promise<void> {
    if (this._shutdown || !this.oiTracer || !span.startedAt) return;
    if (isTracingSuppressed(context.active())) return;

    const startTime = new Date(span.startedAt);
    const parentSpan = span.parentId
      ? this.otelSpans.get(span.parentId)
      : this.rootSpans.get(span.traceId);

    const spanName = this.getSpanName(span);
    const spanKind = this.getSpanKind(span.spanData as SpanData);

    // Create span with parent context if available
    // Use trace.setSpan to properly establish the parent-child relationship
    let parentContext: Context;
    if (parentSpan) {
      parentContext = trace.setSpan(context.active(), parentSpan);
    } else {
      parentContext = context.active();
    }

    const otelSpan = this.oiTracer.startSpan(
      spanName,
      {
        startTime,
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind,
          [SemanticConventions.LLM_SYSTEM]: LLMSystem.OPENAI,
        },
      },
      parentContext,
    );

    this.otelSpans.set(span.spanId, otelSpan);
  }

  /**
   * Called when a span is ended
   */
  async onSpanEnd(span: AnySDKSpan): Promise<void> {
    if (this._shutdown) return;

    const otelSpan = this.otelSpans.get(span.spanId);
    if (!otelSpan) return;

    // Update span name in case it changed
    otelSpan.updateName(this.getSpanName(span));

    // Extract attributes based on span data type
    const attributes = this.extractAttributes(span);
    otelSpan.setAttributes(attributes);

    // Handle handoff tracking for graph visualization
    this.handleHandoffTracking(span, otelSpan);

    // Set error status if present
    if (span.error) {
      otelSpan.setStatus({
        code: SpanStatusCode.ERROR,
        message: `${span.error.message}: ${safelyJSONStringify(span.error.data) || ""}`,
      });
    } else {
      otelSpan.setStatus({ code: SpanStatusCode.OK });
    }

    // End span with correct timestamp
    const endTime = span.endedAt ? new Date(span.endedAt) : undefined;
    otelSpan.end(endTime);

    this.otelSpans.delete(span.spanId);
  }

  /**
   * Shutdown the processor
   *
   * Ends any in-flight OTel spans before clearing. Without this, OTel never
   * sees `.end()` for those spans and silently drops them, and the
   * `TracerProvider` may hang during its own shutdown.
   */
  async shutdown(_timeout?: number): Promise<void> {
    this._shutdown = true;
    for (const span of this.otelSpans.values()) {
      span.end();
    }
    for (const span of this.rootSpans.values()) {
      span.end();
    }
    this.rootSpans.clear();
    this.otelSpans.clear();
    this.reverseHandoffsDict.clear();
  }

  /**
   * Force flush any pending spans
   */
  async forceFlush(): Promise<void> {
    // No buffering in this implementation
  }

  /**
   * Get the span name from SDK span data
   */
  private getSpanName(span: AnySDKSpan): string {
    const data = span.spanData as SpanData;

    if ("name" in data && data.name) {
      return data.name;
    }

    if (data.type === "handoff" && data.to_agent) {
      return `handoff to ${data.to_agent}`;
    }

    return data.type;
  }

  /**
   * Map SDK span data type to OpenInference span kind
   */
  private getSpanKind(data: SpanData): OpenInferenceSpanKind {
    switch (data.type) {
      case "agent":
        return OpenInferenceSpanKind.AGENT;
      case "function":
      case "handoff":
        return OpenInferenceSpanKind.TOOL;
      case "generation":
      case "response":
        return OpenInferenceSpanKind.LLM;
      case "custom":
      case "guardrail":
      case "mcp_tools":
      default:
        return OpenInferenceSpanKind.CHAIN;
    }
  }

  /**
   * Extract OpenInference attributes from SDK span data
   */
  private extractAttributes(span: AnySDKSpan): Attributes {
    const data = span.spanData as SpanData;
    const attributes: Attributes = {};

    switch (data.type) {
      case "generation":
        this.addGenerationAttributes(data, attributes);
        break;
      case "response":
        this.addResponseAttributes(data, attributes);
        break;
      case "function":
        this.addFunctionAttributes(data, attributes);
        break;
      case "agent":
        this.addAgentAttributes(data, span.traceId, attributes);
        break;
      case "mcp_tools":
        this.addMcpToolsAttributes(data, attributes);
        break;
      case "guardrail":
        this.addGuardrailAttributes(data, attributes);
        break;
      case "custom":
        this.addCustomAttributes(data, attributes);
        break;
    }

    return attributes;
  }

  /**
   * Add attributes for generation (LLM) spans
   */
  private addGenerationAttributes(data: GenerationSpanData, attributes: Attributes): void {
    if (data.model) {
      attributes[SemanticConventions.LLM_MODEL_NAME] = data.model;
    }

    if (data.model_config) {
      const config = Object.fromEntries(
        Object.entries(data.model_config).filter(([, v]) => v != null),
      );
      if (Object.keys(config).length > 0) {
        attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
          safelyJSONStringify(config) || "";

        // Extract provider from base_url if present
        const baseUrl = config.base_url as string | undefined;
        if (baseUrl && baseUrl.includes("api.openai.com")) {
          attributes[SemanticConventions.LLM_PROVIDER] = LLMProvider.OPENAI;
        }
      }
    }

    // Add input messages
    if (data.input && Array.isArray(data.input)) {
      Object.assign(
        attributes,
        getInputAttributes({
          value: safelyJSONStringify(data.input) || "",
          mimeType: MimeType.JSON,
        }),
      );
      this.addChatMessagesAttributes(
        data.input,
        `${SemanticConventions.LLM_INPUT_MESSAGES}`,
        attributes,
      );
    }

    // Add output messages
    if (data.output && Array.isArray(data.output)) {
      Object.assign(
        attributes,
        getOutputAttributes({
          value: safelyJSONStringify(data.output) || "",
          mimeType: MimeType.JSON,
        }),
      );
      this.addChatMessagesAttributes(
        data.output,
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}`,
        attributes,
      );
    }

    // Add usage
    if (data.usage) {
      this.addUsageAttributes(data.usage, attributes);
    }
  }

  /**
   * Add attributes for response spans
   */
  private addResponseAttributes(data: ResponseSpanData, attributes: Attributes): void {
    // Handle response data
    if (data._response) {
      const response = data._response;
      Object.assign(
        attributes,
        getOutputAttributes({
          value: safelyJSONStringify(response) || "",
          mimeType: MimeType.JSON,
        }),
      );

      // Extract model name
      if (response.model && typeof response.model === "string") {
        attributes[SemanticConventions.LLM_MODEL_NAME] = response.model;
      }

      // Extract tools
      if (response.tools && Array.isArray(response.tools)) {
        this.addToolsAttributes(response.tools as Record<string, unknown>[], attributes);
      }

      // Extract usage
      if (response.usage) {
        this.addResponseUsageAttributes(response.usage as Record<string, unknown>, attributes);
      }

      // Extract output messages
      if (response.output && Array.isArray(response.output)) {
        this.addResponseOutputAttributes(response.output as Record<string, unknown>[], attributes);
      }

      // Extract invocation parameters. The Responses API span carries the full
      // response object (mostly response-side metadata: id, created_at,
      // output_text, service_tier, …), so use an allow-list of request-side
      // parameters rather than copying everything and deleting a few fields.
      const responseRecord = response as Record<string, unknown>;
      const params: Record<string, unknown> = {};
      for (const key of RESPONSE_INVOCATION_PARAM_KEYS) {
        const value = responseRecord[key];
        if (value != null) {
          params[key] = value;
        }
      }
      if (Object.keys(params).length > 0) {
        attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
          safelyJSONStringify(params) || "";
      }
    }

    // Handle input
    if (data._input) {
      if (typeof data._input === "string") {
        Object.assign(attributes, getInputAttributes(data._input));
        attributes[
          `${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`
        ] = "user";
        attributes[
          `${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`
        ] = data._input;
      } else if (Array.isArray(data._input)) {
        Object.assign(
          attributes,
          getInputAttributes({
            value: safelyJSONStringify(data._input) || "",
            mimeType: MimeType.JSON,
          }),
        );
        this.addResponseInputAttributes(data._input, attributes);
      }
    }
  }

  /**
   * Add attributes for function (tool) spans
   */
  private addFunctionAttributes(data: FunctionSpanData, attributes: Attributes): void {
    if (data.name) {
      const parsedInput = typeof data.input === "string" ? safelyJSONParse(data.input) : data.input;
      Object.assign(
        attributes,
        getToolAttributes({
          name: data.name,
          parameters: isObjectWithStringKeys(parsedInput) ? parsedInput : {},
        }),
      );
    }

    if (data.input) {
      Object.assign(attributes, getInputAttributes({ value: data.input, mimeType: MimeType.JSON }));
    }

    if (data.output) {
      // Tool outputs are typically JSON-encoded strings; fall back to TEXT for
      // bare strings that don't look like JSON. The Agents SDK serializes return
      // values with `JSON.stringify`, so both objects (`{…}`) and arrays (`[…]`)
      // are possible.
      const looksLikeJson =
        data.output.length > 1 &&
        ((data.output.startsWith("{") && data.output.endsWith("}")) ||
          (data.output.startsWith("[") && data.output.endsWith("]")));
      Object.assign(
        attributes,
        getOutputAttributes(
          looksLikeJson ? { value: data.output, mimeType: MimeType.JSON } : data.output,
        ),
      );
    }
  }

  /**
   * Add attributes for agent spans
   */
  private addAgentAttributes(data: AgentSpanData, traceId: string, attributes: Attributes): void {
    if (data.name) {
      attributes[SemanticConventions.GRAPH_NODE_ID] = data.name;
    }

    // Look up parent node from handoff tracking
    if (data.name) {
      const key = `${data.name}:${traceId}`;
      const parentNode = this.reverseHandoffsDict.get(key);
      if (parentNode) {
        attributes[SemanticConventions.GRAPH_NODE_PARENT_ID] = parentNode;
        this.reverseHandoffsDict.delete(key);
      }
    }
  }

  /**
   * Handle handoff tracking for graph visualization
   */
  private handleHandoffTracking(span: AnySDKSpan, _otelSpan: OTelSpan): void {
    const data = span.spanData as SpanData;

    if (data.type === "handoff" && data.to_agent && data.from_agent) {
      const key = `${data.to_agent}:${span.traceId}`;
      this.reverseHandoffsDict.set(key, data.from_agent);

      // Cap the size to prevent unbounded growth when handoffs are never
      // consumed by a following agent span. Evict oldest entries until we are
      // back within the limit.
      while (this.reverseHandoffsDict.size > MAX_HANDOFFS_IN_FLIGHT) {
        const firstKey = this.reverseHandoffsDict.keys().next().value;
        if (firstKey === undefined) {
          break;
        }
        this.reverseHandoffsDict.delete(firstKey);
      }
    }
  }

  /**
   * Add attributes for MCP tools listing spans
   */
  private addMcpToolsAttributes(data: MCPListToolsSpanData, attributes: Attributes): void {
    if (data.result) {
      Object.assign(
        attributes,
        getOutputAttributes({
          value: safelyJSONStringify(data.result) || "",
          mimeType: MimeType.JSON,
        }),
      );
    }
  }

  /**
   * Add attributes for guardrail spans
   */
  private addGuardrailAttributes(data: GuardrailSpanData, attributes: Attributes): void {
    if (data.name) {
      Object.assign(attributes, getToolAttributes({ name: data.name, parameters: {} }));
    }
    attributes["guardrail.triggered"] = data.triggered;
  }

  /**
   * Add attributes for custom spans
   */
  private addCustomAttributes(data: CustomSpanData, attributes: Attributes): void {
    if (data.data) {
      Object.assign(
        attributes,
        getOutputAttributes({
          value: safelyJSONStringify(data.data) || "",
          mimeType: MimeType.JSON,
        }),
      );
    }
  }

  /**
   * Add chat completion message attributes
   */
  private addChatMessagesAttributes(
    messages: Array<Record<string, unknown>>,
    prefix: string,
    attributes: Attributes,
  ): void {
    messages.forEach((msg, msgIdx) => {
      const msgPrefix = `${prefix}.${msgIdx}`;

      // Role
      if (msg.role && typeof msg.role === "string") {
        attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_ROLE}`] = msg.role;
      }

      // Content
      const content = msg.content;
      if (content) {
        if (typeof content === "string") {
          attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_CONTENT}`] = content;
        } else if (Array.isArray(content)) {
          content.forEach((item, contentIdx) => {
            const contentItem = item as Record<string, unknown>;
            const contentPrefix = `${msgPrefix}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIdx}`;

            if (contentItem.type === "text" && contentItem.text) {
              attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
              attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = String(
                contentItem.text,
              );
            }
          });
        }
      }

      // Tool call ID (for tool role messages)
      if (msg.tool_call_id && typeof msg.tool_call_id === "string") {
        attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`] = msg.tool_call_id;
      }

      // Tool calls (for assistant messages). Index is per-message: the
      // semantic-convention key is `…tool_calls.{idx}` scoped under each
      // message, so it must restart at 0 for every message.
      if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
        (msg.tool_calls as Array<Record<string, unknown>>).forEach((tc, toolCallIdx) => {
          const tcPrefix = `${msgPrefix}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIdx}`;

          if (tc.id) {
            attributes[`${tcPrefix}.${SemanticConventions.TOOL_CALL_ID}`] = String(tc.id);
          }

          const func = tc.function as Record<string, unknown> | undefined;
          if (func) {
            if (func.name) {
              attributes[`${tcPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] = String(
                func.name,
              );
            }
            if (func.arguments && func.arguments !== "{}") {
              attributes[`${tcPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`] =
                String(func.arguments);
            }
          }
        });
      }
    });
  }

  /**
   * Add usage attributes from generation spans
   */
  private addUsageAttributes(usage: Record<string, unknown>, attributes: Attributes): void {
    if (usage.input_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = Number(usage.input_tokens);
    }
    if (usage.output_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = Number(usage.output_tokens);
    }
  }

  /**
   * Add usage attributes from response spans
   */
  private addResponseUsageAttributes(usage: Record<string, unknown>, attributes: Attributes): void {
    if (usage.input_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = Number(usage.input_tokens);
    }
    if (usage.output_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = Number(usage.output_tokens);
    }
    if (usage.total_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] = Number(usage.total_tokens);
    }

    // Handle input token details
    const inputDetails = usage.input_tokens_details as Record<string, unknown> | undefined;
    if (inputDetails?.cached_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] = Number(
        inputDetails.cached_tokens,
      );
    }

    // Handle output token details
    const outputDetails = usage.output_tokens_details as Record<string, unknown> | undefined;
    if (outputDetails?.reasoning_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] = Number(
        outputDetails.reasoning_tokens,
      );
    }
  }

  /**
   * Add tools attributes
   */
  private addToolsAttributes(tools: Array<Record<string, unknown>>, attributes: Attributes): void {
    tools.forEach((tool, idx) => {
      if (tool.type === "function") {
        const schema = {
          type: "function",
          function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.parameters,
            strict: tool.strict,
          },
        };
        attributes[
          `${SemanticConventions.LLM_TOOLS}.${idx}.${SemanticConventions.TOOL_JSON_SCHEMA}`
        ] = safelyJSONStringify(schema) || "";
      }
    });
  }

  /**
   * Add output attributes from response spans
   */
  private addResponseOutputAttributes(
    output: Array<Record<string, unknown>>,
    attributes: Attributes,
  ): void {
    const prefix = SemanticConventions.LLM_OUTPUT_MESSAGES;
    // Each top-level item maps to its own message slot. Bundling consecutive
    // `function_call` items into one slot (with a shared `toolCallIdx`) is
    // tempting but breaks indexing once a `message` arrives between them — the
    // reused `msgIdx` collides and `toolCallIdx` keeps growing across slots.
    let msgIdx = 0;

    output.forEach((item) => {
      if (item.type === "message") {
        const msgPrefix = `${prefix}.${msgIdx}`;

        if (item.role) {
          attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_ROLE}`] = String(item.role);
        }

        if (item.content && Array.isArray(item.content)) {
          (item.content as Array<Record<string, unknown>>).forEach((contentItem, contentIdx) => {
            const contentPrefix = `${msgPrefix}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIdx}`;

            if (contentItem.type === "output_text" && contentItem.text) {
              attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
              attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = String(
                contentItem.text,
              );
            } else if (contentItem.type === "refusal" && contentItem.refusal) {
              attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
              attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = String(
                contentItem.refusal,
              );
            }
          });
        }

        msgIdx++;
      } else if (item.type === "function_call") {
        const msgPrefix = `${prefix}.${msgIdx}`;
        attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_ROLE}`] = "assistant";

        const tcPrefix = `${msgPrefix}.${SemanticConventions.MESSAGE_TOOL_CALLS}.0`;

        if (item.call_id) {
          attributes[`${tcPrefix}.${SemanticConventions.TOOL_CALL_ID}`] = String(item.call_id);
        }
        if (item.name) {
          attributes[`${tcPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`] = String(
            item.name,
          );
        }
        if (item.arguments && item.arguments !== "{}") {
          attributes[`${tcPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`] =
            String(item.arguments);
        }

        msgIdx++;
      }
    });
  }

  /**
   * Add input message attributes for response spans.
   *
   * The Responses API input is an array of items: bare messages
   * (`{role, content}` where content is a string or content-part array),
   * or structured items like `function_call_output`.
   */
  private addResponseInputAttributes(
    input: Array<Record<string, unknown>>,
    attributes: Attributes,
  ): void {
    const prefix = SemanticConventions.LLM_INPUT_MESSAGES;
    let msgIdx = 0;

    input.forEach((item) => {
      const itemType = item.type;

      if (itemType === "function_call_output") {
        const msgPrefix = `${prefix}.${msgIdx}`;
        attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_ROLE}`] = "tool";
        if (item.call_id) {
          attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`] = String(
            item.call_id,
          );
        }
        if (item.output !== undefined && item.output !== null) {
          attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_CONTENT}`] =
            typeof item.output === "string" ? item.output : safelyJSONStringify(item.output) || "";
        }
        msgIdx++;
        return;
      }

      // Treat bare messages and explicit `type === "message"` items the same.
      if (itemType !== undefined && itemType !== "message" && item.role === undefined) {
        return;
      }

      const msgPrefix = `${prefix}.${msgIdx}`;
      if (item.role) {
        attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_ROLE}`] = String(item.role);
      }

      const content = item.content;
      if (typeof content === "string") {
        attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_CONTENT}`] = content;
      } else if (Array.isArray(content)) {
        (content as Array<Record<string, unknown>>).forEach((contentItem, contentIdx) => {
          const contentPrefix = `${msgPrefix}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIdx}`;
          const cType = contentItem.type;
          if ((cType === "input_text" || cType === "text") && contentItem.text) {
            attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "text";
            attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`] = String(
              contentItem.text,
            );
          } else if (cType === "input_image" && contentItem.image_url) {
            attributes[`${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`] = "image";
            attributes[
              `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`
            ] = String(contentItem.image_url);
          }
        });
      }

      msgIdx++;
    });
  }
}
