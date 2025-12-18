import {
  safelyJSONStringify,
  TraceConfigOptions,
} from "@arizeai/openinference-core";
import {
  LLMProvider,
  LLMSystem,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  Attributes,
  context,
  Context,
  Span as OTelSpan,
  SpanStatusCode,
  trace,
  Tracer,
} from "@opentelemetry/api";

// Types from @openai/agents - we define minimal interfaces to avoid hard dependency
// These will be duck-typed at runtime

interface SDKTrace {
  type: "trace";
  traceId: string;
  name: string;
  groupId?: string | null;
}

interface SDKSpanData {
  type: string;
  name?: string;
  // Generation span data
  input?: Array<Record<string, unknown>>;
  output?: Array<Record<string, unknown>>;
  model?: string;
  model_config?: Record<string, unknown>;
  usage?: Record<string, unknown>;
  // Response span data
  response_id?: string;
  _input?: string | Record<string, unknown>[];
  _response?: Record<string, unknown>;
  // Function span data
  mcp_data?: string;
  // Handoff span data
  from_agent?: string;
  to_agent?: string;
  // Custom/Guardrail span data
  data?: Record<string, unknown>;
  triggered?: boolean;
  // Agent span data
  handoffs?: string[];
  tools?: string[];
  output_type?: string;
  // MCP span data
  server?: string;
  result?: string[];
}

interface SDKSpanError {
  message: string;
  data?: Record<string, unknown>;
}

interface SDKSpan {
  type: "trace.span";
  traceId: string;
  spanId: string;
  parentId: string | null;
  spanData: SDKSpanData;
  startedAt: string | null;
  endedAt: string | null;
  error: SDKSpanError | null;
}

export interface OpenInferenceTracingProcessorConfig {
  /**
   * The OpenTelemetry tracer to use for creating spans
   */
  tracer?: Tracer;
  /**
   * Optional trace configuration for masking sensitive data
   */
  traceConfig?: TraceConfigOptions;
}

const MAX_HANDOFFS_IN_FLIGHT = 1000;

/**
 * A TracingProcessor implementation that converts OpenAI Agents SDK spans
 * to OpenTelemetry spans with OpenInference semantic conventions.
 *
 * This processor implements the SDK's TracingProcessor interface and can be
 * registered with the global trace provider.
 */
export class OpenInferenceTracingProcessor {
  private tracer: Tracer | null;
  private traceConfig?: TraceConfigOptions;

  // Maps SDK span/trace IDs to OTel spans
  private rootSpans: Map<string, OTelSpan> = new Map();
  private otelSpans: Map<string, OTelSpan> = new Map();

  // Track handoffs for graph visualization
  // Key: "{to_agent}:{trace_id}" -> Value: from_agent
  private reverseHandoffsDict: Map<string, string> = new Map();

  private _shutdown = false;

  constructor(config: OpenInferenceTracingProcessorConfig = {}) {
    this.tracer = config.tracer || null;
    this.traceConfig = config.traceConfig;
  }

  /**
   * Set the tracer to use for creating spans
   */
  setTracer(tracer: Tracer): void {
    this.tracer = tracer;
  }

  /**
   * Called when a trace is started
   */
  async onTraceStart(sdkTrace: SDKTrace): Promise<void> {
    if (this._shutdown || !this.tracer) {
      return;
    }

    const otelSpan = this.tracer.startSpan(sdkTrace.name, {
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.AGENT,
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
  async onSpanStart(span: SDKSpan): Promise<void> {
    if (this._shutdown || !this.tracer || !span.startedAt) return;

    const startTime = new Date(span.startedAt);
    const parentSpan = span.parentId
      ? this.otelSpans.get(span.parentId)
      : this.rootSpans.get(span.traceId);

    const spanName = this.getSpanName(span);
    const spanKind = this.getSpanKind(span.spanData);

    // Create span with parent context if available
    // Use trace.setSpan to properly establish the parent-child relationship
    let parentContext: Context;
    if (parentSpan) {
      parentContext = trace.setSpan(context.active(), parentSpan);
    } else {
      parentContext = context.active();
    }

    const otelSpan = this.tracer.startSpan(
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
  async onSpanEnd(span: SDKSpan): Promise<void> {
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
   */
  async shutdown(_timeout?: number): Promise<void> {
    this._shutdown = true;
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
  private getSpanName(span: SDKSpan): string {
    const data = span.spanData;

    if (data.name && typeof data.name === "string") {
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
  private getSpanKind(data: SDKSpanData): OpenInferenceSpanKind {
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
  private extractAttributes(span: SDKSpan): Attributes {
    const data = span.spanData;
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
  private addGenerationAttributes(
    data: SDKSpanData,
    attributes: Attributes,
  ): void {
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
      attributes[SemanticConventions.INPUT_VALUE] =
        safelyJSONStringify(data.input) || "";
      attributes[SemanticConventions.INPUT_MIME_TYPE] = MimeType.JSON;
      this.addChatMessagesAttributes(
        data.input,
        `${SemanticConventions.LLM_INPUT_MESSAGES}`,
        attributes,
      );
    }

    // Add output messages
    if (data.output && Array.isArray(data.output)) {
      attributes[SemanticConventions.OUTPUT_VALUE] =
        safelyJSONStringify(data.output) || "";
      attributes[SemanticConventions.OUTPUT_MIME_TYPE] = MimeType.JSON;
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
  private addResponseAttributes(
    data: SDKSpanData,
    attributes: Attributes,
  ): void {
    // Handle response data
    if (data._response) {
      const response = data._response;
      attributes[SemanticConventions.OUTPUT_VALUE] =
        safelyJSONStringify(response) || "";
      attributes[SemanticConventions.OUTPUT_MIME_TYPE] = MimeType.JSON;

      // Extract model name
      if (response.model && typeof response.model === "string") {
        attributes[SemanticConventions.LLM_MODEL_NAME] = response.model;
      }

      // Extract tools
      if (response.tools && Array.isArray(response.tools)) {
        this.addToolsAttributes(
          response.tools as Record<string, unknown>[],
          attributes,
        );
      }

      // Extract usage
      if (response.usage) {
        this.addResponseUsageAttributes(
          response.usage as Record<string, unknown>,
          attributes,
        );
      }

      // Extract output messages
      if (response.output && Array.isArray(response.output)) {
        this.addResponseOutputAttributes(
          response.output as Record<string, unknown>[],
          attributes,
        );
      }

      // Extract invocation parameters
      const params = { ...response };
      delete params.object;
      delete params.tools;
      delete params.usage;
      delete params.output;
      delete params.error;
      delete params.status;
      attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
        safelyJSONStringify(params) || "";
    }

    // Handle input
    if (data._input) {
      if (typeof data._input === "string") {
        attributes[SemanticConventions.INPUT_VALUE] = data._input;
      } else if (Array.isArray(data._input)) {
        attributes[SemanticConventions.INPUT_VALUE] =
          safelyJSONStringify(data._input) || "";
        attributes[SemanticConventions.INPUT_MIME_TYPE] = MimeType.JSON;
      }
    }
  }

  /**
   * Add attributes for function (tool) spans
   */
  private addFunctionAttributes(
    data: SDKSpanData,
    attributes: Attributes,
  ): void {
    if (data.name) {
      attributes[SemanticConventions.TOOL_NAME] = data.name;
    }

    // Handle input - for function spans, data.input is the input JSON
    // Use type assertion through unknown to access the input field
    const dataRecord = data as unknown as Record<string, unknown>;
    const inputValue = dataRecord.input;
    if (inputValue) {
      if (typeof inputValue === "string") {
        attributes[SemanticConventions.INPUT_VALUE] = inputValue;
        attributes[SemanticConventions.INPUT_MIME_TYPE] = MimeType.JSON;
      } else {
        attributes[SemanticConventions.INPUT_VALUE] =
          safelyJSONStringify(inputValue) || "";
        attributes[SemanticConventions.INPUT_MIME_TYPE] = MimeType.JSON;
      }
    }

    // Handle output - for function spans, data.output is the output string/object
    const outputValue = dataRecord.output;
    if (outputValue !== undefined && outputValue !== null) {
      const outputStr =
        typeof outputValue === "string"
          ? outputValue
          : safelyJSONStringify(outputValue) || "";
      attributes[SemanticConventions.OUTPUT_VALUE] = outputStr;

      // Set mime type if output looks like JSON
      if (
        typeof outputValue === "string" &&
        outputValue.length > 1 &&
        outputValue.startsWith("{") &&
        outputValue.endsWith("}")
      ) {
        attributes[SemanticConventions.OUTPUT_MIME_TYPE] = MimeType.JSON;
      }
    }
  }

  /**
   * Add attributes for agent spans
   */
  private addAgentAttributes(
    data: SDKSpanData,
    traceId: string,
    attributes: Attributes,
  ): void {
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
  private handleHandoffTracking(span: SDKSpan, _otelSpan: OTelSpan): void {
    const data = span.spanData;

    if (data.type === "handoff" && data.to_agent && data.from_agent) {
      const key = `${data.to_agent}:${span.traceId}`;
      this.reverseHandoffsDict.set(key, data.from_agent);

      // Cap the size to prevent memory leaks
      if (this.reverseHandoffsDict.size > MAX_HANDOFFS_IN_FLIGHT) {
        const firstKey = this.reverseHandoffsDict.keys().next().value;
        if (firstKey) {
          this.reverseHandoffsDict.delete(firstKey);
        }
      }
    }
  }

  /**
   * Add attributes for MCP tools listing spans
   */
  private addMcpToolsAttributes(
    data: SDKSpanData,
    attributes: Attributes,
  ): void {
    if (data.result) {
      attributes[SemanticConventions.OUTPUT_VALUE] =
        safelyJSONStringify(data.result) || "";
      attributes[SemanticConventions.OUTPUT_MIME_TYPE] = MimeType.JSON;
    }
  }

  /**
   * Add attributes for guardrail spans
   */
  private addGuardrailAttributes(
    data: SDKSpanData,
    attributes: Attributes,
  ): void {
    if (data.name) {
      attributes[SemanticConventions.TOOL_NAME] = data.name;
    }
    if (data.triggered !== undefined) {
      attributes["guardrail.triggered"] = data.triggered;
    }
  }

  /**
   * Add attributes for custom spans
   */
  private addCustomAttributes(
    data: SDKSpanData,
    attributes: Attributes,
  ): void {
    if (data.data) {
      attributes[SemanticConventions.OUTPUT_VALUE] =
        safelyJSONStringify(data.data) || "";
      attributes[SemanticConventions.OUTPUT_MIME_TYPE] = MimeType.JSON;
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
    let toolCallIdx = 0;

    messages.forEach((msg, msgIdx) => {
      const msgPrefix = `${prefix}.${msgIdx}`;

      // Role
      if (msg.role && typeof msg.role === "string") {
        attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_ROLE}`] =
          msg.role;
      }

      // Content
      const content = msg.content;
      if (content) {
        if (typeof content === "string") {
          attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_CONTENT}`] =
            content;
        } else if (Array.isArray(content)) {
          content.forEach((item, contentIdx) => {
            const contentItem = item as Record<string, unknown>;
            const contentPrefix = `${msgPrefix}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIdx}`;

            if (contentItem.type === "text" && contentItem.text) {
              attributes[
                `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`
              ] = "text";
              attributes[
                `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
              ] = String(contentItem.text);
            }
          });
        }
      }

      // Tool call ID (for tool role messages)
      if (msg.tool_call_id && typeof msg.tool_call_id === "string") {
        attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`] =
          msg.tool_call_id;
      }

      // Tool calls (for assistant messages)
      if (msg.tool_calls && Array.isArray(msg.tool_calls)) {
        (msg.tool_calls as Array<Record<string, unknown>>).forEach((tc) => {
          const tcPrefix = `${msgPrefix}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIdx}`;

          if (tc.id) {
            attributes[`${tcPrefix}.${SemanticConventions.TOOL_CALL_ID}`] =
              String(tc.id);
          }

          const func = tc.function as Record<string, unknown> | undefined;
          if (func) {
            if (func.name) {
              attributes[
                `${tcPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
              ] = String(func.name);
            }
            if (func.arguments && func.arguments !== "{}") {
              attributes[
                `${tcPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
              ] = String(func.arguments);
            }
          }

          toolCallIdx++;
        });
      }
    });
  }

  /**
   * Add usage attributes from generation spans
   */
  private addUsageAttributes(
    usage: Record<string, unknown>,
    attributes: Attributes,
  ): void {
    if (usage.input_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = Number(
        usage.input_tokens,
      );
    }
    if (usage.output_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = Number(
        usage.output_tokens,
      );
    }
  }

  /**
   * Add usage attributes from response spans
   */
  private addResponseUsageAttributes(
    usage: Record<string, unknown>,
    attributes: Attributes,
  ): void {
    if (usage.input_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = Number(
        usage.input_tokens,
      );
    }
    if (usage.output_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = Number(
        usage.output_tokens,
      );
    }
    if (usage.total_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] = Number(
        usage.total_tokens,
      );
    }

    // Handle input token details
    const inputDetails = usage.input_tokens_details as
      | Record<string, unknown>
      | undefined;
    if (inputDetails?.cached_tokens !== undefined) {
      attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] =
        Number(inputDetails.cached_tokens);
    }

    // Handle output token details
    const outputDetails = usage.output_tokens_details as
      | Record<string, unknown>
      | undefined;
    if (outputDetails?.reasoning_tokens !== undefined) {
      attributes[
        SemanticConventions.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING
      ] = Number(outputDetails.reasoning_tokens);
    }
  }

  /**
   * Add tools attributes
   */
  private addToolsAttributes(
    tools: Array<Record<string, unknown>>,
    attributes: Attributes,
  ): void {
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
    let msgIdx = 0;
    let toolCallIdx = 0;

    output.forEach((item) => {
      const prefix = SemanticConventions.LLM_OUTPUT_MESSAGES;

      if (item.type === "message") {
        const msgPrefix = `${prefix}.${msgIdx}`;

        if (item.role) {
          attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_ROLE}`] =
            String(item.role);
        }

        if (item.content && Array.isArray(item.content)) {
          (item.content as Array<Record<string, unknown>>).forEach(
            (contentItem, contentIdx) => {
              const contentPrefix = `${msgPrefix}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIdx}`;

              if (contentItem.type === "output_text" && contentItem.text) {
                attributes[
                  `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`
                ] = "text";
                attributes[
                  `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
                ] = String(contentItem.text);
              } else if (
                contentItem.type === "refusal" &&
                contentItem.refusal
              ) {
                attributes[
                  `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`
                ] = "text";
                attributes[
                  `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
                ] = String(contentItem.refusal);
              }
            },
          );
        }

        msgIdx++;
      } else if (item.type === "function_call") {
        const msgPrefix = `${prefix}.${msgIdx}`;
        attributes[`${msgPrefix}.${SemanticConventions.MESSAGE_ROLE}`] =
          "assistant";

        const tcPrefix = `${msgPrefix}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIdx}`;

        if (item.call_id) {
          attributes[`${tcPrefix}.${SemanticConventions.TOOL_CALL_ID}`] = String(
            item.call_id,
          );
        }
        if (item.name) {
          attributes[
            `${tcPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
          ] = String(item.name);
        }
        if (item.arguments && item.arguments !== "{}") {
          attributes[
            `${tcPrefix}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
          ] = String(item.arguments);
        }

        toolCallIdx++;
      }
    });
  }
}
