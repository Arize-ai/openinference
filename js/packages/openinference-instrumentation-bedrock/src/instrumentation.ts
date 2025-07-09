import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import {
  diag,
  context,
  trace,
  SpanKind,
  SpanStatusCode,
  Span,
} from "@opentelemetry/api";
import {
  SemanticConventions,
  OpenInferenceSpanKind,
  MimeType,
  LLMSystem,
  LLMProvider,
} from "@arizeai/openinference-semantic-conventions";
import { OITracer, TraceConfigOptions } from "@arizeai/openinference-core";
import { TextDecoder } from "util";
import { VERSION } from "./version";

const MODULE_NAME = "@aws-sdk/client-bedrock-runtime";

export interface BedrockInstrumentationConfig extends InstrumentationConfig {
  traceConfig?: TraceConfigOptions;
}

export class BedrockInstrumentation extends InstrumentationBase<BedrockInstrumentationConfig> {
  static readonly COMPONENT = "@arizeai/openinference-instrumentation-bedrock";
  static readonly VERSION = VERSION;

  private oiTracer: OITracer;

  constructor(config: BedrockInstrumentationConfig = {}) {
    super(
      BedrockInstrumentation.COMPONENT,
      BedrockInstrumentation.VERSION,
      config,
    );

    this.oiTracer = new OITracer({
      tracer: this.tracer,
      traceConfig: config.traceConfig,
    });
  }

  protected init(): InstrumentationModuleDefinition<unknown>[] {
    const module = new InstrumentationNodeModuleDefinition<unknown>(
      MODULE_NAME,
      ["^3.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return [module];
  }

  private patch(moduleExports: any, moduleVersion?: string) {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);

    if (moduleExports?.BedrockRuntimeClient) {
      // eslint-disable-next-line @typescript-eslint/no-this-alias
      const instrumentation = this;

      // Wrap the client's send method to intercept commands
      this._wrap(
        moduleExports.BedrockRuntimeClient.prototype,
        "send",
        (original: any) => {
          return function patchedSend(this: unknown, command: any) {
            if (command?.constructor?.name === "InvokeModelCommand") {
              return instrumentation._handleInvokeModelCommand(
                command,
                original,
                this,
              );
            }

            // Pass through other commands without instrumentation
            return original.apply(this, [command]);
          };
        },
      );
    }

    return moduleExports;
  }

  private _handleInvokeModelCommand(command: any, original: any, client: any) {
    const requestAttributes =
      this._extractInvokeModelRequestAttributes(command);

    const span = this.oiTracer.startSpan("bedrock.invoke_model", {
      kind: SpanKind.CLIENT,
      attributes: requestAttributes,
    });

    try {
      const result = original.apply(client, [command]);

      // AWS SDK v3 send() method always returns a Promise
      return result
        .then((response: any) => {
          this._extractInvokeModelResponseAttributes(span, response);
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
          return response;
        })
        .catch((error: any) => {
          span.recordException(error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: error.message,
          });
          span.end();
          throw error;
        });
    } catch (error: any) {
      // Handle errors that occur before the Promise is returned (e.g. invalid parameters)
      span.recordException(error);
      span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      span.end();
      throw error;
    }
  }

  /**
   * Extracts base request attributes (model, system, provider, invocation parameters)
   */
  private _extractBaseRequestAttributes(command: any): Record<string, any> {
    const modelId = command.input?.modelId || "unknown";
    const requestBody = command.input?.body
      ? JSON.parse(command.input.body)
      : {};

    const attributes: Record<string, any> = {
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      [SemanticConventions.LLM_SYSTEM]: "bedrock",
      [SemanticConventions.LLM_MODEL_NAME]: modelId,
      [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
      [SemanticConventions.LLM_PROVIDER]: LLMProvider.AWS,
    };

    // Add invocation parameters for model configuration
    const invocationParams: Record<string, any> = {};
    if (requestBody.anthropic_version)
      invocationParams.anthropic_version = requestBody.anthropic_version;
    if (requestBody.max_tokens)
      invocationParams.max_tokens = requestBody.max_tokens;

    if (Object.keys(invocationParams).length > 0) {
      attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS] =
        JSON.stringify(invocationParams);
    }

    return attributes;
  }

  /**
   * Extracts input messages attributes from request body
   */
  private _extractInputMessagesAttributes(
    requestBody: any,
  ): Record<string, any> {
    const attributes: Record<string, any> = {};

    // Extract user's message text as primary input value
    let inputValue = "";
    if (
      requestBody.messages &&
      Array.isArray(requestBody.messages) &&
      requestBody.messages.length > 0
    ) {
      const userMessage = requestBody.messages.find(
        (msg: any) => msg.role === "user",
      );
      if (userMessage && userMessage.content) {
        inputValue = userMessage.content;
      }
    }

    attributes[SemanticConventions.INPUT_VALUE] = inputValue;

    // Add structured input message attributes
    if (requestBody.messages && Array.isArray(requestBody.messages)) {
      requestBody.messages.forEach((message: any, index: number) => {
        if (message.role && message.content) {
          attributes[
            `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_ROLE}`
          ] = message.role;
          attributes[
            `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_CONTENT}`
          ] = message.content;
        }
      });
    }

    return attributes;
  }

  /**
   * Extracts input tool attributes from request body
   */
  private _extractInputToolAttributes(requestBody: any): Record<string, any> {
    const attributes: Record<string, any> = {};

    // Add input tools from request if present
    if (requestBody.tools && Array.isArray(requestBody.tools)) {
      requestBody.tools.forEach((tool: any, index: number) => {
        // Convert Bedrock tool format to OpenAI format for consistent schema
        const openAIToolFormat = {
          type: "function",
          function: {
            name: tool.name,
            description: tool.description,
            parameters: tool.input_schema,
          },
        };
        attributes[
          `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`
        ] = JSON.stringify(openAIToolFormat);
      });
    }

    return attributes;
  }

  /**
   * Extracts semantic convention attributes from InvokeModel request command
   */
  private _extractInvokeModelRequestAttributes(
    command: any,
  ): Record<string, any> {
    const requestBody = command.input?.body
      ? JSON.parse(command.input.body)
      : {};

    // Start with base attributes
    const attributes = this._extractBaseRequestAttributes(command);

    // Add input messages attributes
    Object.assign(
      attributes,
      this._extractInputMessagesAttributes(requestBody),
    );

    // Add input tool attributes
    Object.assign(attributes, this._extractInputToolAttributes(requestBody));

    return attributes;
  }

  /**
   * Extracts output messages attributes from response body
   */
  private _extractOutputMessagesAttributes(responseBody: any, span: Span) {
    // Extract assistant's message text as primary output value
    let outputValue = "";
    if (responseBody.content && Array.isArray(responseBody.content)) {
      const textContent = responseBody.content.find(
        (content: any) => content.type === "text",
      );
      if (textContent && textContent.text) {
        outputValue = textContent.text;
      }
    }

    span.setAttributes({
      [SemanticConventions.OUTPUT_VALUE]: outputValue,
      [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
    });

    // Add structured output message attributes for text content
    if (responseBody.content && Array.isArray(responseBody.content)) {
      responseBody.content.forEach((content: any, index: number) => {
        if (content.type === "text") {
          span.setAttributes({
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
              "assistant",
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
              content.text,
          });
        }
      });
    }
  }

  /**
   * Extracts tool call attributes from response body
   */
  private _extractToolCallAttributes(responseBody: any, span: Span) {
    if (responseBody.content && Array.isArray(responseBody.content)) {
      let toolCallIndex = 0;
      responseBody.content.forEach((content: any, index: number) => {
        if (content.type === "tool_use") {
          // Extract tool call attributes following OpenAI pattern
          const toolCallAttributes: Record<string, string> = {};

          if (content.name) {
            toolCallAttributes[
              `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`
            ] = content.name;
          }
          if (content.input) {
            toolCallAttributes[
              `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`
            ] = JSON.stringify(content.input);
          }
          if (content.id) {
            toolCallAttributes[
              `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolCallIndex}.${SemanticConventions.TOOL_CALL_ID}`
            ] = content.id;
          }

          span.setAttributes(toolCallAttributes);
          toolCallIndex++;
        }
      });
    }
  }

  /**
   * Extracts usage attributes from response body
   */
  private _extractUsageAttributes(responseBody: any, span: Span) {
    // Add token usage metrics
    if (responseBody.usage) {
      const tokenAttributes: Record<string, number> = {};

      if (responseBody.usage.input_tokens) {
        tokenAttributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] =
          responseBody.usage.input_tokens;
      }
      if (responseBody.usage.output_tokens) {
        tokenAttributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] =
          responseBody.usage.output_tokens;
      }
      if (responseBody.usage.input_tokens && responseBody.usage.output_tokens) {
        tokenAttributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] =
          responseBody.usage.input_tokens + responseBody.usage.output_tokens;
      }

      span.setAttributes(tokenAttributes);
    }
  }

  /**
   * Extracts semantic convention attributes from InvokeModel response and adds them to the span
   */
  private _extractInvokeModelResponseAttributes(span: Span, response: any) {
    try {
      if (!response.body) return;

      const responseText = new TextDecoder().decode(response.body);
      const responseBody = JSON.parse(responseText);

      // Extract output messages attributes
      this._extractOutputMessagesAttributes(responseBody, span);

      // Extract tool call attributes
      this._extractToolCallAttributes(responseBody, span);

      // Extract usage attributes
      this._extractUsageAttributes(responseBody, span);
    } catch (error) {
      diag.warn("Failed to extract InvokeModel response attributes:", error);
    }
  }

  private unpatch(moduleExports: any, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);

    if (moduleExports?.BedrockRuntimeClient) {
      this._unwrap(moduleExports.BedrockRuntimeClient.prototype, "send");
    }
  }
}
