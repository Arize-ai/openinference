import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import { diag, SpanKind, SpanStatusCode, Span } from "@opentelemetry/api";
import { OITracer, TraceConfigOptions } from "@arizeai/openinference-core";
import {
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import { VERSION } from "./version";
import { 
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand 
} from "@aws-sdk/client-bedrock-runtime";
import { extractInvokeModelRequestAttributes } from "./attributes/request-attributes";
import { extractInvokeModelResponseAttributes } from "./attributes/response-attributes";
import {
  isToolUseContent,
} from "./types/bedrock-types";
import { splitStream } from "@smithy/util-stream";

const MODULE_NAME = "@aws-sdk/client-bedrock-runtime";

/**
 * Track if the Bedrock instrumentation is patched
 */
let _isBedrockPatched = false;

/**
 * Check if Bedrock instrumentation is enabled/disabled
 */
export function isPatched() {
  return _isBedrockPatched;
}

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

            if (command?.constructor?.name === "InvokeModelWithResponseStreamCommand") {
              return instrumentation._handleInvokeModelWithResponseStreamCommand(
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

      _isBedrockPatched = true;
    }

    return moduleExports;
  }

  private _handleInvokeModelCommand(
    command: InvokeModelCommand,
    original: any,
    client: any,
  ) {
    const requestAttributes = extractInvokeModelRequestAttributes(command);

    const span = this.oiTracer.startSpan("bedrock.invoke_model", {
      kind: SpanKind.CLIENT,
      attributes: requestAttributes,
    });

    try {
      const result = original.apply(client, [command]);

      // AWS SDK v3 send() method always returns a Promise
      return result
        .then((response: any) => {
          extractInvokeModelResponseAttributes(span, response);
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
   * Consumes AWS Bedrock streaming response chunks and extracts attributes for OpenTelemetry span.
   * 
   * This function processes the Bedrock streaming format which consists of JSON lines
   * containing different event types (message_start, content_block_delta, etc.).
   * It accumulates the response content and sets appropriate semantic convention attributes
   * on the provided span. This function is designed to run in the background without
   * blocking the user's stream consumption.
   * 
   * @param stream - The Bedrock response stream (AsyncIterable), typically from splitStream()
   * @param span - The OpenTelemetry span to set attributes on
   * @throws {Error} If critical stream processing errors occur
   * 
   * @example
   * ```typescript
   * // Background processing after stream splitting
   * const [instrumentationStream, userStream] = await splitStream(response.body);
   * this._consumeBedrockStreamChunks(instrumentationStream, span)
   *   .then(() => span.end())
   *   .catch(error => { span.recordException(error); span.end(); });
   * ```
   */
  private async _consumeBedrockStreamChunks(stream: any, span: Span): Promise<void> {
    let outputText = "";
    const contentBlocks: any[] = [];
    let usage: any = {};

    for await (const chunk of stream) {
      if (chunk.chunk?.bytes) {
        const text = new TextDecoder().decode(chunk.chunk.bytes);
        const lines = text.split('\n').filter(line => line.trim());
        
        for (const line of lines) {
          if (line.trim()) {
            const data = JSON.parse(line);
            
            // Handle different event types
            if (data.type === 'message_start' && data.message) {
              usage = data.message.usage || {};
            }
            
            if (data.type === 'content_block_start' && data.content_block) {
              contentBlocks.push(data.content_block);
            }
            
            if (data.type === 'content_block_delta' && data.delta?.text) {
              // Accumulate text content
              outputText += data.delta.text;
              
              // Also update the content block for tool processing
              const lastTextBlock = contentBlocks.find(block => block.type === 'text');
              if (lastTextBlock) {
                lastTextBlock.text = (lastTextBlock.text || '') + data.delta.text;
              } else {
                contentBlocks.push({
                  type: 'text',
                  text: data.delta.text
                });
              }
            }
            
            if (data.type === 'message_delta' && data.usage) {
              usage = { ...usage, ...data.usage };
            }
          }
        }
      }
    }

    // Set output value and MIME type attributes directly
    const mimeType = typeof outputText === "string" && outputText.trim() 
      ? MimeType.TEXT 
      : MimeType.JSON;

    span.setAttributes({
      [SemanticConventions.OUTPUT_VALUE]: outputText,
      [SemanticConventions.OUTPUT_MIME_TYPE]: mimeType,
    });

    // Set structured output message attributes for text content
    if (outputText) {
      span.setAttributes({
        [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]: "assistant",
        [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]: outputText,
      });
    }

    // Extract tool call attributes from content blocks
    const toolUseBlocks = contentBlocks.filter(isToolUseContent);
    toolUseBlocks.forEach((content, toolCallIndex) => {
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

      if (Object.keys(toolCallAttributes).length > 0) {
        span.setAttributes(toolCallAttributes);
      }
    });

    // Set usage attributes directly
    if (usage) {
      const tokenAttributes: Record<string, number> = {};

      if (usage.input_tokens) {
        tokenAttributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT] = usage.input_tokens;
      }
      if (usage.output_tokens) {
        tokenAttributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION] = usage.output_tokens;
      }
      if (usage.input_tokens && usage.output_tokens) {
        tokenAttributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL] = 
          usage.input_tokens + usage.output_tokens;
      }

      if (Object.keys(tokenAttributes).length > 0) {
        span.setAttributes(tokenAttributes);
      }
    }

    span.setStatus({ code: SpanStatusCode.OK });
  }

  /**
   * Handles streaming InvokeModel commands with stream preservation.
   * 
   * Uses @smithy/util-stream splitStream to create two identical streams:
   * one for instrumentation processing and one for user consumption.
   * The instrumentation processing happens in the background while the
   * user receives their stream immediately for optimal performance.
   * 
   * @param command - The InvokeModelWithResponseStreamCommand
   * @param original - The original AWS SDK send method
   * @param client - The Bedrock client instance
   * @returns Promise resolving to the response with preserved user stream
   */
  private _handleInvokeModelWithResponseStreamCommand(
    command: InvokeModelWithResponseStreamCommand,
    original: any,
    client: any,
  ) {
    const requestAttributes = extractInvokeModelRequestAttributes(command as any);

    const span = this.oiTracer.startSpan("bedrock.invoke_model", {
      kind: SpanKind.CLIENT,
      attributes: requestAttributes,
    });

    try {
      const result = original.apply(client, [command]);

      // AWS SDK v3 send() method always returns a Promise
      return result
        .then(async (response: any) => {
          try {
            // Split the stream for instrumentation and user consumption
            const [instrumentationStream, userStream] = await splitStream(response.body);
            
            // Process instrumentation stream in background (non-blocking)
            this._consumeBedrockStreamChunks(instrumentationStream, span)
              .then(() => {
                span.end();
              })
              .catch((streamError: any) => {
                span.recordException(streamError);
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: streamError.message,
                });
                span.end();
              });
            
            // Return response with user stream immediately
            return { ...response, body: userStream };
          } catch (splitError: any) {
            // If stream splitting fails, fall back to original behavior
            diag.warn("Stream splitting failed, falling back to direct consumption:", splitError);
            try {
              await this._consumeBedrockStreamChunks(response.body, span);
            } catch (streamError: any) {
              span.recordException(streamError);
              span.setStatus({
                code: SpanStatusCode.ERROR,
                message: streamError.message,
              });
            }
            span.end();
            return response;
          }
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

  private unpatch(moduleExports: any, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);

    if (moduleExports?.BedrockRuntimeClient) {
      this._unwrap(moduleExports.BedrockRuntimeClient.prototype, "send");
      _isBedrockPatched = false;
    }
  }
}
