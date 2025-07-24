import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import {
  diag,
  SpanKind,
  SpanStatusCode,
  context,
} from "@opentelemetry/api";
import {
  SemanticConventions,
  OpenInferenceSpanKind,
} from "@arizeai/openinference-semantic-conventions";
import { getAttributesFromContext } from "@arizeai/openinference-core";
import { VERSION } from "./version";
import {
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand,
  ConverseCommand,
  BedrockRuntimeClient,
  InvokeModelResponse,
  ConverseResponse,
} from "@aws-sdk/client-bedrock-runtime";
import { extractInvokeModelRequestAttributes } from "./attributes/invoke-model-request-attributes";
import { extractInvokeModelResponseAttributes } from "./attributes/invoke-model-response-attributes";
import { extractConverseRequestAttributes } from "./attributes/converse-request-attributes";
import { extractConverseResponseAttributes } from "./attributes/converse-response-attributes";
import { consumeBedrockStreamChunks } from "./attributes/invoke-model-streaming-response-attributes";
import { splitStream } from "@smithy/util-stream";

const MODULE_NAME = "@aws-sdk/client-bedrock-runtime";

// AWS SDK module interface for proper typing (following OpenAI pattern)
interface BedrockModuleExports {
  BedrockRuntimeClient: typeof BedrockRuntimeClient;
}

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
  // Remove traceConfig since we'll use direct OpenTelemetry API
}

export class BedrockInstrumentation extends InstrumentationBase<BedrockInstrumentationConfig> {
  static readonly COMPONENT = "@arizeai/openinference-instrumentation-bedrock";
  static readonly VERSION = VERSION;

  constructor(config: BedrockInstrumentationConfig = {}) {
    super(
      BedrockInstrumentation.COMPONENT,
      BedrockInstrumentation.VERSION,
      config,
    );
  }

  protected init(): InstrumentationModuleDefinition<BedrockModuleExports>[] {
    const module = new InstrumentationNodeModuleDefinition<BedrockModuleExports>(
      MODULE_NAME,
      ["^3.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return [module];
  }

  private patch(moduleExports: BedrockModuleExports, moduleVersion?: string) {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);

    if (moduleExports?.BedrockRuntimeClient) {
      // eslint-disable-next-line @typescript-eslint/no-this-alias
      const instrumentation = this;

      // Wrap the client's send method to intercept commands
      this._wrap(
        moduleExports.BedrockRuntimeClient.prototype,
        "send",
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (original: any) => {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          return function patchedSend(this: unknown, command: any) {
            if (command?.constructor?.name === "InvokeModelCommand") {
              return instrumentation.handleInvokeModelCommand(
                command as InvokeModelCommand,
                original,
                this,
              );
            }

            if (
              command?.constructor?.name ===
              "InvokeModelWithResponseStreamCommand"
            ) {
              return instrumentation.handleInvokeModelWithResponseStreamCommand(
                command as InvokeModelWithResponseStreamCommand,
                original,
                this,
              );
            }

            if (command?.constructor?.name === "ConverseCommand") {
              return instrumentation.handleConverseCommand(
                command as ConverseCommand,
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

  private handleInvokeModelCommand(
    command: InvokeModelCommand,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    original: any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    client: any,
  ) {
    const span = this.tracer.startSpan("bedrock.invoke_model", {
      kind: SpanKind.INTERNAL,
    });

    // Add OpenInference span kind attribute
    span.setAttribute(
      SemanticConventions.OPENINFERENCE_SPAN_KIND,
      OpenInferenceSpanKind.LLM,
    );

    // Add OpenInference context attributes
    const contextAttributes = getAttributesFromContext(context.active());
    span.setAttributes(contextAttributes);

    // Extract request attributes directly onto the span
    extractInvokeModelRequestAttributes({ span, command });

    try {
      const result = original.apply(client, [command]) as Promise<InvokeModelResponse>;

      // AWS SDK v3 send() method always returns a Promise
      return result
        .then((response: InvokeModelResponse) => {
          extractInvokeModelResponseAttributes({ span, response });
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
          return response;
        })
        .catch((error: Error) => {
          span.recordException(error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: error.message,
          });
          span.end();
          throw error;
        });
    } catch (error) {
      // Handle errors that occur before the Promise is returned (e.g. invalid parameters)
      if (error instanceof Error) {
        span.recordException(error);
        span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      } else {
        span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
      }
      span.end();
      throw error;
    }
  }

  private unpatch(moduleExports: BedrockModuleExports, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);

    if (moduleExports?.BedrockRuntimeClient) {
      this._unwrap(moduleExports.BedrockRuntimeClient.prototype, "send");
      _isBedrockPatched = false;
    }
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
  private handleInvokeModelWithResponseStreamCommand(
    command: InvokeModelWithResponseStreamCommand,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    original: any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    client: any,
  ) {
    const span = this.tracer.startSpan("bedrock.invoke_model", {
      kind: SpanKind.INTERNAL,
    });

    // Add OpenInference span kind attribute
    span.setAttribute(
      SemanticConventions.OPENINFERENCE_SPAN_KIND,
      OpenInferenceSpanKind.LLM,
    );

    // Add OpenInference context attributes
    const contextAttributes = getAttributesFromContext(context.active());
    span.setAttributes(contextAttributes);

    // Extract request attributes directly onto the span
    // Note: InvokeModelWithResponseStreamCommand has compatible input structure with InvokeModelCommand
    extractInvokeModelRequestAttributes({ 
      span, 
      command: command as unknown as InvokeModelCommand 
    });

    try {
      const result = original.apply(client, [command]) as Promise<{ body: AsyncIterable<unknown> }>;

      // AWS SDK v3 send() method always returns a Promise
      return result
        .then(async (response: { body: AsyncIterable<unknown> }) => {
          try {
            // Check if response.body exists before splitting
            if (!response.body) {
              span.setStatus({ code: SpanStatusCode.ERROR, message: "Response body is undefined" });
              span.end();
              return response;
            }

            // Split the stream for instrumentation and user consumption
            // Note: splitStream expects a Node.js Readable or Web ReadableStream
            const splitResult = await splitStream(
              response.body as Parameters<typeof splitStream>[0],
            ) as [AsyncIterable<unknown>, AsyncIterable<unknown>];
            
            // splitStream always returns [stream1, stream2], never null
            const instrumentationStream = splitResult[0];
            const userStream = splitResult[1];

            // Process instrumentation stream in background (non-blocking)
            if (instrumentationStream) {
              // @ts-expect-error - splitStream from @smithy/util-stream always returns non-null streams
              consumeBedrockStreamChunks({ 
                stream: instrumentationStream, 
                span 
              })
                .then(() => {
                  span.end();
                })
                .catch((streamError: Error) => {
                  span.recordException(streamError);
                  span.setStatus({
                    code: SpanStatusCode.ERROR,
                    message: streamError.message,
                  });
                  span.end();
                });
            } else {
              // Fallback: end span if no stream available
              span.end();
            }

            // Return response with user stream immediately
            return { ...response, body: userStream };
          } catch (splitError) {
            // If stream splitting fails, fall back to original behavior
            const errorMessage = splitError instanceof Error ? splitError.message : String(splitError);
            diag.warn(
              "Stream splitting failed, falling back to direct consumption:",
              errorMessage,
            );
            try {
              if (response.body) {
                await consumeBedrockStreamChunks({ 
                  stream: response.body, 
                  span 
                });
              }
            } catch (streamError) {
              const streamErrorMessage = streamError instanceof Error ? streamError.message : String(streamError);
              if (streamError instanceof Error) {
                span.recordException(streamError);
              }
              span.setStatus({
                code: SpanStatusCode.ERROR,
                message: streamErrorMessage,
              });
            }
            span.end();
            return response;
          }
        })
        .catch((error: Error) => {
          span.recordException(error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: error.message,
          });
          span.end();
          throw error;
        });
    } catch (error) {
      // Handle errors that occur before the Promise is returned (e.g. invalid parameters)
      if (error instanceof Error) {
        span.recordException(error);
        span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      } else {
        span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
      }
      span.end();
      throw error;
    }
  }

  private handleConverseCommand(
    command: ConverseCommand,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    original: any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    client: any,
  ) {
    const span = this.tracer.startSpan("bedrock.converse", {
      kind: SpanKind.INTERNAL,
    });

    // Add OpenInference span kind attribute
    span.setAttribute(
      SemanticConventions.OPENINFERENCE_SPAN_KIND,
      OpenInferenceSpanKind.LLM,
    );

    // Add OpenInference context attributes
    const contextAttributes = getAttributesFromContext(context.active());
    span.setAttributes(contextAttributes);

    // Extract request attributes directly onto the span
    extractConverseRequestAttributes({ span, command });

    try {
      const result = original.apply(client, [command]) as Promise<ConverseResponse>;

      // AWS SDK v3 send() method always returns a Promise
      return result
        .then((response: ConverseResponse) => {
          // Extract response attributes
          extractConverseResponseAttributes({ span, response });
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
          return response;
        })
        .catch((error: Error) => {
          span.recordException(error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: error.message,
          });
          span.end();
          throw error;
        });
    } catch (error) {
      // Handle errors that occur before the Promise is returned (e.g. invalid parameters)
      if (error instanceof Error) {
        span.recordException(error);
        span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      } else {
        span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
      }
      span.end();
      throw error;
    }
  }
}