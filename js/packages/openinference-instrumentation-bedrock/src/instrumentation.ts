import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import { diag, SpanKind, SpanStatusCode, context } from "@opentelemetry/api";
import {
  SemanticConventions,
  OpenInferenceSpanKind,
} from "@arizeai/openinference-semantic-conventions";
import {
  getAttributesFromContext,
  OITracer,
  TraceConfigOptions,
} from "@arizeai/openinference-core";
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
import {
  consumeBedrockStreamChunks,
  safelySplitStream,
} from "./attributes/invoke-model-streaming-response-attributes";

const MODULE_NAME = "@aws-sdk/client-bedrock-runtime";

/**
 * AWS SDK module interface for proper typing
 * Defines the structure of the @aws-sdk/client-bedrock-runtime module exports
 */
interface BedrockModuleExports {
  BedrockRuntimeClient: typeof BedrockRuntimeClient;
}

/**
 * Track if the Bedrock instrumentation is patched
 */
let _isBedrockPatched = false;

/**
 * Type guard that checks if the Bedrock instrumentation is currently enabled
 * @returns {boolean} True if instrumentation is active, false otherwise
 */
export function isPatched(): boolean {
  return _isBedrockPatched;
}

/**
 * An auto instrumentation class for AWS Bedrock that creates {@link https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md|OpenInference} Compliant spans for Bedrock API calls
 *
 * Supports instrumentation of:
 * - InvokeModel commands (synchronous)
 * - InvokeModelWithResponseStream commands (streaming)
 * - Converse commands (conversation API)
 *
 * @param instrumentationConfig The config for the instrumentation @see {@link InstrumentationConfig}
 * @param traceConfig The OpenInference trace configuration. Can be used to mask or redact sensitive information on spans. @see {@link TraceConfigOptions}
 */
export class BedrockInstrumentation extends InstrumentationBase<BedrockModuleExports> {
  static readonly COMPONENT = "@arizeai/openinference-instrumentation-bedrock";
  static readonly VERSION = VERSION;
  private oiTracer: OITracer;

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    /**
     * The config for the instrumentation
     * @see {@link InstrumentationConfig}
     */
    instrumentationConfig?: InstrumentationConfig;
    /**
     * The OpenInference trace configuration. Can be used to mask or redact sensitive information on spans.
     * @see {@link TraceConfigOptions}
     */
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super(
      BedrockInstrumentation.COMPONENT,
      BedrockInstrumentation.VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this.oiTracer = new OITracer({
      tracer: this.tracer,
      traceConfig,
    });
  }

  /**
   * Initializes the instrumentation module definition for AWS SDK Bedrock Runtime
   * @returns {InstrumentationModuleDefinition<BedrockModuleExports>[]} Array containing the module definition
   */
  protected init(): InstrumentationModuleDefinition<BedrockModuleExports>[] {
    const module =
      new InstrumentationNodeModuleDefinition<BedrockModuleExports>(
        MODULE_NAME,
        ["^3.0.0"],
        this.patch.bind(this),
        this.unpatch.bind(this),
      );
    return [module];
  }

  /**
   * Patches the BedrockRuntimeClient to intercept and instrument send() method calls
   * Wraps the send method to capture InvokeModel, InvokeModelWithResponseStream, and Converse commands
   *
   * @param moduleExports The module exports from @aws-sdk/client-bedrock-runtime
   * @param moduleVersion The version of the module being patched
   * @returns {BedrockModuleExports} The patched module exports
   */
  private patch(
    moduleExports: BedrockModuleExports,
    moduleVersion?: string,
  ): BedrockModuleExports {
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

  /**
   * Handles instrumentation for synchronous InvokeModel commands
   * Creates a span, extracts request attributes, executes the command, and processes the response
   *
   * @param command The InvokeModelCommand to instrument
   * @param original The original send method from the Bedrock client
   * @param client The Bedrock client instance
   * @returns {Promise<InvokeModelResponse>} The response from the InvokeModel call
   */
  private handleInvokeModelCommand(
    command: InvokeModelCommand,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    original: any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    client: any,
  ): Promise<InvokeModelResponse> {
    const span = this.oiTracer.startSpan("bedrock.invoke_model", {
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
      const result = original.apply(client, [
        command,
      ]) as Promise<InvokeModelResponse>;

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

  /**
   * Removes the instrumentation patch from the BedrockRuntimeClient
   * Unwraps the send method and resets the patched state
   *
   * @param moduleExports The module exports from @aws-sdk/client-bedrock-runtime
   * @param moduleVersion The version of the module being unpatched
   */
  private unpatch(
    moduleExports: BedrockModuleExports,
    moduleVersion?: string,
  ): void {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);

    if (moduleExports?.BedrockRuntimeClient) {
      this._unwrap(moduleExports.BedrockRuntimeClient.prototype, "send");
      _isBedrockPatched = false;
    }
  }

  /**
   * Handles instrumentation for streaming InvokeModel commands with guaranteed stream preservation
   *
   * This method ensures the original user stream is preserved regardless of instrumentation
   * success or failure. The user stream is returned immediately while instrumentation
   * happens asynchronously in the background using stream splitting.
   *
   * @param command The InvokeModelWithResponseStreamCommand to instrument
   * @param original The original send method from the Bedrock client
   * @param client The Bedrock client instance
   * @returns {Promise<{body: AsyncIterable<unknown>}>} Promise resolving to the response with preserved user stream
   */
  private handleInvokeModelWithResponseStreamCommand(
    command: InvokeModelWithResponseStreamCommand,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    original: any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    client: any,
  ) {
    const span = this.oiTracer.startSpan("bedrock.invoke_model", {
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
    extractInvokeModelRequestAttributes({
      span,
      command: command as unknown as InvokeModelCommand,
    });

    // Execute AWS SDK call and handle stream splitting outside error boundaries
    // This ensures the user stream is ALWAYS returned, regardless of instrumentation failures
    const result = original.apply(client, [command]) as Promise<{
      body: AsyncIterable<unknown>;
    }>;

    return result
      .then((response: { body: AsyncIterable<unknown> }) => {
        // Guard against missing response body - return original response
        if (!response.body) {
          span.recordException(new Error("Response body is undefined"));
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: "Response body is undefined",
          });
          span.end();
          return response;
        }

        // Split the stream safely - this preserves the original stream if splitting fails
        const { instrumentationStream, userStream } = safelySplitStream({
          originalStream: response.body,
        });

        // Start background instrumentation processing (non-blocking)
        // Only the instrumentation is wrapped in error handling - the user stream is protected
        if (instrumentationStream) {
          // @ts-expect-error - instrumentationStream is guaranteed non-null by the if check above
          consumeBedrockStreamChunks({
            stream: instrumentationStream,
            span,
          })
            .then(() => {
              span.setStatus({ code: SpanStatusCode.OK });
              span.end();
            })
            .catch((error: Error) => {
              span.recordException(error);
              span.setStatus({
                code: SpanStatusCode.ERROR,
                message: error.message,
              });
              span.end();
            });
        } else {
          // No instrumentation stream available, end span cleanly
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
        }

        // Return user stream immediately - instrumentation cannot interfere
        return { ...response, body: userStream };
      })
      .catch((error: Error) => {
        // If the AWS SDK call itself fails, record error but still try to return original response
        span.recordException(error);
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: error.message,
        });
        span.end();
        throw error; // Re-throw since we have no stream to return
      });
  }

  /**
   * Handles instrumentation for Converse API commands
   * Creates a span, extracts request attributes, executes the command, and processes the response
   *
   * @param command The ConverseCommand to instrument
   * @param original The original send method from the Bedrock client
   * @param client The Bedrock client instance
   * @returns {Promise<ConverseResponse>} The response from the Converse call
   */
  private handleConverseCommand(
    command: ConverseCommand,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    original: any,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    client: any,
  ): Promise<ConverseResponse> {
    const span = this.oiTracer.startSpan("bedrock.converse", {
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
      const result = original.apply(client, [
        command,
      ]) as Promise<ConverseResponse>;

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
