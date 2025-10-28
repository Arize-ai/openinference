import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import {
  context,
  diag,
  SpanKind,
  SpanStatusCode,
  Tracer,
  TracerProvider,
} from "@opentelemetry/api";
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
  ConverseStreamCommand,
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
import { consumeConverseStreamChunks } from "./attributes/converse-streaming-response-attributes";
import {
  getSystemFromModelId,
  setBasicSpanAttributes,
} from "./attributes/attribute-helpers";
import { isPromise } from "util/types";

const MODULE_NAME = "@aws-sdk/client-bedrock-runtime";
const INSTRUMENTATION_NAME = "@arizeai/openinference-instrumentation-bedrock";
const INSTRUMENTATION_VERSION = VERSION;

/**
 * AWS SDK module interface for proper typing
 * Defines the structure of the @aws-sdk/client-bedrock-runtime module exports
 */
interface BedrockModuleExports {
  BedrockRuntimeClient: typeof BedrockRuntimeClient;
}

// Strong types for Bedrock client send method and instance
type BedrockClient = InstanceType<typeof BedrockRuntimeClient>;
// Note: relax SendMethod here to satisfy _wrap signature expectations
// while preserving strong typing in handlers
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type SendMethod = any;

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
 * - ConverseStream commands (streaming conversation API)
 *
 * @param instrumentationConfig The config for the instrumentation @see {@link InstrumentationConfig}
 * @param traceConfig The OpenInference trace configuration. Can be used to mask or redact sensitive information on spans. @see {@link TraceConfigOptions}
 */
export class BedrockInstrumentation extends InstrumentationBase<BedrockModuleExports> {
  private oiTracer: OITracer;
  private tracerProvider?: TracerProvider;
  private traceConfig?: TraceConfigOptions;

  constructor({
    instrumentationConfig,
    traceConfig,
    tracerProvider,
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
    /**
     * An optional custom trace provider to be used for tracing. If not provided, a tracer will be created using the global tracer provider.
     * This is useful if you want to use a non-global tracer provider.
     *
     * @see {@link TracerProvider}
     */
    tracerProvider?: TracerProvider;
  } = {}) {
    super(
      INSTRUMENTATION_NAME,
      INSTRUMENTATION_VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this.tracerProvider = tracerProvider;
    this.traceConfig = traceConfig;
    this.oiTracer = new OITracer({
      tracer:
        this.tracerProvider?.getTracer(INSTRUMENTATION_NAME) ?? this.tracer,
      traceConfig,
    });
  }

  /**
   * Manually instruments the BedrockRuntimeClient, this allows for guaranteed patching of the module when import order is hard to control.
   */
  public manuallyInstrument(module: BedrockModuleExports) {
    diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
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

  get tracer(): Tracer {
    if (this.tracerProvider) {
      return this.tracerProvider.getTracer(this.instrumentationName);
    }
    return super.tracer;
  }

  setTracerProvider(tracerProvider: TracerProvider): void {
    super.setTracerProvider(tracerProvider);
    this.tracerProvider = tracerProvider;
    this.oiTracer = new OITracer({
      tracer: this.tracer,
      traceConfig: this.traceConfig,
    });
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
        (original: SendMethod) => {
          return function patchedSend(
            this: BedrockClient,
            ...args: Parameters<SendMethod>
          ) {
            const command = args[0];
            if (command instanceof InvokeModelCommand) {
              return instrumentation.handleInvokeModelCommand(
                args,
                command,
                original,
                this,
              );
            }

            if (command instanceof InvokeModelWithResponseStreamCommand) {
              return instrumentation.handleInvokeModelWithResponseStreamCommand(
                args,
                command,
                original,
                this,
              );
            }

            if (command instanceof ConverseCommand) {
              return instrumentation.handleConverseCommand(
                args,
                command,
                original,
                this,
              );
            }

            if (command instanceof ConverseStreamCommand) {
              return instrumentation.handleConverseStreamCommand(
                args,
                command,
                original,
                this,
              );
            }

            // Pass through other commands without instrumentation
            return original.apply(this, args);
          };
        },
      );

      _isBedrockPatched = true;
    }

    return moduleExports;
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
   * Handles instrumentation for synchronous InvokeModel commands
   * Creates a span, extracts request attributes, executes the command, and processes the response
   *
   * @param command The InvokeModelCommand to instrument
   * @param original The original send method from the Bedrock client
   * @param client The Bedrock client instance
   * @returns {Promise<InvokeModelResponse>} The response from the InvokeModel call
   */
  private handleInvokeModelCommand(
    args: Parameters<SendMethod>,
    command: InvokeModelCommand,
    original: SendMethod,
    client: BedrockClient,
  ): Promise<InvokeModelResponse> {
    const span = this.oiTracer.startSpan("bedrock.invoke_model", {
      kind: SpanKind.INTERNAL,
    });

    const modelId = command.input.modelId;
    const system = getSystemFromModelId(modelId ?? "");
    setBasicSpanAttributes(span, system);

    const contextAttributes = getAttributesFromContext(context.active());
    span.setAttributes(contextAttributes);

    extractInvokeModelRequestAttributes({ span, command, system });

    try {
      const result = original.apply(client, args);

      // AWS SDK v3 send() method should always return a Promise
      if (!isPromise(result)) {
        diag.warn(
          "Expected Promise from AWS SDK send method, got:",
          typeof result,
        );
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: "Unexpected return type from AWS SDK",
        });
        span.end();
        return result as Promise<InvokeModelResponse>;
      }

      return (result as Promise<InvokeModelResponse>)
        .then((response: InvokeModelResponse) => {
          extractInvokeModelResponseAttributes({
            span,
            response,
            modelType: system,
          });
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
    args: Parameters<SendMethod>,
    command: InvokeModelWithResponseStreamCommand,
    original: SendMethod,
    client: BedrockClient,
  ) {
    const span = this.oiTracer.startSpan("bedrock.invoke_model", {
      kind: SpanKind.INTERNAL,
    });

    const modelId = command.input.modelId;
    const system = getSystemFromModelId(modelId ?? "");
    setBasicSpanAttributes(span, system);

    const contextAttributes = getAttributesFromContext(context.active());
    span.setAttributes(contextAttributes);

    extractInvokeModelRequestAttributes({
      span,
      command: command as unknown as InvokeModelCommand,
      system,
    });

    // Execute AWS SDK call and handle stream splitting outside error boundaries
    // This ensures the user stream is ALWAYS returned, regardless of instrumentation failures
    const result = original.apply(client, args);

    // Validate that we got a Promise as expected
    if (!isPromise(result)) {
      diag.warn(
        "Expected Promise from AWS SDK send method, got:",
        typeof result,
      );
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: "Unexpected return type from AWS SDK",
      });
      span.end();
      return result as Promise<{ body: AsyncIterable<unknown> }>;
    }

    return (result as Promise<{ body: AsyncIterable<unknown> }>)
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

        const { instrumentationStream, userStream } = safelySplitStream({
          originalStream: response.body,
        });

        // Start background instrumentation processing (non-blocking)
        // Only the instrumentation is wrapped in error handling - the user stream is protected
        if (instrumentationStream) {
          const consumed = consumeBedrockStreamChunks({
            stream: instrumentationStream,
            span,
            modelType: system,
          });
          if (consumed == null) {
            span.setStatus({ code: SpanStatusCode.OK });
            span.end();
          } else {
            consumed
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
          }
        } else {
          diag.debug(
            "No instrumentation stream available, ending span cleanly",
          );
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
    args: Parameters<SendMethod>,
    command: ConverseCommand,
    original: SendMethod,
    client: BedrockClient,
  ): Promise<ConverseResponse> {
    const span = this.oiTracer.startSpan("bedrock.converse", {
      kind: SpanKind.INTERNAL,
    });

    const modelId = command.input.modelId;
    const system = getSystemFromModelId(modelId ?? "");
    setBasicSpanAttributes(span, system);

    const contextAttributes = getAttributesFromContext(context.active());
    span.setAttributes(contextAttributes);

    extractConverseRequestAttributes({ span, command });

    try {
      const result = original.apply(client, args);

      // AWS SDK v3 send() method should always return a Promise
      if (!isPromise(result)) {
        diag.warn(
          "Expected Promise from AWS SDK send method, got:",
          typeof result,
        );
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: "Unexpected return type from AWS SDK",
        });
        span.end();
        return result as Promise<ConverseResponse>;
      }

      return (result as Promise<ConverseResponse>)
        .then((response: ConverseResponse) => {
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

  /**
   * Handles instrumentation for streaming Converse commands with guaranteed stream preservation
   *
   * This method ensures the original user stream is preserved regardless of instrumentation
   * success or failure. The user stream is returned immediately while instrumentation
   * happens asynchronously in the background using stream splitting.
   *
   * @param command The ConverseStreamCommand to instrument
   * @param original The original send method from the Bedrock client
   * @param client The Bedrock client instance
   * @returns {Promise<{stream: AsyncIterable<unknown>}>} Promise resolving to the response with preserved user stream
   */
  private handleConverseStreamCommand(
    args: Parameters<SendMethod>,
    command: ConverseStreamCommand,
    original: SendMethod,
    client: BedrockClient,
  ) {
    const span = this.oiTracer.startSpan("bedrock.converse", {
      kind: SpanKind.INTERNAL,
    });

    const modelId = command.input.modelId;
    const system = getSystemFromModelId(modelId ?? "");
    setBasicSpanAttributes(span, system);

    const contextAttributes = getAttributesFromContext(context.active());
    span.setAttributes(contextAttributes);

    extractConverseRequestAttributes({
      span,
      command: command,
    });

    // Execute AWS SDK call and handle stream splitting outside error boundaries
    // This ensures the user stream is ALWAYS returned, regardless of instrumentation failures
    const result = original.apply(client, args);

    // Validate that we got a Promise as expected
    if (!isPromise(result)) {
      diag.warn(
        "Expected Promise from AWS SDK send method, got:",
        typeof result,
      );
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: "Unexpected return type from AWS SDK",
      });
      span.end();
      return result as Promise<{ stream: AsyncIterable<unknown> }>;
    }

    return (result as Promise<{ stream: AsyncIterable<unknown> }>)
      .then((response: { stream: AsyncIterable<unknown> }) => {
        // Guard against missing response stream - return original response
        if (!response.stream) {
          span.recordException(new Error("Response stream is undefined"));
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: "Response stream is undefined",
          });
          span.end();
          return response;
        }

        const { instrumentationStream, userStream } = safelySplitStream({
          originalStream: response.stream,
        });

        // Start background instrumentation processing (non-blocking)
        // Only the instrumentation is wrapped in error handling - the user stream is protected
        if (instrumentationStream) {
          // @ts-expect-error - instrumentationStream is guaranteed non-null by the if check above
          consumeConverseStreamChunks({
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
          diag.debug(
            "No instrumentation stream available, ending span cleanly",
          );
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
        }

        // Return user stream immediately - instrumentation cannot interfere
        return { ...response, stream: userStream };
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
}
