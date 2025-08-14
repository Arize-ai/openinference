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
  Tracer,
  TracerProvider,
} from "@opentelemetry/api";
import { OITracer, TraceConfigOptions } from "@arizeai/openinference-core";
import { VERSION } from "./version";
import { InvokeAgentCommand } from "@aws-sdk/client-bedrock-agent-runtime";
import { safelyExtractBaseRequestAttributes } from "./attributes/requestAttributes";
import { interceptAgentResponse } from "./streamUtils";
import { CallbackHandler } from "./callbackHandler";
import type { InvokeAgentCommandOutput } from "@aws-sdk/client-bedrock-agent-runtime/dist-types/commands/InvokeAgentCommand";
import type * as bedrockAgentRunTime from "@aws-sdk/client-bedrock-agent-runtime";

const MODULE_NAME = "@aws-sdk/client-bedrock-agent-runtime";

const INSTRUMENTATION_NAME =
  "@arizeai/openinference-instrumentation-bedrock-agent";

let _isBedrockAgentPatched = false;

export function isPatched() {
  return _isBedrockAgentPatched;
}

export class BedrockAgentInstrumentation extends InstrumentationBase<InstrumentationConfig> {
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
      VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this.tracerProvider = tracerProvider;
    this.traceConfig = traceConfig;
    this.oiTracer = new OITracer({
      tracer:
        this.tracerProvider?.getTracer(INSTRUMENTATION_NAME) ?? this.tracer,
      traceConfig: traceConfig,
    });
  }

  /**
   * Manually patches the BedrockAgentRuntimeClient, this allows for guaranteed patching of the module when import order is hard to control.
   */
  public manuallyInstrument(module: typeof bedrockAgentRunTime) {
    diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
  }

  protected init(): InstrumentationModuleDefinition<
    typeof bedrockAgentRunTime
  >[] {
    const module = new InstrumentationNodeModuleDefinition<
      typeof bedrockAgentRunTime
    >(MODULE_NAME, ["^3.0.0"], this.patch.bind(this), this.unpatch.bind(this));
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

  private patch(
    moduleExports: typeof bedrockAgentRunTime,
    moduleVersion?: string,
  ) {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
    if (_isBedrockAgentPatched) return moduleExports;
    if (!moduleExports) return moduleExports;

    type SendMethod =
      typeof moduleExports.BedrockAgentRuntimeClient.prototype.send;
    this._wrap(
      moduleExports.BedrockAgentRuntimeClient.prototype,
      "send",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (original: SendMethod): any => {
        /* eslint-disable @typescript-eslint/no-this-alias */
        const instrumentationInstance = this;
        return function patchedSend(
          this: typeof moduleExports.BedrockAgentRuntimeClient.prototype,
          ...args: Parameters<SendMethod>
        ) {
          const command = args[0];
          if (command instanceof InvokeAgentCommand) {
            return instrumentationInstance._handleInvokeAgentCommand(
              args,
              command,
              original,
              this,
            );
          }
          return original.apply(this, args);
        };
      },
    );
    _isBedrockAgentPatched = true;
    return moduleExports;
  }

  private _handleInvokeAgentCommand(
    args: Parameters<
      typeof bedrockAgentRunTime.BedrockAgentRuntimeClient.prototype.send
    >,
    command: bedrockAgentRunTime.InvokeAgentCommand,
    original: (
      ...args: Parameters<
        typeof bedrockAgentRunTime.BedrockAgentRuntimeClient.prototype.send
      >
    ) => Promise<InvokeAgentCommandOutput>,
    client: unknown,
  ) {
    const span = this.oiTracer.startSpan("bedrock.invoke_agent", {
      kind: SpanKind.INTERNAL,
      attributes: safelyExtractBaseRequestAttributes(command) ?? undefined,
    });
    const result = original.apply(client, args);
    return result
      .then((response: InvokeAgentCommandOutput) => {
        const callback = new CallbackHandler(this.oiTracer, span);
        if (
          response.completion &&
          Symbol.asyncIterator in response.completion
        ) {
          try {
            response.completion = interceptAgentResponse(
              response.completion,
              callback,
            );
          } catch (err: unknown) {
            let errorMessage: string;
            if (err instanceof Error || typeof err === "string") {
              diag.warn(
                `Openinference warning, unable to intercept agent response, some spans may be missing or incomplete. Error: ${err instanceof Error ? err.message : err}`,
              );
              span.recordException(err);
              errorMessage = err instanceof Error ? err.message : err;
            } else {
              errorMessage =
                "Unknown error occurred in Openinference BedrockAgentInstrumentation while trying to process agent response.";
            }

            diag.warn(
              `Openinference warning, unable to intercept agent response, some spans may be missing or incomplete. Error: ${errorMessage}`,
            );
            span.setStatus({
              code: SpanStatusCode.ERROR,
              message: errorMessage,
            });
            span.end();
          }
        } else {
          // End the span if response.completion is not a stream
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
        }
        return response;
      })
      .catch((err: Error) => {
        span.recordException(err);
        span.setStatus({ code: SpanStatusCode.ERROR, message: err.message });
        span.end();
        throw err;
      });
  }

  private unpatch(
    moduleExports: typeof bedrockAgentRunTime,
    moduleVersion?: string,
  ) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);
    if (!moduleExports) return moduleExports;
    this._unwrap(moduleExports.BedrockAgentRuntimeClient.prototype, "send");
    _isBedrockAgentPatched = false;
  }
}
