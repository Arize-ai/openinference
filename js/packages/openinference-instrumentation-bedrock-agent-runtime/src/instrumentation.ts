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
import { extractBaseRequestAttributes } from "./attributes/requestAttributes";
import { interceptAgentResponse } from "./streamUtils";
import { CallbackHandler } from "./callback-handler";
import { InvokeAgentCommandOutput } from "@aws-sdk/client-bedrock-agent-runtime/dist-types/commands/InvokeAgentCommand";
import * as bedrockAgentRunTime from "@aws-sdk/client-bedrock-agent-runtime";

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
    this._wrap(
      moduleExports.BedrockAgentRuntimeClient.prototype,
      "send",
      (
        original: (
          command: InvokeAgentCommand,
        ) => Promise<InvokeAgentCommandOutput>,
      ) => {
        /* eslint-disable @typescript-eslint/no-this-alias */
        const instrumentationInstance = this;
        return function patchedSend(
          this: typeof moduleExports.BedrockAgentRuntimeClient.prototype,
          command: InvokeAgentCommand,
        ) {
          if (command instanceof InvokeAgentCommand) {
            return instrumentationInstance._handleInvokeAgentCommand(
              command,
              original,
              this,
            );
          }
          return original.apply(this, [command]);
        };
      },
    );
    _isBedrockAgentPatched = true;
    return moduleExports;
  }

  private _handleInvokeAgentCommand(
    command: InvokeAgentCommand,
    original: (
      command: InvokeAgentCommand,
    ) => Promise<InvokeAgentCommandOutput>,
    client: unknown,
  ) {
    const span = this.oiTracer.startSpan("bedrock.invoke_agent", {
      kind: SpanKind.INTERNAL,
      attributes: extractBaseRequestAttributes(command),
    });
    const result = original.apply(client, [command]);
    return result
      .then((response: InvokeAgentCommandOutput) => {
        const callback = new CallbackHandler(this.oiTracer, span);
        if (
          response.completion &&
          Symbol.asyncIterator in response.completion
        ) {
          response.completion = interceptAgentResponse(
            response.completion,
            callback,
          );
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
