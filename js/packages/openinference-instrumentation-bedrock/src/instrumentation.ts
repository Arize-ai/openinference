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
} from "@opentelemetry/api";
import { OITracer, TraceConfigOptions } from "@arizeai/openinference-core";
import { VERSION } from "./version";
import { InvokeModelCommand } from "./types/bedrock-types";
import { extractInvokeModelRequestAttributes } from "./attributes/request-attributes";
import { extractInvokeModelResponseAttributes } from "./attributes/response-attributes";

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

  private _handleInvokeModelCommand(command: InvokeModelCommand, original: any, client: any) {
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


  private unpatch(moduleExports: any, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);

    if (moduleExports?.BedrockRuntimeClient) {
      this._unwrap(moduleExports.BedrockRuntimeClient.prototype, "send");
    }
  }
}
