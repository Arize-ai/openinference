import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import { diag, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { OITracer, TraceConfigOptions } from "@arizeai/openinference-core";
import { VERSION } from "./version";
import { InvokeAgentCommand } from "@aws-sdk/client-bedrock-agent-runtime";
import { extractBaseRequestAttributes } from "./attributes/request-attributes";
import { interceptAgentResponse } from "./stream-utils";
import { ResponseHandler } from "./response-handler";
import { InvokeAgentCommandOutput } from "@aws-sdk/client-bedrock-agent-runtime/dist-types/commands/InvokeAgentCommand";

const MODULE_NAME = "@aws-sdk/client-bedrock-agent-runtime";

let _isBedrockAgentPatched = false;

export function isPatched() {
  return _isBedrockAgentPatched;
}

export interface BedrockAgentInstrumentationConfig
  extends InstrumentationConfig {
  traceConfig?: TraceConfigOptions;
}

export class BedrockAgentInstrumentation extends InstrumentationBase<BedrockAgentInstrumentationConfig> {
  static readonly COMPONENT =
    "@arizeai/openinference-instrumentation-bedrock-agent";
  static readonly VERSION = VERSION;

  private oiTracer: OITracer;

  constructor(config: BedrockAgentInstrumentationConfig = {}) {
    super(
      BedrockAgentInstrumentation.COMPONENT,
      BedrockAgentInstrumentation.VERSION,
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

  private getBedrockAgentModule(moduleExports: unknown):
    | {
        BedrockAgentRuntimeClient: {
          prototype: {
            send: (
              command: InvokeAgentCommand,
            ) => Promise<InvokeAgentCommandOutput>;
          };
        };
      }
    | undefined {
    if (
      typeof moduleExports === "object" &&
      moduleExports !== null &&
      "BedrockAgentRuntimeClient" in moduleExports
    ) {
      return moduleExports as {
        BedrockAgentRuntimeClient: {
          prototype: {
            send: (
              command: InvokeAgentCommand,
            ) => Promise<InvokeAgentCommandOutput>;
          };
        };
      };
    }
    return undefined;
  }

  private patch(moduleExports: unknown, moduleVersion?: string) {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
    if (_isBedrockAgentPatched) return moduleExports;
    const bedrockModule = this.getBedrockAgentModule(moduleExports);
    if (!bedrockModule) return moduleExports;
    this._wrap(
      bedrockModule.BedrockAgentRuntimeClient.prototype,
      "send",
      (
        original: (
          command: InvokeAgentCommand,
        ) => Promise<InvokeAgentCommandOutput>,
      ) => {
        /* eslint-disable @typescript-eslint/no-this-alias */
        const instrumentationInstance = this;
        return function patchedSend(
          this: typeof bedrockModule.BedrockAgentRuntimeClient.prototype,
          command: InvokeAgentCommand,
        ) {
          if (command?.constructor?.name === "InvokeAgentCommand") {
            return instrumentationInstance._handleInvokeAgentCommand(
              command,
              original,
              this,
            );
          }
          return original.apply(this, [command]);
        };
        /* eslint-disable @typescript-eslint/no-this-alias */
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
      kind: SpanKind.CLIENT,
      attributes: extractBaseRequestAttributes(command),
    });
    const result = original.apply(client, [command]);
    return result
      .then((response: InvokeAgentCommandOutput) => {
        const callback = new ResponseHandler(span);
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

  private unpatch(moduleExports: unknown, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);
    const bedrockModule = this.getBedrockAgentModule(moduleExports);
    if (!bedrockModule) return moduleExports;
    this._unwrap(bedrockModule.BedrockAgentRuntimeClient.prototype, "send");
    _isBedrockAgentPatched = false;
  }
}
