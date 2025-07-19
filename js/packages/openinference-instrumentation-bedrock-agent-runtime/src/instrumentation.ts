import "./instrumentation";

import {
    InstrumentationBase,
    InstrumentationConfig,
    InstrumentationModuleDefinition,
    InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import {diag, SpanKind, SpanStatusCode} from "@opentelemetry/api";
import {OITracer, TraceConfigOptions} from "@arizeai/openinference-core";
import {VERSION} from "./version";
import {InvokeAgentCommand} from "@aws-sdk/client-bedrock-agent-runtime";
import {extractBaseRequestAttributes} from "./attributes/request-attributes";
import {interceptAgentResponse} from "./stream-utils";
import {ResponseHandler} from "../scripts/response-handler";
import * as console from "node:console";

const MODULE_NAME = "@aws-sdk/client-bedrock-agent-runtime";

let _isBedrockAgentPatched = false;

export function isPatched() {
    return _isBedrockAgentPatched;
}


export interface BedrockAgentInstrumentationConfig extends InstrumentationConfig {
    traceConfig?: TraceConfigOptions;
}

export class BedrockAgentInstrumentation extends InstrumentationBase<BedrockAgentInstrumentationConfig> {
    static readonly COMPONENT = "@arizeai/openinference-instrumentation-bedrock-agent";
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

    private patch(moduleExports: any, moduleVersion?: string) {
        diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
        if (moduleExports?.BedrockAgentRuntimeClient) {
            const instrumentation = this;
            this._wrap(
                moduleExports.BedrockAgentRuntimeClient.prototype,
                "send",
                (original: any) => {
                    return function patchedSend(this: unknown, command: any) {
                        if (command?.constructor?.name === "InvokeAgentCommand") {
                            return instrumentation._handleInvokeAgentCommand(
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
        }
        return moduleExports;
    }

    private _handleInvokeAgentCommand(
        command: InvokeAgentCommand,
        original: any,
        client: any,
    ) {
        const span = this.oiTracer.startSpan("bedrock.invoke_agent", {
            kind: SpanKind.CLIENT,
            attributes: extractBaseRequestAttributes(command)
        });
        const result = original.apply(client, [command]);
        return result.then((response: any) => {
            const callback = new ResponseHandler(span);
            if (response.completion && Symbol.asyncIterator in response.completion) {
                response.completion = interceptAgentResponse(response.completion, callback);
            }
            else{
                console.log(response.trace);
            }
            return response;
        }).catch((err: any) => {
            span.recordException(err);
            span.setStatus({code: SpanStatusCode.ERROR, message: err.message});
            span.end();
            throw err;
        });

    }

    private unpatch(moduleExports: any, moduleVersion?: string) {
        diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);
        if (moduleExports?.BedrockAgentRuntimeClient) {
            this._unwrap(moduleExports.BedrockAgentRuntimeClient.prototype, "send");
            _isBedrockAgentPatched = false;
        }
    }
}
