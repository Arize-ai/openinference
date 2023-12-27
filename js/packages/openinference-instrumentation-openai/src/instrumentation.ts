import * as openai from "openai";
import {
    InstrumentationBase,
    InstrumentationConfig,
    InstrumentationModuleDefinition,
    InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import { context, trace, isSpanContextValid, Span } from "@opentelemetry/api";
import { VERSION } from "./version";

export class OpenAIInstrumentation extends InstrumentationBase<typeof openai> {
    constructor(config?: InstrumentationConfig) {
        super(
            "@arizeai/openinference-instrumentation-openai",
            VERSION,
            Object.assign({}, config)
        );
    }

    protected init(): InstrumentationModuleDefinition<typeof openai> {
        const module = new InstrumentationNodeModuleDefinition<typeof openai>(
            "openai",
            [">=4.24.1<5"],
            this.patch.bind(this),
            this.unpatch.bind(this)
        );
        return module;
    }
    private patch(
        module: typeof openai & { openInferencePatched?: boolean },
        moduleVersion: string | undefined
    ) {
        console.log("patching...");
        const plugin = this;
        if (module?.openInferencePatched) {
            return module;
        }
        console.log("wrapping");
        this._wrap(
            module.OpenAI.Chat.Completions.prototype,
            "create",
            (original: Function) => {
                console.log("wrapped");
                return function patchedCreate(...args) {
                    return original.apply(plugin, args);
                };
            }
        );
        module.openInferencePatched = true;
        return module;
    }
    private unpatch(moduleExports: typeof openai) {}
}
