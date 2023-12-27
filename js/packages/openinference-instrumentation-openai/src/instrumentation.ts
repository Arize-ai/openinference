import * as openai from "openai";
import {
    InstrumentationBase,
    InstrumentationConfig,
    InstrumentationModuleDefinition,
    InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import {
    diag,
    context,
    trace,
    isSpanContextValid,
    Span,
    SpanKind,
} from "@opentelemetry/api";
import { VERSION } from "./version";
import {
    SemanticConventions,
    OpenInferenceSpanKind,
} from "@arizeai/openinference-semantic-conventions";

const MODULE_NAME = "openai";

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
            ["^4.0.0"],
            this.patch.bind(this),
            this.unpatch.bind(this)
        );
        return module;
    }
    private patch(
        module: typeof openai & { openInferencePatched?: boolean },
        moduleVersion?: string
    ) {
        console.log(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
        const plugin = this;
        if (module?.openInferencePatched) {
            return module;
        }
        const instrumentation: OpenAIInstrumentation = this;
        this._wrap(
            module.OpenAI.Chat.Completions.prototype,
            "create",
            (
                original: typeof module.OpenAI.Chat.Completions.prototype.create
            ) => {
                return function patchedCreate(
                    this: unknown,
                    ...args: Parameters<
                        typeof module.OpenAI.Chat.Completions.prototype.create
                    >
                ) {
                    instrumentation.tracer.startSpan(
                        `OpenAI Chat Completions`,
                        {
                            kind: SpanKind.CLIENT,
                            attributes: {
                                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                                    OpenInferenceSpanKind.LLM,
                            },
                        }
                    );
                    return original.apply(this, args);
                };
            }
        );
        module.openInferencePatched = true;
        return module;
    }
    private unpatch(moduleExports: typeof openai, moduleVersion?: string) {
        diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);
    }
}
