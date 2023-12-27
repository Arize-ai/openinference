import * as openai from "openai";
import {
    InstrumentationBase,
    InstrumentationConfig,
    InstrumentationModuleDefinition,
    InstrumentationNodeModuleDefinition,
    safeExecuteInTheMiddle,
} from "@opentelemetry/instrumentation";
import {
    diag,
    context,
    trace,
    isSpanContextValid,
    Span,
    SpanKind,
    Attributes,
} from "@opentelemetry/api";
import { VERSION } from "./version";
import {
    SemanticConventions,
    OpenInferenceSpanKind,
} from "@arizeai/openinference-semantic-conventions";
import { ChatCompletionCreateParamsBase } from "openai/resources/chat/completions";

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
        type CompletionCreateType =
            typeof module.OpenAI.Chat.Completions.prototype.create;
        this._wrap(
            module.OpenAI.Chat.Completions.prototype,
            "create",
            (original: CompletionCreateType): any => {
                return function patchedCreate(
                    this: unknown,
                    ...args: Parameters<
                        typeof module.OpenAI.Chat.Completions.prototype.create
                    >
                ) {
                    const body = args[0];
                    const span = instrumentation.tracer.startSpan(
                        `OpenAI Chat Completions`,
                        {
                            kind: SpanKind.CLIENT,
                            attributes: {
                                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                                    OpenInferenceSpanKind.LLM,
                                ...getLLMInputMessagesAttributes(body),
                            },
                        }
                    );
                    const execContext = trace.setSpan(context.active(), span);
                    const execPromise = safeExecuteInTheMiddle<
                        ReturnType<CompletionCreateType>
                    >(
                        () => {
                            return context.with(execContext, () => {
                                return original.apply(this, args);
                            });
                        },
                        (error) => {
                            // TODO handler the error
                        }
                    );
                    const wrappedPromise = execPromise.then((result) => {
                        // TODO set span attributes
                        span.end();
                        return result;
                    });
                    return context.bind(execContext, wrappedPromise);
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

/**
 * Converts the body input
 */
function getLLMInputMessagesAttributes(
    body: ChatCompletionCreateParamsBase
): Attributes {
    return body.messages.reduce((acc, message, index) => {
        const index_prefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}`;
        acc[`${index_prefix}.${SemanticConventions.MESSAGE_CONTENT}`] = String(
            message.content
        );
        acc[`${index_prefix}.${SemanticConventions.MESSAGE_ROLE}`] = String(
            message.role
        );
        return acc;
    }, {} as Attributes);
}
