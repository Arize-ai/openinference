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
    SpanStatusCode,
} from "@opentelemetry/api";
import { VERSION } from "./version";
import {
    SemanticConventions,
    OpenInferenceSpanKind,
    MimeType,
} from "@arizeai/openinference-semantic-conventions";
import {
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionCreateParamsBase,
} from "openai/resources/chat/completions";
import { Stream } from "openai/streaming";

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
        diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
        if (module?.openInferencePatched) {
            return module;
        }
        const instrumentation: OpenAIInstrumentation = this;
        type CompletionCreateType =
            typeof module.OpenAI.Chat.Completions.prototype.create;

        // Patch create chat completions
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
                    const { messages, ...invocationParameters } = body;
                    const span = instrumentation.tracer.startSpan(
                        `OpenAI Chat Completions`,
                        {
                            kind: SpanKind.CLIENT,
                            attributes: {
                                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                                    OpenInferenceSpanKind.LLM,
                                [SemanticConventions.LLM_MODEL_NAME]:
                                    body.model,
                                [SemanticConventions.INPUT_VALUE]:
                                    JSON.stringify(body),
                                [SemanticConventions.INPUT_MIME_TYPE]:
                                    MimeType.JSON,
                                [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
                                    JSON.stringify(invocationParameters),
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
                            // Push the error to the span
                            if (error) {
                                span.recordException(error);
                            }
                        }
                    );
                    const wrappedPromise = execPromise.then((result) => {
                        if (result) {
                            // Record the results
                            span.setAttributes({
                                [SemanticConventions.OUTPUT_VALUE]:
                                    JSON.stringify(result),
                                [SemanticConventions.OUTPUT_MIME_TYPE]:
                                    MimeType.JSON,
                                ...getLLMOutputMessagesAttributes(result),
                                ...getUsageAttributes(result),
                            });
                        }
                        span.setStatus({ code: SpanStatusCode.OK });
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

/**
 * Get Usage attributes
 */
function getUsageAttributes(
    response: Stream<ChatCompletionChunk> | ChatCompletion
) {
    if (response.hasOwnProperty("usage")) {
        const completion = response as ChatCompletion;
        // TODO fill this out
        return {};
    }
    return {};
}

/**
 * Converts the result to LLM output attributes
 */
function getLLMOutputMessagesAttributes(
    response: Stream<ChatCompletionChunk> | ChatCompletion
): Attributes {
    // Handle chat completion
    if (response.hasOwnProperty("choices")) {
        const completion = response as ChatCompletion;
        // Right now support just the first choice
        const choice = completion.choices[0];
        if (!choice) {
            return {};
        }
        return [choice.message].reduce((acc, message, index) => {
            const index_prefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${index}`;
            acc[`${index_prefix}.${SemanticConventions.MESSAGE_CONTENT}`] =
                String(message.content);
            acc[`${index_prefix}.${SemanticConventions.MESSAGE_ROLE}`] =
                message.role;
            return acc;
        }, {} as Attributes);
    }
    return {};
}
