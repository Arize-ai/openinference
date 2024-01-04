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
import {
  CreateEmbeddingResponse,
  EmbeddingCreateParams,
} from "openai/resources";

const MODULE_NAME = "openai";

export class OpenAIInstrumentation extends InstrumentationBase<typeof openai> {
  constructor(config?: InstrumentationConfig) {
    super(
      "@arizeai/openinference-instrumentation-openai",
      VERSION,
      Object.assign({}, config),
    );
  }

  protected init(): InstrumentationModuleDefinition<typeof openai> {
    const module = new InstrumentationNodeModuleDefinition<typeof openai>(
      "openai",
      ["^4.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }
  /**
   * Patches the OpenAI module
   */
  private patch(
    module: typeof openai & { openInferencePatched?: boolean },
    moduleVersion?: string,
  ) {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
    if (module?.openInferencePatched) {
      return module;
    }
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation: OpenAIInstrumentation = this;
    type CompletionCreateType =
      typeof module.OpenAI.Chat.Completions.prototype.create;

    // Patch create chat completions
    this._wrap(
      module.OpenAI.Chat.Completions.prototype,
      "create",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (original: CompletionCreateType): any => {
        return function patchedCreate(
          this: unknown,
          ...args: Parameters<
            typeof module.OpenAI.Chat.Completions.prototype.create
          >
        ) {
          const body = args[0];
          const { messages: _messages, ...invocationParameters } = body;
          const span = instrumentation.tracer.startSpan(
            `OpenAI Chat Completions`,
            {
              kind: SpanKind.CLIENT,
              attributes: {
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
                [SemanticConventions.LLM_MODEL_NAME]: body.model,
                [SemanticConventions.INPUT_VALUE]: JSON.stringify(body),
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
                [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
                  JSON.stringify(invocationParameters),
                ...getLLMInputMessagesAttributes(body),
              },
            },
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
            },
          );
          const wrappedPromise = execPromise.then((result) => {
            if (result) {
              // Record the results
              span.setAttributes({
                [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(result),
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
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
      },
    );

    // Patch embeddings
    type EmbeddingsCreateType =
      typeof module.OpenAI.Embeddings.prototype.create;
    this._wrap(
      module.OpenAI.Embeddings.prototype,
      "create",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (original: EmbeddingsCreateType): any => {
        return function patchedEmbeddingCreate(
          this: unknown,
          ...args: Parameters<typeof module.OpenAI.Embeddings.prototype.create>
        ) {
          const body = args[0];
          const { input } = body;
          const isStringInput = typeof input == "string";
          const span = instrumentation.tracer.startSpan(`OpenAI Embeddings`, {
            kind: SpanKind.CLIENT,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.EMBEDDING,
              [SemanticConventions.EMBEDDING_MODEL_NAME]: body.model,
              [SemanticConventions.INPUT_VALUE]: isStringInput
                ? input
                : JSON.stringify(input),
              [SemanticConventions.INPUT_MIME_TYPE]: isStringInput
                ? MimeType.TEXT
                : MimeType.JSON,
              ...getEmbeddingTextAttributes(body),
            },
          });
          const execContext = trace.setSpan(context.active(), span);
          const execPromise = safeExecuteInTheMiddle<
            ReturnType<EmbeddingsCreateType>
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
            },
          );
          const wrappedPromise = execPromise.then((result) => {
            if (result) {
              // Record the results
              span.setAttributes({
                // Do not record the output data as it can be large
                ...getEmbeddingEmbeddingsAttributes(result),
              });
            }
            span.setStatus({ code: SpanStatusCode.OK });
            span.end();
            return result;
          });
          return context.bind(execContext, wrappedPromise);
        };
      },
    );

    module.openInferencePatched = true;
    return module;
  }
  /**
   * Un-patches the OpenAI module's chat completions API
   */
  private unpatch(moduleExports: typeof openai, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);
    this._unwrap(moduleExports.OpenAI.Chat.Completions.prototype, "create");
    this._unwrap(moduleExports.OpenAI.Embeddings.prototype, "create");
  }
}

/**
 * Converts the body of the request to LLM input messages
 */
function getLLMInputMessagesAttributes(
  body: ChatCompletionCreateParamsBase,
): Attributes {
  return body.messages.reduce((acc, message, index) => {
    const index_prefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}`;
    acc[`${index_prefix}.${SemanticConventions.MESSAGE_CONTENT}`] = String(
      message.content,
    );
    acc[`${index_prefix}.${SemanticConventions.MESSAGE_ROLE}`] = String(
      message.role,
    );
    return acc;
  }, {} as Attributes);
}

/**
 * Get Usage attributes
 */
function getUsageAttributes(
  response: Stream<ChatCompletionChunk> | ChatCompletion,
) {
  if (Object.prototype.hasOwnProperty.call(response, "usage")) {
    const completion = response as ChatCompletion;
    if (completion.usage) {
      return {
        [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]:
          completion.usage.completion_tokens,
        [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]:
          completion.usage.prompt_tokens,
        [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]:
          completion.usage.total_tokens,
      };
    }
  }
  return {};
}

/**
 * Converts the result to LLM output attributes
 */
function getLLMOutputMessagesAttributes(
  response: Stream<ChatCompletionChunk> | ChatCompletion,
): Attributes {
  // Handle chat completion
  if (Object.prototype.hasOwnProperty.call(response, "choices")) {
    const completion = response as ChatCompletion;
    // Right now support just the first choice
    const choice = completion.choices[0];
    if (!choice) {
      return {};
    }
    return [choice.message].reduce((acc, message, index) => {
      const index_prefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${index}`;
      acc[`${index_prefix}.${SemanticConventions.MESSAGE_CONTENT}`] = String(
        message.content,
      );
      acc[`${index_prefix}.${SemanticConventions.MESSAGE_ROLE}`] = message.role;
      return acc;
    }, {} as Attributes);
  }
  return {};
}

/**
 * Converts the embedding result payload to embedding attributes
 */
function getEmbeddingTextAttributes(
  request: EmbeddingCreateParams,
): Attributes {
  if (typeof request.input == "string") {
    return {
      [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
        request.input,
    };
  } else if (
    Array.isArray(request.input) &&
    request.input.length > 0 &&
    typeof request.input[0] == "string"
  ) {
    return request.input.reduce((acc, input, index) => {
      const index_prefix = `${SemanticConventions.EMBEDDING_EMBEDDINGS}.${index}`;
      acc[`${index_prefix}.${SemanticConventions.EMBEDDING_TEXT}`] = input;
      return acc;
    }, {} as Attributes);
  }
  // Ignore other cases where input is a number or an array of numbers
  return {};
}

/**
 * Converts the embedding result payload to embedding attributes
 */
function getEmbeddingEmbeddingsAttributes(
  response: CreateEmbeddingResponse,
): Attributes {
  return response.data.reduce((acc, embedding, index) => {
    const index_prefix = `${SemanticConventions.EMBEDDING_EMBEDDINGS}.${index}`;
    acc[`${index_prefix}.${SemanticConventions.EMBEDDING_VECTOR}`] =
      embedding.embedding;
    return acc;
  }, {} as Attributes);
}
