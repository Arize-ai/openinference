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
  Span,
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
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";
import { Stream } from "openai/streaming";
import {
  CreateEmbeddingResponse,
  EmbeddingCreateParams,
} from "openai/resources";
import { assertUnreachable } from "./typeUtils";

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
              kind: SpanKind.INTERNAL,
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
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
                span.end();
              }
            },
          );
          const wrappedPromise = execPromise.then((result) => {
            if (isChatCompletionResponse(result)) {
              // Record the results
              span.setAttributes({
                [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(result),
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
                // Override the model from the value sent by the server
                [SemanticConventions.LLM_MODEL_NAME]: result.model,
                ...getLLMOutputMessagesAttributes(result),
                ...getUsageAttributes(result),
              });
              span.setStatus({ code: SpanStatusCode.OK });
              span.end();
            } else {
              // This is a streaming response
              // handle the chunks and add them to the span
              // First split the stream via tee
              const [leftStream, rightStream] = result.tee();
              consumeChatCompletionStreamChunks(rightStream, span);
              result = leftStream;
            }

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
            kind: SpanKind.INTERNAL,
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
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
                span.end();
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
 * type-guard that checks if the response is a chat completion response
 */
function isChatCompletionResponse(
  response: Stream<ChatCompletionChunk> | ChatCompletion,
): response is ChatCompletion {
  return "choices" in response;
}

/**
 * Converts the body of the request to LLM input messages
 */
function getLLMInputMessagesAttributes(
  body: ChatCompletionCreateParamsBase,
): Attributes {
  return body.messages.reduce((acc, message, index) => {
    const messageAttributes = getChatCompletionInputMessageAttributes(message);
    const index_prefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}`;
    // Flatten the attributes on the index prefix
    for (const [key, value] of Object.entries(messageAttributes)) {
      acc[`${index_prefix}.${key}`] = value;
    }
    return acc;
  }, {} as Attributes);
}

function getChatCompletionInputMessageAttributes(
  message: ChatCompletionMessageParam,
): Attributes {
  const role = message.role;
  const attributes: Attributes = {
    [SemanticConventions.MESSAGE_ROLE]: role,
  };
  switch (role) {
    case "user":
      // Add the content only if it is a string
      if (typeof message.content == "string")
        attributes[SemanticConventions.MESSAGE_CONTENT] = message.content;
      break;
    case "assistant":
      if (message.tool_calls) {
        message.tool_calls.forEach((toolCall, index) => {
          const toolCallIndexPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${index}`;
          attributes[
            toolCallIndexPrefix + SemanticConventions.TOOL_CALL_FUNCTION_NAME
          ] = toolCall.function.name;
          attributes[
            toolCallIndexPrefix +
              SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
          ] = JSON.stringify(toolCall.function.arguments);
        });
      }
      break;
    case "function":
      // Add the content only if it is a string
      if (typeof message.content == "string")
        attributes[SemanticConventions.MESSAGE_CONTENT] = message.content;
      attributes[SemanticConventions.MESSAGE_FUNCTION_CALL_NAME] = message.name;
      break;
    case "tool":
      if (typeof message.content == "string")
        attributes[SemanticConventions.MESSAGE_CONTENT] = message.content;
      break;
    case "system":
      if (typeof message.content == "string")
        attributes[SemanticConventions.MESSAGE_CONTENT] = message.content;
      break;
    default:
      assertUnreachable(role);
      break;
  }
  return attributes;
}

/**
 * Get Usage attributes
 */
function getUsageAttributes(completion: ChatCompletion): Attributes {
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
  return {};
}

/**
 * Converts the result to LLM output attributes
 */
function getLLMOutputMessagesAttributes(
  completion: ChatCompletion,
): Attributes {
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

/**
 * Consumes the stream chunks and adds them to the span
 */
async function consumeChatCompletionStreamChunks(
  stream: Stream<ChatCompletionChunk>,
  span: Span,
) {
  let streamResponse = "";
  for await (const chunk of stream) {
    if (chunk.choices.length > 0 && chunk.choices[0].delta.content) {
      streamResponse += chunk.choices[0].delta.content;
    }
  }
  span.setAttributes({
    [SemanticConventions.OUTPUT_VALUE]: streamResponse,
    [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
    [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
      streamResponse,
    [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
      "assistant",
  });
  span.end();
}
