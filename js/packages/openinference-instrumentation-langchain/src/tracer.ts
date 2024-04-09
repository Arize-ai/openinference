import { BaseTracer, Run } from "@langchain/core/tracers/base";
import {
  Tracer,
  SpanKind,
  Span,
  context,
  trace,
  SpanStatusCode,
  Attributes,
} from "@opentelemetry/api";
import { isAttributeValue, isTracingSuppressed } from "@opentelemetry/core";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import {
  assertUnreachable,
  isNonEmptyArray,
  isObject,
  isString,
} from "./typeUtils";
import {
  LLMMessage,
  LLMMessageFunctionCall,
  LLMMessageToolCalls,
  LLMMessagesAttributes,
} from "./types";

type RunWithSpan = {
  run: Run;
  span: Span;
};

export class LangChainTracer extends BaseTracer {
  private _tracer: Tracer;
  private _runs: Record<string, RunWithSpan | undefined> = {};
  constructor(tracer: Tracer) {
    super();
    this._tracer = tracer;
  }
  name: string = "OpenInferenceLangChainTracer";
  protected persistRun(_run: Run): Promise<void> {
    return Promise.resolve();
  }

  protected async _startTrace(run: Run) {
    await super._startTrace(run);
    if (isTracingSuppressed(context.active())) {
      return;
    }

    let activeContext = context.active();
    const parentCtx = this._getParentSpanContext(run);
    if (parentCtx != null) {
      activeContext = trace.setSpanContext(context.active(), parentCtx);
    }

    const span = this._tracer.startSpan(
      run.name,
      {
        kind: SpanKind.INTERNAL,
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            this._getOpenInferenceSpanKindFromRunType(run.run_type),
        },
      },
      activeContext,
    );

    this._runs[run.id] = { run, span };
  }
  protected async _endTrace(run: Run) {
    await super._endTrace(run);
    if (isTracingSuppressed(context.active())) {
      return;
    }
    const runWithSpan = this._runs[run.id];
    if (!runWithSpan) {
      return;
    }
    const { span } = runWithSpan;
    if (run.error != null) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: run.error,
      });
    } else {
      span.setStatus({ code: SpanStatusCode.OK });
    }

    span.setAttributes(
      this._flattenAttributes({
        ...this._formatIO({ io: run.inputs, type: "input" }),
        ...this._formatIO({ io: run.outputs, type: "output" }),
        ...this._formatPrompts(run.inputs),
        ...this._formatInputMessages(run.inputs),
        ...this._formatOutputMessages(run.outputs),
      }),
    );

    runWithSpan.span.end();
  }

  private _getParentSpanContext(run: Run) {
    if (run.parent_run_id == null) {
      return;
    }
    const maybeParent = this._runs[run.parent_run_id];
    if (maybeParent == null) {
      return;
    }

    return maybeParent.span.spanContext();
  }

  /**
   * Flattens nested attributes into a single level object
   */
  private _flattenAttributes(
    attributes: Record<string, unknown>,
    baseKey: string = "",
  ): Attributes {
    const result: Attributes = {};
    for (const key in attributes) {
      const newKey = baseKey ? `${baseKey}.${key}` : key;
      const value = attributes[key];

      if (value == null) {
        continue;
      }

      if (isObject(value)) {
        Object.assign(result, this._flattenAttributes(value, newKey));
      } else if (Array.isArray(value)) {
        value.forEach((item, index) => {
          if (isObject(item)) {
            Object.assign(
              result,
              this._flattenAttributes(item, `${newKey}.${index}`),
            );
          } else {
            result[`${newKey}.${index}`] = item;
          }
        });
      } else if (isAttributeValue(value)) {
        result[newKey] = value;
      }
    }
    return result;
  }

  private _getOpenInferenceSpanKindFromRunType(runType: string) {
    const normalizedRunType = runType.toUpperCase();
    if (normalizedRunType.includes("AGENT")) {
      return OpenInferenceSpanKind.AGENT;
    }

    if (normalizedRunType in OpenInferenceSpanKind) {
      return OpenInferenceSpanKind[
        normalizedRunType as keyof typeof OpenInferenceSpanKind
      ];
    }
    return "UNKNOWN";
  }

  private _formatIO({
    io,
    type,
  }: {
    io: Run["inputs"] | Run["outputs"];
    type: "input" | "output";
  }) {
    let valueAttribute: string;
    let mimeTypeAttribute: string;
    switch (type) {
      case "input": {
        valueAttribute = SemanticConventions.INPUT_VALUE;
        mimeTypeAttribute = SemanticConventions.INPUT_MIME_TYPE;
        break;
      }
      case "output": {
        valueAttribute = SemanticConventions.OUTPUT_VALUE;
        mimeTypeAttribute = SemanticConventions.OUTPUT_MIME_TYPE;
        break;
      }
      default:
        assertUnreachable(type);
    }
    if (io == null) {
      return {};
    }
    const values = Object.values(io);
    if (values.length === 1 && typeof values[0] === "string") {
      return {
        [valueAttribute]: values[0],
        [mimeTypeAttribute]: MimeType.TEXT,
      };
    }

    return {
      [valueAttribute]: JSON.stringify(io),
      [mimeTypeAttribute]: MimeType.JSON,
    };
  }

  private _formatPrompts(input: Run["inputs"]) {
    const maybePrompts = input.prompts;
    if (maybePrompts == null) {
      return {};
    }
    return {
      [SemanticConventions.LLM_PROMPTS]: maybePrompts,
    };
  }

  private _getRoleFromMessageData(
    messageData: Record<string, unknown>,
  ): string | undefined {
    const messageIds = messageData.lc_id;
    if (!isNonEmptyArray(messageIds)) {
      return;
    }
    const langchainMessageClass = messageIds[messageIds.length - 1];
    const normalizedLangchainMessageClass = isString(langchainMessageClass)
      ? langchainMessageClass.toLowerCase()
      : "";
    if (normalizedLangchainMessageClass.includes("human")) {
      return "user";
    }
    if (normalizedLangchainMessageClass.includes("ai")) {
      return "assistant";
    }
    if (normalizedLangchainMessageClass.includes("system")) {
      return "system";
    }
    if (normalizedLangchainMessageClass.includes("function")) {
      return "function";
    }
    if (
      normalizedLangchainMessageClass.includes("chat") &&
      isObject(messageData.kwargs) &&
      isString(messageData.kwargs.role)
    ) {
      return messageData.kwargs.role;
    }
  }

  private _getContentFromMessageData(
    messageKwargs: Record<string, unknown>,
  ): string | undefined {
    return isString(messageKwargs.content) ? messageKwargs.content : undefined;
  }
  private _getFunctionCallDataFromAdditionalKwargs(
    additionalKwargs: Record<string, unknown>,
  ): LLMMessageFunctionCall {
    const functionCall = additionalKwargs.function_call;
    if (!isObject(functionCall)) {
      return {};
    }
    const functionCallName = isString(functionCall.name)
      ? functionCall.name
      : undefined;
    const functionCallArgs = isString(functionCall.args)
      ? functionCall.args
      : undefined;
    return {
      [SemanticConventions.MESSAGE_FUNCTION_CALL_NAME]: functionCallName,
      [SemanticConventions.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON]:
        functionCallArgs,
    };
  }

  private _getToolCallDataFromAdditionalKwargs(
    additionalKwargs: Record<string, unknown>,
  ): LLMMessageToolCalls {
    const toolCalls = additionalKwargs.tool_calls;
    if (!Array.isArray(toolCalls)) {
      return {};
    }
    const formattedToolCalls = toolCalls.map((toolCall) => {
      if (!isObject(toolCall) && !isObject(toolCall.function)) {
        return {};
      }
      const toolCallName = isString(toolCall.function.name)
        ? toolCall.function.name
        : undefined;
      const toolCallArgs = isString(toolCall.function.arguments)
        ? toolCall.function.arguments
        : undefined;
      return {
        [SemanticConventions.TOOL_CALL_FUNCTION_NAME]: toolCallName,
        [SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON]: toolCallArgs,
      };
    });
    return {
      [SemanticConventions.MESSAGE_TOOL_CALLS]: formattedToolCalls,
    };
  }

  private _parseMessage(messageData: Record<string, unknown>): LLMMessage {
    const message: LLMMessage = {};

    message[SemanticConventions.MESSAGE_ROLE] =
      this._getRoleFromMessageData(messageData);

    const messageKwargs = messageData.lc_kwargs;
    if (!isObject(messageKwargs)) {
      return message;
    }
    message[SemanticConventions.MESSAGE_CONTENT] =
      this._getContentFromMessageData(messageKwargs);

    const additionalKwargs = messageKwargs.additional_kwargs;
    if (!isObject(additionalKwargs)) {
      return message;
    }
    return {
      ...message,
      ...this._getFunctionCallDataFromAdditionalKwargs(additionalKwargs),
      ...this._getToolCallDataFromAdditionalKwargs(additionalKwargs),
    };
  }

  private _formatInputMessages(
    input: Run["inputs"],
  ): LLMMessagesAttributes | null {
    const maybeMessages = input.messages;
    if (!isNonEmptyArray(maybeMessages)) {
      return null;
    }

    // Only support the first 'set' of messages
    const firstMessages = maybeMessages[0];
    if (!Array.isArray(firstMessages)) {
      return null;
    }

    const parsedMessages: LLMMessage[] = [];
    firstMessages.forEach((messageData) => {
      if (!isObject(messageData)) {
        return;
      }
      parsedMessages.push(this._parseMessage(messageData));
    });

    if (parsedMessages.length > 0) {
      return { [SemanticConventions.LLM_INPUT_MESSAGES]: parsedMessages };
    }

    return null;
  }

  private _formatOutputMessages(
    output: Run["outputs"],
  ): LLMMessagesAttributes | null {
    if (output == null) {
      return null;
    }
    const maybeGenerations = output.generations;

    if (!isNonEmptyArray(maybeGenerations)) {
      return null;
    }
    // Only support the first 'set' of generations
    const firstGenerations = maybeGenerations[0];
    if (!Array.isArray(firstGenerations)) {
      return null;
    }

    const parsedMessages: LLMMessage[] = [];
    firstGenerations.forEach((generation) => {
      if (!isObject(generation) && !isObject(generation.message)) {
        return;
      }
      parsedMessages.push(this._parseMessage(generation.message));
    });

    if (parsedMessages.length > 0) {
      return { [SemanticConventions.LLM_OUTPUT_MESSAGES]: parsedMessages };
    }

    return null;
  }
}
