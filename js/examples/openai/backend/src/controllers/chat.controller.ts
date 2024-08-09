import { Request, Response } from "express";
import { SpanStatusCode, trace } from "@opentelemetry/api";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { streamToResponse, StreamData, OpenAIStream } from "ai";
import { ChatCompletionMessageParam } from "openai/resources";
import OpenAI from "openai";

export const chat = async (req: Request, res: Response) => {
  const tracer = trace.getTracer("openai-service");
  // Add a custom span to trace the chat request
  tracer.startActiveSpan("chat", async (span) => {
    try {
      const { messages }: { messages: ChatCompletionMessageParam[] } = req.body;
      span.setAttributes({
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.CHAIN,
        [SemanticConventions.INPUT_VALUE]: JSON.stringify({ messages }),
        [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
      });

      if (!messages) {
        return res.status(400).json({
          error: "messages are required in the request body",
        });
      }

      const spanId = span.spanContext().spanId;

      res.setHeader("assistant-message-span-id", spanId);

      const openai = new OpenAI();
      const stream = await openai.chat.completions.create({
        messages,
        model: "gpt-3.5-turbo",
        stream: true,
      });

      const data = new StreamData();
      data.appendMessageAnnotation({
        spanId: spanId,
      });
      let streamedResponse = "";

      const streamThingy = OpenAIStream(stream, {
        onCompletion(completion) {
          streamedResponse += completion;
        },
        onFinal() {
          span.setAttributes({
            [SemanticConventions.OUTPUT_VALUE]: streamedResponse,
            [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
              streamedResponse,
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
              "assistant",
          });
          span.setStatus({ code: SpanStatusCode.OK });
          // End the span
          span.end();
          data.close();
        },
      });
      streamToResponse(streamThingy, res, {}, data);

      // Add OpenInference attributes to the span
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error("Error:", error);
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: (error as Error).message,
      });
      span.end();
      return res.status(500).json({
        error: (error as Error).message,
      });
    }
  });
};
