import { Attributes, Span } from "@opentelemetry/api";
import {
  ResponseCreateParamsBase,
  ResponseStreamEvent,
  Response as ResponseType,
} from "openai/resources/responses/responses";
import { Stream } from "openai/streaming";

export function getResponsesInputAttributes(
  body: ResponseCreateParamsBase,
): Attributes {
  return {};
}

export function getResponsesUsageAttributes(
  response: ResponseType,
): Attributes {
  return {};
}

export function getResponsesOutputMessagesAttributes(
  response: ResponseType,
): Attributes {
  return {};
}

export async function consumeResponseStreamEvents(
  stream: Stream<ResponseStreamEvent>,
  span: Span,
) {
  for await (const event of stream) {
    console.log(event);
  }
}
