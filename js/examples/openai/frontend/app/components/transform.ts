import { JSONValue, Message } from "ai";

export const isValidMessageData = (rawData: JSONValue | undefined) => {
  if (!rawData || typeof rawData !== "object") return false;
  if (Object.keys(rawData).length === 0) return false;
  return true;
};

/**
 * Messages streamed back from the server may contain annotations.
 * These provide additional context about the message, such as the span ID.
 * These annotations are a next.js construct and do not relate to span annotations.
 * Here we extract the span ID associated with the message from the message annotations.
 * This allows us to associate feedback with the correct message.
 * @param annotations
 * @returns
 */
const getSpanIdFromAnnotations = (annotations?: JSONValue[]) => {
  if (!annotations) return;
  for (const annotation of annotations) {
    if (
      annotation != null &&
      typeof annotation === "object" &&
      "spanId" in annotation &&
      typeof annotation.spanId === "string"
    ) {
      return annotation.spanId;
    }
  }
};

export const formatMessages = ({
  messages,
  messageIdToFeedbackMap,
}: {
  messages: Message[];
  messageIdToFeedbackMap: Record<string, number>;
}) =>
  messages.map((message) => {
    return {
      id: message.id,
      content: message.content,
      role: message.role,
      spanId: getSpanIdFromAnnotations(message.annotations),
      feedback: messageIdToFeedbackMap[message.id],
    };
  });
