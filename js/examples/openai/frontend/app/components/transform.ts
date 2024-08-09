import { JSONValue, Message } from "ai";

export const isValidMessageData = (rawData: JSONValue | undefined) => {
  if (!rawData || typeof rawData !== "object") return false;
  if (Object.keys(rawData).length === 0) return false;
  return true;
};

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

export const formatMessages = (messages: Message[]) =>
  messages.map((message) => {
    return {
      id: message.id,
      content: message.content,
      role: message.role,
      spanId: getSpanIdFromAnnotations(message.annotations),
    };
  });

export const insertDataIntoMessages = (
  messages: Message[],
  data: JSONValue[] | undefined,
) => {
  if (!data) return messages;
  messages.forEach((message, i) => {
    const rawData = data[i];
    if (isValidMessageData(rawData)) message.data = rawData;
  });
  return messages;
};
