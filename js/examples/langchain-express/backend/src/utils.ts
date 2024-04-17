import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { Message } from "./types";

export const getMessageHistoryFromChat = (
  messages: Message[],
): BaseMessage[] => {
  return messages.slice(0, -1).map((message) => {
    switch (message.role) {
      case "user":
        return new HumanMessage(message.content);
      case "assistant":
        return new AIMessage(message.content);
      default:
        return new HumanMessage(message.content);
    }
  });
};

export const getUserQuestion = (messages: Message[]) =>
  messages[messages.length - 1].content;
