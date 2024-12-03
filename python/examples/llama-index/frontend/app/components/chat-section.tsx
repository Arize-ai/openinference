"use client";

import { useChat } from "ai/react";
import { useMemo } from "react";
import { insertDataIntoMessages } from "./transform";
import { ChatInput, ChatMessages } from "./ui/chat";

export default function ChatSection() {
  // Get or create session and user IDs
  const sessionId = useMemo(() => {
    // Check if we're in a browser environment
    if (typeof window === 'undefined') return "";
    
    const stored = sessionStorage.getItem("sessionId");
    if (!stored) {
      const newId = crypto.randomUUID();
      sessionStorage.setItem("sessionId", newId);
      return newId;
    }
    return stored;
  }, []);

  const userId = useMemo(() => {
    // Check if we're in a browser environment
    if (typeof window === 'undefined') return "";
    
    const stored = sessionStorage.getItem("userId");
    if (!stored) {
      const newId = crypto.randomUUID();
      sessionStorage.setItem("userId", newId);
      return newId;
    }
    return stored;
  }, []);
  console.log({sessionId, userId});

  console.log(`process.env.NEXT_PUBLIC_CHAT_API: ${process.env.NEXT_PUBLIC_CHAT_API}`);
  const {
    messages,
    input,
    isLoading,
    handleSubmit,
    handleInputChange,
    reload,
    stop,
    data,
  } = useChat({
    api: process.env.NEXT_PUBLIC_CHAT_API,
    headers: {
      "X-Session-Id": sessionId,
      "X-User-Id": userId,
    },
  });

  const transformedMessages = useMemo(() => {
    return insertDataIntoMessages(messages, data);
  }, [messages, data]);

  return (
    <div className="space-y-4 max-w-5xl w-full">
      <ChatMessages
        messages={transformedMessages}
        isLoading={isLoading}
        reload={reload}
        stop={stop}
      />
      <ChatInput
        input={input}
        handleSubmit={handleSubmit}
        handleInputChange={handleInputChange}
        isLoading={isLoading}
        multiModal={process.env.NEXT_PUBLIC_MODEL === "gpt-4-vision-preview"}
      />
    </div>
  );
}
