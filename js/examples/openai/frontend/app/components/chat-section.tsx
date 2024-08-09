"use client";

import { useChat } from "ai/react";
import { useMemo, useState } from "react";
import { formatMessages } from "./transform";
import { ChatInput, ChatMessages } from "./ui/chat";

export default function ChatSection() {
  const {
    messages,
    input,
    isLoading,
    handleSubmit,
    handleInputChange,
    reload,
    stop,
  } = useChat({
    api: `${process.env.NEXT_PUBLIC_API_ENDPOINT}/api/chat`,
    headers: {
      "Content-Type": "application/json", // using JSON because of vercel/ai 2.2.26
    },
  });

  // Map of messageId to feedback score to preserve which messages have had feedback submitted and the score of that feedback
  const [messageIdToFeedbackMap, setMessageIdToFeedbackMap] = useState<
    Record<string, number>
  >({});

  const handleFeedback = (messageId: string, feedbackScore: number) => {
    setMessageIdToFeedbackMap((prev) => ({
      ...prev,
      [messageId]: feedbackScore,
    }));
  };

  const transformedMessages = useMemo(() => {
    return formatMessages({
      messages,
      messageIdToFeedbackMap,
    });
  }, [messages, messageIdToFeedbackMap]);

  return (
    <div className="space-y-4 max-w-5xl w-full">
      <ChatMessages
        messages={transformedMessages}
        handleFeedback={handleFeedback}
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
