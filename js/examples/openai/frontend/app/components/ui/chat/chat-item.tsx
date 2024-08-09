"use client";

import { ButtonHTMLAttributes, useCallback, useEffect, useState } from "react";
import ChatAvatar from "./chat-avatar";
import { Message } from "./chat-messages";

function FeedbackButton({
  onClick,
  children,
  ...buttonProps
}: {
  onClick: () => void;
  children: React.ReactNode;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={
        "ease-in-out duration-300 hover:bg-gray-200 px-2 rounded active:bg-gray-400"
      }
      onClick={onClick}
      {...buttonProps}
    >
      {children}
    </button>
  );
}

const FeedbackNotification = ({
  message,
  variant,
  onClose,
}: {
  message: string;
  onClose: () => void;
  variant: "error" | "success";
}) => {
  const color = variant === "error" ? "bg-red-500" : "bg-green-500";
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 2000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div
      className={`fixed bottom-4 right-4 ${color} text-white px-8 py-4 rounded shadow-lg`}
    >
      {message}
    </div>
  );
};

export default function ChatItem(message: Message) {
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false);
  const [notification, setNotification] = useState<{
    message: string;
    variant: "error" | "success";
  } | null>(null);
  const onFeedbackClick = useCallback(
    async (feedbackScore: number) => {
      if (isSubmittingFeedback) return;
      setIsSubmittingFeedback(true);
      if (message.spanId) {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_ENDPOINT}/api/feedback`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              spanId: message.spanId,
              feedbackScore,
            }),
          },
        );
        if (response.status !== 200) {
          setNotification({
            message: "Failed to send feedback",
            variant: "error",
          });
        } else {
          setNotification({
            message: "Feedback sent successfully",
            variant: "success",
          });
        }
      }
      setIsSubmittingFeedback(false);
    },
    [isSubmittingFeedback, message.spanId],
  );

  return (
    <div className="flex flex-col gap-2 pr-8">
      <div className="flex items-start gap-4 pt-5">
        <ChatAvatar {...message} />
        <p className="break-words">{message.content}</p>
      </div>
      {message.role === "assistant" && message.spanId && (
        <div className="flex gap-2 self-end">
          <FeedbackButton
            onClick={() => onFeedbackClick(1)}
            disabled={isSubmittingFeedback}
          >
            👍
          </FeedbackButton>
          <FeedbackButton
            onClick={() => onFeedbackClick(0)}
            disabled={isSubmittingFeedback}
          >
            👎
          </FeedbackButton>
        </div>
      )}
      {notification && (
        <FeedbackNotification
          {...notification}
          onClose={() => setNotification(null)}
        />
      )}
    </div>
  );
}
