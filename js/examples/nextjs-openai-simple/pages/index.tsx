import { useState } from "react";

type Message = {
  message: string;
  time: string;
  role: "user" | "bot";
};
export default function Home() {
  const [question, setQuestion] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);

  const askQuestion = async (question: string) => {
    const userMessage: Message = {
      message: question,
      role: "user",
      time: new Date().toLocaleTimeString(),
    };
    setMessages([...messages, userMessage]);
    const res = await fetch("/api", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const { answer } = await res.json();
    setMessages([
      ...messages,
      userMessage,
      {
        message: answer,
        role: "bot",
        time: new Date().toLocaleTimeString(),
      },
    ]);
  };

  return (
    <main className="flex flex-col bg-black h-screen">
      <h2 className="text-white flex-none p-2">Ask GPT Questions</h2>
      <div className="flex flex-col flex-auto items-start gap-2.5 overflow-y-auto p-8 w-screen">
        {messages.map((msg, index) => (
          <ChatBubble key={index} {...msg} />
        ))}
      </div>
      <div className="flex-none flex flex-row gap-2.5 p-4 border-t-4 border-indigo-500 ">
        <textarea
          aria-label="question"
          id="message"
          rows={4}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          className="flex-auto block p-2.5 w-full text-sm text-gray-900 bg-gray-700 border-gray-600 placeholder-gray-400 text-white focus:ring-blue-500 focus:border-blue-500"
          placeholder="Write your question..."
        ></textarea>
        <button
          onClick={() => {
            askQuestion(question);
            setQuestion("");
          }}
          className="flex-none bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-20"
        >
          Ask
        </button>
      </div>
    </main>
  );
}

function ChatBubble({ message, time, role }: Message) {
  const isBot = role === "bot";
  return (
    <div
      className={`flex items-start gap-2.5 text-right ${
        isBot ? "flex-row" : "text-right self-end flex-row-reverse"
      }`}
    >
      <div
        className={`w-8 h-8 rounded-full flex-none ${
          isBot ? "bg-gray-400" : "bg-blue-400"
        }`}
      />
      <div
        className={`flex flex-col w-full max-w-[320px] p-4 border-gray-200 w-96 rounded-xl ${
          isBot
            ? "rounded-tl-none leading-1.5 bg-gray-600"
            : "rounded-tr-none trailing-1.5 text-right bg-blue-600"
        }`}
      >
        <div
          className={`flex items-center gap-2.5 ${
            isBot ? "" : "flex-row-reverse"
          }`}
        >
          <span
            className={`text-sm font-semibold text-gray-900 text-white ${
              isBot ? "" : "text-right"
            }`}
          >
            {role.toUpperCase()}
          </span>
          <span className="text-sm font-normal ext-gray-400">{time}</span>
        </div>
        <p className="text-sm font-normal py-2.5  text-white">{message}</p>
        <span className="text-sm font-normal  text-gray-400">Delivered</span>
      </div>
    </div>
  );
}
