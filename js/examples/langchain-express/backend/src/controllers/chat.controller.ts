import { Request, Response } from "express";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";

type Message = {
  content: string;
  role: "user" | "assistant";
};

export const chat = async (req: Request, res: Response) => {
  try {
    const { messages }: { messages: Message[] } = req.body;

    if (!messages) {
      return res.status(400).json({
        error: "messages are required in the request body",
      });
    }

    const parser = new StringOutputParser();

    const chatModel = new ChatOpenAI({
      modelName: "gpt-3.5-turbo",
    });
    const stream = await chatModel
      .pipe(parser)
      .stream(messages.map((message) => message.content));

    for await (const chunk of stream) {
      res.write(chunk);
    }
    res.end();
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error("Error:", error);
    return res.status(500).json({
      error: (error as Error).message,
    });
  }
};
