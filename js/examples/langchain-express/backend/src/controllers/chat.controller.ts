import { Request, Response } from "express";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { SYSTEM_PROMPT_TEMPLATE } from "../constants";
import { getMessageHistoryFromChat, getUserQuestion } from "../utils";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { Message } from "../types";

export const createChatController =
  (vectorStore: MemoryVectorStore) => async (req: Request, res: Response) => {
    try {
      const { messages }: { messages: Message[] } = req.body;

      if (!messages) {
        return res.status(400).json({
          error: "messages are required in the request body",
        });
      }

      const llm = new ChatOpenAI({
        modelName: "gpt-3.5-turbo",
        streaming: true,
      });

      const qaPrompt = ChatPromptTemplate.fromMessages([
        ["system", SYSTEM_PROMPT_TEMPLATE],
        ...getMessageHistoryFromChat(messages),
        ["human", "{input}"],
      ]);

      const retriever = vectorStore.asRetriever();

      const combineDocsChain = await createStuffDocumentsChain({
        llm,
        prompt: qaPrompt,
      });
      const ragChain = await createRetrievalChain({
        combineDocsChain,
        retriever,
      });

      const response = await ragChain.invoke({
        input: getUserQuestion(messages),
        llm,
      });

      if (response.answer == null) {
        throw new Error("No response from the model");
      }
      res.send(response.answer);
      res.end();
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error("Error:", error);
      return res.status(500).json({
        error: (error as Error).message,
      });
    }
  };
