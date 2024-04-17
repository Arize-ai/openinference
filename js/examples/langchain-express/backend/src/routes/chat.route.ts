import express from "express";
import { createChatController } from "../controllers/chat.controller";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

export const createChatRouter = (vectorStore: MemoryVectorStore) => {
  const llmRouter = express.Router();

  llmRouter.route("/").post(createChatController(vectorStore));
  return llmRouter;
};
