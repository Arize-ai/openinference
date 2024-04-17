import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import "cheerio";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

export const initializeVectorStore = async () => {
  const loader = new CheerioWebBaseLoader(
    "https://docs.arize.com/phoenix/programming-languages/javascript",
  );

  const documents = await loader.load();
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkOverlap: 200,
    chunkSize: 1000,
  });

  const splits = await textSplitter.splitDocuments(documents);
  return await MemoryVectorStore.fromDocuments(splits, new OpenAIEmbeddings());
};
