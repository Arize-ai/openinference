import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";
import { DOCUMENT_URLS, WEB_LOADER_TIMEOUT } from "../constants";

export const initializeVectorStore = async () => {
  const loader = new CheerioWebBaseLoader("https://docs.arize.com/phoenix");
  const webContent = await CheerioWebBaseLoader.scrapeAll(
    DOCUMENT_URLS,
    loader.caller,
    WEB_LOADER_TIMEOUT,
  );

  const documents = webContent.map((doc, i) => {
    const text = doc("body").text();
    return new Document({
      pageContent: text,
      metadata: { source: DOCUMENT_URLS[i] },
    });
  });
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkOverlap: 100,
    chunkSize: 500,
  });

  const splits = await textSplitter.splitDocuments(documents);
  return await MemoryVectorStore.fromDocuments(splits, new OpenAIEmbeddings());
};
