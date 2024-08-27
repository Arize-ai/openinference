import "./instrumentation";
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const testDocuments = [
  "dogs are cute",
  "rainbows are colorful",
  "water is wet",
];

const PROMPT_TEMPLATE = `Use the context below to answer the question.
----------------
{context}

Question:
{input}
`;
const prompt = ChatPromptTemplate.fromTemplate(PROMPT_TEMPLATE);

const main = async () => {
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const response = await chatModel.invoke("Hello! How are you?");

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
  });
  const docs = await textSplitter.createDocuments(testDocuments);
  const vectorStore = await MemoryVectorStore.fromDocuments(
    docs,
    new OpenAIEmbeddings({}),
  );
  const combineDocsChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
  });
  const chain = await createRetrievalChain({
    combineDocsChain: combineDocsChain,
    retriever: vectorStore.asRetriever(),
  });

  await chain.invoke({
    input: "What are cats?",
  });

  // eslint-disable-next-line no-console
  console.log(response.content);

  return response;
};

main();
