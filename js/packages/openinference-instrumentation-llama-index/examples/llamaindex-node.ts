import "./instrumentation";
import fs from "fs/promises";
import { VectorStoreIndex } from "llamaindex";
import { Document } from "@llamaindex/core/dist/schema";

async function main() {
  // Load essay from abramov.txt in Node
  const essay = await fs.readFile(
    "node_modules/llamaindex/examples/abramov.txt",
    "utf-8",
  );

  // Create Document object with essay
  const document = new Document({ text: essay });

  // Split text and create embeddings. Store them in a VectorStoreIndex
  const index = await VectorStoreIndex.fromDocuments([document]);

  // Query the index
  const queryEngine = index.asQueryEngine();
  const response = await queryEngine.query({
    query: "What did the author do in college?",
  });

  // Output response
  // eslint-disable-next-line no-console
  console.log(response.toString());
}

main();
