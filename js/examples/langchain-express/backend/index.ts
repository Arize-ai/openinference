/* eslint-disable no-console */
import "./instrumentation";
import cors from "cors";
import "dotenv/config";
import express, { Express, Request, Response } from "express";
import { createChatRouter } from "./src/routes/chat.route";
import { initializeVectorStore } from "./src/vector_store/store";

const app: Express = express();
const port = parseInt(process.env.PORT || "8000");

const env = process.env["NODE_ENV"];
const isDevelopment = !env || env === "development";
const prodCorsOrigin = process.env["PROD_CORS_ORIGIN"];

app.use(express.json());

if (isDevelopment) {
  console.warn("Running in development mode - allowing CORS for all origins");
  app.use(cors());
} else if (prodCorsOrigin) {
  console.log(
    `Running in production mode - allowing CORS for domain: ${prodCorsOrigin}`,
  );
  const corsOptions = {
    origin: prodCorsOrigin, // Restrict to production domain
  };
  app.use(cors(corsOptions));
} else {
  console.warn("Production CORS origin not set, defaulting to no CORS.");
}

app.get("/", (req: Request, res: Response) => {
  res.send("Arize Express Server");
});

initializeVectorStore()
  .then((vectorStore) => {
    app.use("/api/chat", createChatRouter(vectorStore));
    app.listen(port, () => {
      console.log(`⚡️[server]: Server is running at http://localhost:${port}`);
    });
  })
  .catch((error) => {
    console.error("Error initializing store:", error);
  });
