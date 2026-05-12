/* eslint-disable no-console */
/**
 * Structured output example for OpenAI Agents SDK instrumentation
 *
 * This example demonstrates how to:
 * 1. Set up OpenInference instrumentation for OpenAI Agents
 * 2. Define structured output types using Zod schemas
 * 3. Get type-safe structured responses from agents
 * 4. Track structured outputs in traces
 *
 * Prerequisites:
 * - Set OPENAI_API_KEY environment variable
 * - Install dependencies: @openai/agents, @opentelemetry/sdk-trace-node, zod
 */

import { instrumentation, provider } from "./instrumentation";

// IMPORTANT: Import the SDK as a namespace so we can pass it to instrument()
import * as agentsSdk from "@openai/agents";
import { z } from "zod";

// Define a structured output schema for movie recommendations
const MovieRecommendation = z.object({
  title: z.string().describe("The movie title"),
  year: z.number().describe("The release year"),
  genre: z.string().describe("The primary genre"),
  director: z.string().describe("The director's name"),
  rating: z.number().min(1).max(10).describe("Rating out of 10"),
  summary: z.string().describe("A brief plot summary"),
  whyWatch: z.string().describe("Why the user should watch this movie"),
});

const MovieRecommendations = z.object({
  recommendations: z
    .array(MovieRecommendation)
    .length(3)
    .describe("Exactly 3 movie recommendations"),
  theme: z
    .string()
    .describe("The common theme connecting these recommendations"),
});

// Create an agent with structured output
const movieAgent = new agentsSdk.Agent({
  name: "MovieRecommender",
  instructions: `You are a movie recommendation expert. When asked for recommendations,
  provide exactly 3 carefully selected movies that match the user's criteria.
  Include detailed information about each movie and explain why it's a good choice.`,
  outputType: MovieRecommendations,
});

// Define a structured output for analysis
const SentimentAnalysis = z.object({
  sentiment: z
    .enum(["positive", "negative", "neutral"])
    .describe("Overall sentiment"),
  confidence: z
    .number()
    .min(0)
    .max(1)
    .describe("Confidence score between 0 and 1"),
  keyPhrases: z
    .array(z.string())
    .describe("Key phrases that influenced the analysis"),
  explanation: z.string().describe("Brief explanation of the analysis"),
});

// Create an agent for sentiment analysis
const sentimentAgent = new agentsSdk.Agent({
  name: "SentimentAnalyzer",
  instructions: `You are a sentiment analysis expert. Analyze the sentiment of the provided text
  and return a structured analysis including the overall sentiment, confidence score,
  key phrases, and explanation.`,
  outputType: SentimentAnalysis,
});

async function main() {
  // Instrument using the SDK module from our static import
  instrumentation.instrument(agentsSdk);

  console.log("Running structured output example...\n");

  try {
    // Example 1: Movie recommendations
    console.log("--- Example 1: Movie Recommendations ---");
    console.log('Request: "Recommend sci-fi movies with time travel"\n');

    const movieResult = await agentsSdk.run(
      movieAgent,
      "Recommend sci-fi movies with time travel",
    );

    // Access the structured output
    const movies = movieResult.finalOutput as z.infer<
      typeof MovieRecommendations
    >;
    console.log("Theme:", movies.theme);
    console.log("\nRecommendations:");
    movies.recommendations.forEach((movie, index) => {
      console.log(`\n${index + 1}. ${movie.title} (${movie.year})`);
      console.log(`   Genre: ${movie.genre}`);
      console.log(`   Director: ${movie.director}`);
      console.log(`   Rating: ${movie.rating}/10`);
      console.log(`   Summary: ${movie.summary}`);
      console.log(`   Why Watch: ${movie.whyWatch}`);
    });

    console.log("\n---\n");

    // Example 2: Sentiment analysis
    console.log("--- Example 2: Sentiment Analysis ---");
    const textToAnalyze =
      "I absolutely loved the new restaurant! The food was incredible, the service was top-notch, and the atmosphere was perfect. I can't wait to go back!";
    console.log(`Text: "${textToAnalyze}"\n`);

    const sentimentResult = await agentsSdk.run(
      sentimentAgent,
      `Analyze: "${textToAnalyze}"`,
    );

    // Access the structured output
    const analysis = sentimentResult.finalOutput as z.infer<
      typeof SentimentAnalysis
    >;
    console.log("Analysis Results:");
    console.log(`  Sentiment: ${analysis.sentiment}`);
    console.log(`  Confidence: ${(analysis.confidence * 100).toFixed(1)}%`);
    console.log(`  Key Phrases: ${analysis.keyPhrases.join(", ")}`);
    console.log(`  Explanation: ${analysis.explanation}`);
  } catch (error) {
    console.error("Error running agent:", error);
  }

  // Force flush spans to ensure they are exported
  await provider.forceFlush();

  // Give time for spans to be exported
  await new Promise((resolve) => setTimeout(resolve, 2000));

  // Shutdown provider
  await provider.shutdown();
}

main().catch(console.error);
