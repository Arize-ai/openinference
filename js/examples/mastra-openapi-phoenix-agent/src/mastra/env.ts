import { z } from "zod";

export const envSchema = z.object({
  OPENAI_API_KEY: z.string().describe("OpenAI API key"),
  PHOENIX_API_KEY: z.string().optional().describe("Phoenix API key"),
  PHOENIX_API_URL: z.string().describe("Phoenix API URL"),
  GITHUB_PERSONAL_ACCESS_TOKEN: z
    .string()
    .optional()
    .describe("GitHub personal access token"),
});

export const env = envSchema.parse(process.env);
