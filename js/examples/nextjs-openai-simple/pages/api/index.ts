import { OpenAI } from "openai";
import { NextApiRequest, NextApiResponse } from "next";
import {
  SemanticConventions,
  OpenInferenceSpanKind,
  INPUT_VALUE,
  OUTPUT_VALUE,
} from "@arizeai/openinference-semantic-conventions";

const openai = new OpenAI();
import opentelemetry from "@opentelemetry/api";

const tracer = opentelemetry.trace.getTracer("chat");

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const question = req.body.question;

  return tracer.startActiveSpan("chat", async (span) => {
    span.setAttributes({
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
        OpenInferenceSpanKind.CHAIN,
      [INPUT_VALUE]: question,
    });
    const chatCompletion = await openai.chat.completions.create({
      messages: [{ role: "user", content: question }],
      model: "gpt-3.5-turbo",
    });
    const answer: string = chatCompletion.choices[0].message.content || "";
    span.setAttributes({
      [OUTPUT_VALUE]: answer,
    });
    // Be sure to end the span!
    span.end();
    return res.json({ answer });
  });
}
