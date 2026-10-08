import { OITracer, setMetadata, setSession } from "@arizeai/openinference-core";
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";
import { OpenInferenceSpanKind, SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import { context, SpanStatusCode } from "@opentelemetry/api";
import { choice, TypeSafeClient } from "@typesafe-ai/sdk";
import OpenAI from "openai";

import { setupTracing } from "./instrumentation";

async function main() {
  const tracing = setupTracing("typesafe-guardrail-routing");
  const openaiInstrumentation = new OpenAIInstrumentation({ tracerProvider: tracing.provider });
  openaiInstrumentation.manuallyInstrument(OpenAI);
  const tracer = new OITracer({ tracer: tracing.provider.getTracer("typesafe-routing-example") });
  const state = "I was charged twice for my subscription. Can you help me request a refund?";
  const ctx = setMetadata(setSession(context.active(), { sessionId: "typesafe-routing-example" }), { workflow: "support-routing" });
  try {
    const typesafe = new TypeSafeClient();
    const openai = new OpenAI();
    await context.with(ctx, () => tracer.startActiveSpan("support.route", {
      attributes: { [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN },
    }, async (span) => {
      try {
        const decision = await typesafe.systemOne({
          state,
          questions: {
            route: choice("Choose whether a support assistant can safely draft a response.", {
              allow: "Ordinary customer support request",
              review: "Needs human review or requests harmful or inappropriate content",
            }),
          },
        });
        const { route } = decision.answers;
        if (route.choice === "allow" && route.confidence >= 0.8) {
          const response = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
              { role: "system", content: "Draft a brief, helpful support reply. Do not claim to have issued a refund." },
              { role: "user", content: state },
            ],
          });
          console.log(response.choices[0]?.message.content);
        } else {
          console.log("Routed to human review", route);
        }
        span.setStatus({ code: SpanStatusCode.OK });
      } catch (error) {
        span.recordException(error instanceof Error ? error : String(error));
        span.setStatus({ code: SpanStatusCode.ERROR });
        throw error;
      } finally {
        span.end();
      }
    }));
  } finally {
    openaiInstrumentation.disable();
    await tracing.shutdown();
  }
}

main().catch((error: unknown) => { console.error(error); process.exitCode = 1; });
