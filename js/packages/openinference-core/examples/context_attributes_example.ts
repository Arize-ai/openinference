/**
 * Typesafe example backing the code snippets in docs/context-attributes.md
 *
 * Exercises: setSession, getSession, clearSession, setUser, getUser, clearUser,
 * setMetadata, getMetadata, clearMetadata, setTags, getTags, clearTags,
 * setPromptTemplate, getPromptTemplate, clearPromptTemplate, setAttributes,
 * getAttributes, clearAttributes, getAttributesFromContext
 */

import { context, trace } from "@opentelemetry/api";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { resourceFromAttributes } from "@opentelemetry/resources";
import {
  ConsoleSpanExporter,
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";

import {
  clearMetadata,
  clearSession,
  clearTags,
  clearUser,
  getAttributesFromContext,
  getMetadata,
  getSession,
  getTags,
  getUser,
  setAttributes,
  setMetadata,
  setPromptTemplate,
  setSession,
  setTags,
  setUser,
  withSpan,
} from "../src";

// -- Provider setup -----------------------------------------------------------

const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [SEMRESATTRS_PROJECT_NAME]: "context-attributes-example",
  }),
  spanProcessors: [new SimpleSpanProcessor(new ConsoleSpanExporter())],
});
provider.register();

// -- Traced helper functions --------------------------------------------------

const myTracedFunction = withSpan(
  async (input: string) => `processed: ${input}`,
  { name: "my-traced-function" },
);

// -- Basic setter / getter round-trip -----------------------------------------

function basicSetterGetterDemo() {
  // Set session
  let ctx = setSession(context.active(), { sessionId: "sess-42" });
  const session = getSession(ctx);
  console.log("session:", session); // { sessionId: "sess-42" }

  // Set user
  ctx = setUser(ctx, { userId: "user-7" });
  const user = getUser(ctx);
  console.log("user:", user); // { userId: "user-7" }

  // Set metadata
  ctx = setMetadata(ctx, {
    tenant: "acme",
    environment: "prod",
    requestId: "req-123",
  });
  const metadata = getMetadata(ctx);
  console.log("metadata:", metadata);

  // Set tags
  ctx = setTags(ctx, ["support", "priority-high", "v2"]);
  const tags = getTags(ctx);
  console.log("tags:", tags);

  // Clear and verify
  ctx = clearSession(ctx);
  console.log("session after clear:", getSession(ctx)); // undefined

  ctx = clearUser(ctx);
  console.log("user after clear:", getUser(ctx)); // undefined

  ctx = clearMetadata(ctx);
  console.log("metadata after clear:", getMetadata(ctx)); // undefined

  ctx = clearTags(ctx);
  console.log("tags after clear:", getTags(ctx)); // undefined
}

// -- Composing multiple context attributes (docs/context-attributes.md) -------

async function compositionDemo() {
  const enriched = setTags(
    setMetadata(
      setUser(
        setSession(context.active(), { sessionId: "sess-42" }),
        { userId: "user-7" },
      ),
      { tenant: "acme", environment: "prod" },
    ),
    ["support", "priority-high"],
  );

  await context.with(enriched, async () => {
    // All spans created here include session, user, metadata, and tags
    await myTracedFunction("composing multiple attributes");
  });
}

// -- Prompt template ----------------------------------------------------------

async function promptTemplateDemo() {
  const ctx = setPromptTemplate(context.active(), {
    template: "Answer the question about {topic} using the provided context.",
    variables: { topic: "billing" },
    version: "v3",
  });

  await context.with(ctx, async () => {
    await myTracedFunction("prompt template demo");
  });
}

// -- Generic attributes -------------------------------------------------------

async function genericAttributesDemo() {
  const ctx = setAttributes(context.active(), {
    "app.request_id": "req-123",
    "app.feature_flag": "new-model-enabled",
  });

  await context.with(ctx, async () => {
    await myTracedFunction("generic attributes demo");
  });
}

// -- getAttributesFromContext with raw OTel tracer -----------------------------

function getAttributesFromContextDemo() {
  const enriched = setSession(context.active(), { sessionId: "sess-42" });

  context.with(enriched, () => {
    const tracer = trace.getTracer("manual");
    const span = tracer.startSpan("manual-span");

    // Manually apply propagated attributes
    span.setAttributes(getAttributesFromContext(context.active()));

    span.end();
  });
}

// -- Scoping context to a single operation ------------------------------------

async function scopingDemo() {
  const myTracedRetriever = withSpan(
    async (query: string) => [`result for: ${query}`],
    { name: "retriever", kind: "RETRIEVER" },
  );

  // Only this specific call gets the metadata
  await context.with(
    setMetadata(context.active(), { experiment: "new-embeddings-v2" }),
    () => myTracedRetriever("search query"),
  );

  // This call does NOT have the metadata
  await myTracedRetriever("another query");
}

// -- Run all demos ------------------------------------------------------------

async function main() {
  basicSetterGetterDemo();
  await compositionDemo();
  await promptTemplateDemo();
  await genericAttributesDemo();
  getAttributesFromContextDemo();
  await scopingDemo();
}

main();
