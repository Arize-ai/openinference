/**
 * The trace hero's data and its state at any moment. One support-agent run,
 * recorded as an OpenInference trace, is replayed on a single clock: every
 * part of the hero (the conversation, the code, the waterfall, the span
 * detail) is a pure function of the trace time `t`, in milliseconds.
 *
 * The run is real-shaped: the code in `CODE` is the agent, instrumented with
 * the OpenAI instrumentor, the requests instrumentor and the `@tracer`
 * decorators, and `SPANS` is the tree it emits (names, kinds, nesting and
 * attribute keys as the instrumentors write them). Only the timings and
 * values are made up.
 *
 * Shared by the server render (the finished trace, for readers without
 * JavaScript) and the client, which replays it.
 */

export type Kind = "AGENT" | "LLM" | "TOOL" | "RETRIEVER" | "EMBEDDING" | "RERANKER" | "GUARDRAIL" | "HTTP";

/** One attribute, as the span records it. */
export interface Attribute {
  key: string;
  value: string;
  /** Recorded when the span ends (outputs, token counts, status codes), so not shown before. */
  end?: boolean;
  /** Marks the value as the failure. */
  error?: boolean;
}

export interface Span {
  name: string;
  kind: Kind;
  /** 0 for the root. A span's parent is the nearest earlier span with a smaller depth. */
  depth: number;
  start: number;
  end: number;
  /** Spans that one chat step (or the answer) stands for. */
  group: string;
  /** The code line running while this is the deepest open span. */
  line: number;
  /** The line that called it, lit while it runs. */
  call?: number;
  /** An HTTP span's status code, shown after its name once it ends. */
  note?: string;
  error?: boolean;
  attributes: Attribute[];
}

/** The run's length: the root span's duration. */
export const T = 8200;
/** The replay starts a little before the trace, with only the question showing. */
export const PRE_ROLL = -600;

export const TRACE_ID = "4bf92e71c0d38a16e5b0c1a9f27d4e83";

export const QUESTION = "I was charged twice for March. Can you fix it?";
export const ANSWER =
  "You were charged twice on March 3. I refunded the duplicate $49.00 payment on invoice INV‑2291; it should reach your card in 3–5 business days.";

const MODEL = "gpt-5-2025-08-07";
const GENERATED_CODE = 'inv = billing.get_invoice(month="2026-03")\nbilling.refund_payment(inv.payments[1])';

const kind = (value: Kind): Attribute => ({ key: "openinference.span.kind", value });
const llm = (prompt: number, completion: number, output: string): Attribute[] => [
  kind("LLM"),
  { key: "llm.model_name", value: MODEL },
  { key: "llm.token_count.prompt", value: String(prompt), end: true },
  { key: "llm.token_count.completion", value: String(completion), end: true },
  { key: "output.value", value: output, end: true },
];
const tool = (name: string, input: string, output: string): Attribute[] => [
  kind("TOOL"),
  { key: "tool.name", value: name },
  { key: "input.value", value: input },
  { key: "output.value", value: output, end: true },
];
const http = (method: string, status: number, url: string): Attribute[] => [
  { key: "http.method", value: method },
  { key: "http.status_code", value: String(status), end: true, error: status >= 500 },
  { key: "http.url", value: url },
];

/** The agent, as the Code view shows it. Spans find their lines in it by content (`lineOf`). */
export const CODE = `from openai import OpenAI
from openinference.instrumentation import TracerProvider
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))  # any OTLP backend
OpenAIInstrumentor().instrument(tracer_provider=provider)
RequestsInstrumentor().instrument(tracer_provider=provider)
tracer = provider.get_tracer(__name__)
client = OpenAI()

@tracer.tool
def load_skill(name: str) -> str:
    return SKILLS[name].read_text()

@tracer.tool
def search_help_center(query: str) -> list[str]:
    emb = client.embeddings.create(
        model=EMBED, input=query)
    hits = vector_search(emb)  # @tracer.retriever
    return rerank(query, hits)  # @tracer.reranker

@tracer.tool
def execute_code(code: str) -> str:
    return sandbox.run(code, billing)

@tracer.agent
def support_agent(question: str) -> str:
    input_guard(question)  # @tracer.guardrail
    messages = [user_message(question)]
    while True:
        response = client.responses.create(
            model=MODEL, tools=TOOLS,
            input=messages)
        if not (calls := tool_calls(response)):
            return output_guard(response)
        messages += response.output
        messages += run_tools(calls)`;

/** The 1-based line of the code that contains `text`. */
const lineOf = (text: string) => {
  const index = CODE.split("\n").findIndex((line) => line.includes(text));
  if (index < 0) throw new Error(`No code line contains ${text}`);
  return index + 1;
};

export const SPANS: Span[] = [
  {
    name: "support_agent", kind: "AGENT", depth: 0, start: 0, end: T, group: "root", line: lineOf("def support_agent"),
    attributes: [kind("AGENT"), { key: "input.value", value: QUESTION }, { key: "output.value", value: ANSWER, end: true }],
  },
  {
    name: "input_guard", kind: "GUARDRAIL", depth: 1, start: 30, end: 210, group: "guard", line: lineOf("input_guard(question)"),
    attributes: [kind("GUARDRAIL"), { key: "input.value", value: QUESTION }, { key: "output.value", value: "pass", end: true }],
  },
  {
    name: "Response", kind: "LLM", depth: 1, start: 250, end: 1180, group: "plan", line: lineOf("client.responses.create("),
    attributes: llm(1480, 64, 'load_skill(name="billing-disputes")  search_help_center(query="charged twice for March")'),
  },
  {
    name: "load_skill", kind: "TOOL", depth: 1, start: 1220, end: 1340, group: "skill", line: lineOf("read_text()"), call: lineOf("run_tools"),
    attributes: tool("load_skill", '{"name": "billing-disputes"}', "# Billing disputes\nRefund a duplicate charge once you have found both payments…"),
  },
  {
    name: "search_help_center", kind: "TOOL", depth: 1, start: 1220, end: 2480, group: "search", line: lineOf("def search_help_center"), call: lineOf("run_tools"),
    attributes: tool("search_help_center", '{"query": "charged twice for March"}', '["Why was I charged twice?", "Refund timelines", "Duplicate payments"]'),
  },
  {
    name: "CreateEmbeddings", kind: "EMBEDDING", depth: 2, start: 1250, end: 1490, group: "search", line: lineOf("client.embeddings.create("),
    attributes: [
      kind("EMBEDDING"),
      { key: "embedding.model_name", value: "text-embedding-3-small" },
      { key: "embedding.embeddings.0.embedding.text", value: "charged twice for March" },
      { key: "llm.token_count.prompt", value: "6", end: true },
    ],
  },
  {
    name: "vector_search", kind: "RETRIEVER", depth: 2, start: 1520, end: 1890, group: "search", line: lineOf("vector_search(emb)"),
    attributes: [
      kind("RETRIEVER"),
      { key: "input.value", value: '{"embedding": [0.0132, -0.0207, 0.0041, …], "k": 20}' },
      { key: "output.value", value: '[{"id": "kb-118", "score": 0.82}, {"id": "kb-204", "score": 0.79}, …]', end: true },
    ],
  },
  {
    name: "rerank", kind: "RERANKER", depth: 2, start: 1920, end: 2440, group: "search", line: lineOf("return rerank"),
    attributes: [
      kind("RERANKER"),
      { key: "input.value", value: '{"query": "charged twice for March", "hits": [20 items], "top_n": 3}' },
      { key: "output.value", value: '["kb-204", "kb-118", "kb-311"]', end: true },
    ],
  },
  {
    name: "Response", kind: "LLM", depth: 1, start: 2530, end: 3980, group: "code", line: lineOf("client.responses.create("),
    attributes: llm(4210, 236, `execute_code(code=${JSON.stringify(GENERATED_CODE)})`),
  },
  {
    name: "execute_code", kind: "TOOL", depth: 1, start: 4020, end: 5960, group: "code", line: lineOf("sandbox.run"), call: lineOf("run_tools"),
    attributes: tool("execute_code", GENERATED_CODE, '{"refund": "re_3Pq9", "amount": 49.00}'),
  },
  {
    name: "get_invoice", kind: "TOOL", depth: 2, start: 4100, end: 4980, group: "code", line: lineOf("sandbox.run"),
    attributes: tool("get_invoice", '{"month": "2026-03"}', '{"id": "INV-2291", "payments": [49.00, 49.00]}'),
  },
  {
    name: "GET", kind: "HTTP", depth: 3, start: 4130, end: 4420, group: "code", line: lineOf("sandbox.run"), note: "500", error: true,
    attributes: http("GET", 500, "https://billing.internal/v1/invoices?month=2026-03"),
  },
  {
    name: "GET", kind: "HTTP", depth: 3, start: 4700, end: 4950, group: "code", line: lineOf("sandbox.run"), note: "200",
    attributes: http("GET", 200, "https://billing.internal/v1/invoices?month=2026-03"),
  },
  {
    name: "refund_payment", kind: "TOOL", depth: 2, start: 5020, end: 5900, group: "code", line: lineOf("sandbox.run"),
    attributes: tool("refund_payment", '{"payment": "pay_8Qx2"}', '{"refund": "re_3Pq9", "status": "succeeded"}'),
  },
  {
    name: "POST", kind: "HTTP", depth: 3, start: 5050, end: 5870, group: "code", line: lineOf("sandbox.run"), note: "200",
    attributes: http("POST", 200, "https://billing.internal/v1/refunds"),
  },
  {
    name: "Response", kind: "LLM", depth: 1, start: 6000, end: 7880, group: "answer", line: lineOf("client.responses.create("),
    attributes: llm(4630, 118, ANSWER),
  },
  {
    name: "output_guard", kind: "GUARDRAIL", depth: 1, start: 7910, end: 8120, group: "answer", line: lineOf("output_guard("),
    attributes: [kind("GUARDRAIL"), { key: "input.value", value: ANSWER }, { key: "output.value", value: "pass", end: true }],
  },
];

/**
 * The answer as a model streams it: token by token (words, numbers and
 * punctuation, roughly as a tokenizer splits them), at an uneven cadence.
 * Each token's arrival is a fraction of the span, fixed so every replay is
 * the same.
 */
export const ANSWER_TOKENS = ANSWER.match(/\s*[A-Za-z]+|\s*\d+(?:[.,]\d+)*|\s*[^\sA-Za-z\d]/g) ?? [ANSWER];
const TOKEN_ARRIVALS = (() => {
  const gaps = ANSWER_TOKENS.map((_, index) => 0.55 + (Math.abs(Math.sin(index * 12.9898) * 43758.5453) % 1) * 0.9);
  const total = gaps.reduce((sum, gap) => sum + gap, 0);
  let at = 0;
  return gaps.map((gap) => (at += gap) / total);
})();
/** The answer as far as it has streamed, `progress` (0 to 1) through its span. */
export const answerAt = (progress: number) =>
  ANSWER_TOKENS.slice(0, TOKEN_ARRIVALS.filter((arrival) => arrival <= progress).length).join("");

/** The answer streams from this span. */
export const ANSWER_SPAN = 15;

/** What the person sees while the agent works: one line per group, and how it reads while it runs. */
export const STEPS = [
  { group: "guard", text: "Checked your message", live: "Checking your message…" },
  { group: "plan", text: "Planned a response", live: "Planning a response…" },
  { group: "skill", text: "Read the billing disputes guide", live: "Reading the billing disputes guide…" },
  { group: "search", text: "Searched the help center", live: "Searching the help center…", note: "3 articles" },
  { group: "code", text: "Checked your invoice", live: "Checking your invoice…", note: "refund issued" },
]

/** The setup at the top of the code (imports, provider, instrumentors, through the client): the Code view rests just past it. */
export const SETUP_LINES = lineOf("client = OpenAI()");

// --- Geometry -----------------------------------------------------------------

/** The timeline's left edge and width, in % of the lane. */
const X0 = 1;
const XW = 97;
export const xAt = (ms: number) => X0 + (ms / T) * XW;

const parentOf = SPANS.map((span, index) => {
  for (let i = index - 1; i >= 0; i--) if (SPANS[i]!.depth < span.depth) return i;
  return -1;
});
const laterSibling = (index: number) => {
  for (let i = index + 1; i < SPANS.length; i++) {
    if (SPANS[i]!.depth < SPANS[index]!.depth) return -1;
    if (parentOf[i] === parentOf[index]) return i;
  }
  return -1;
};

/**
 * One segment of the span tree's hierarchy lines, drawn in a row in the
 * column of a depth `level`'s glyph. A segment shows once the span that needs
 * it has started (`after`), so the tree grows with the trace.
 */
export interface Guide {
  level: number;
  /** The column's top half (up to the row above), its bottom half (down to the next), the arm across to the glyph, or a stem down from a glyph to its first child. */
  part: "up" | "down" | "arm" | "stem";
  after: number;
}

/** The hierarchy segments each row draws. */
export const GUIDES: Guide[][] = SPANS.map((span, index) => {
  const guides: Guide[] = [];
  if (span.depth > 0) {
    guides.push({ level: span.depth - 1, part: "up", after: span.start }, { level: span.depth - 1, part: "arm", after: span.start });
    const sibling = laterSibling(index);
    if (sibling >= 0) guides.push({ level: span.depth - 1, part: "down", after: SPANS[sibling]!.start });
    // Columns of ancestors that have a later sibling pass through this row.
    for (let ancestor = parentOf[index]!; ancestor > 0; ancestor = parentOf[ancestor]!) {
      const next = laterSibling(ancestor);
      if (next >= 0) {
        const level = SPANS[ancestor]!.depth - 1;
        guides.push({ level, part: "up", after: SPANS[next]!.start }, { level, part: "down", after: SPANS[next]!.start });
      }
    }
  }
  // A stem from this row's glyph down to its first child.
  if (parentOf[index + 1] === index) guides.push({ level: span.depth, part: "stem", after: SPANS[index + 1]!.start });
  return guides;
});

/** Bars ending this close to the right edge (in %) put their duration before the bar. */
export const LABEL_FLIP = 86;

export const TICKS = Array.from({ length: T / 1000 + 1 }, (_, second) => ({ label: `${second}s`, x: xAt(second * 1000) }));

// --- State at time t -----------------------------------------------------------

export const formatDuration = (ms: number) => (ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(2)}s`);
export const formatClock = (t: number) => `${(Math.max(0, t) / 1000).toFixed(2)}s`;

export type SpanState = "pending" | "open" | "done";
export const stateOf = (span: Span, t: number): SpanState =>
  t < span.start ? "pending" : t < span.end ? "open" : "done";

/** How far through a span `t` is, 0 to 1. */
export const progressOf = (span: Span, t: number) =>
  Math.max(0, Math.min(1, (t - span.start) / (span.end - span.start)));

/** The bar's width at `t`, in % of the lane. */
export const barWidth = (span: Span, t: number) =>
  Math.max(0.4, xAt(Math.min(t, span.end)) - xAt(span.start));

/** The deepest open span: the last one started that hasn't ended. */
export const deepestOpen = (t: number) => {
  for (let i = SPANS.length - 1; i >= 0; i--) if (stateOf(SPANS[i]!, t) === "open") return i;
  return -1;
};

/**
 * The span the detail would show now: the hovered one, else the agent's step
 * that is running (a failed request first, just after it fails), else (once
 * the run ends) the root. Between spans it has
 * no preference (null), so the detail keeps what it shows.
 */
export const detailCandidate = (t: number, hovered: number | null): number | null => {
  if (hovered !== null && SPANS[hovered]!.start <= t) return hovered;
  if (t < 0) return -1;
  if (t >= T) return 0;
  // A failure is the story: it holds the detail for a moment after it ends.
  const failed = SPANS.findIndex((span) => span.error && t >= span.end && t < span.end + 1200);
  if (failed >= 0) return failed;
  // Follow the agent's own steps, not every short call inside them.
  for (let i = SPANS.length - 1; i > 0; i--) if (SPANS[i]!.depth === 1 && stateOf(SPANS[i]!, t) === "open") return i;
  return null;
};

/** The span's status once it ends: OpenTelemetry leaves a plain HTTP success UNSET. */
export const statusOf = (span: Span) => (span.error ? "ERROR" : span.kind === "HTTP" ? "UNSET" : "OK");

/** The attributes recorded by `t`: the inputs from the start, the rest when the span ends. */
export const attributesAt = (span: Span, t: number) =>
  span.attributes
    .filter((attribute) => !attribute.end || t >= span.end)
    .map((attribute) => ({
      key: attribute.key,
      value: attribute.value.replace(/\n/g, " "),
      error: !!attribute.error,
    }));

/** Each step's extent: from its first span's start to its last span's end. */
export const STEP_SPANS = STEPS.map((step) => {
  const members = SPANS.filter((span) => span.group === step.group);
  return { start: Math.min(...members.map((s) => s.start)), end: Math.max(...members.map((s) => s.end)) };
});

/** The code lines lit at `t`: the deepest open span's line, and every open span's call site. */
export function codeAt(t: number) {
  const running = new Set<number>();
  let active = 0;
  for (const span of SPANS) {
    if (stateOf(span, t) !== "open") continue;
    active = span.line;
    running.add(span.line);
    if (span.call) running.add(span.call);
  }
  return { active, running };
}
