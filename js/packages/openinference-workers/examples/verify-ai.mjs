import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { setTimeout } from "node:timers/promises";

const [
  base = "http://localhost:8787",
  project = process.env.PHOENIX_PROJECT ?? "openinference-workers-cloudflare",
] = process.argv.slice(2);
const run = `ai-${randomUUID()}`;
const cases = [
  "",
  "/stream",
  "/object",
  "/masked",
  "/masked/stream",
  "/tools",
  "/error",
  "/suppressed",
];
const results = new Map();
for (const path of cases) {
  const response = await fetch(`${base}/ai${path}?session=${run}${path}`);
  assert.equal(response.status, path === "/error" ? 500 : 200, `Status for ${path}`);
  const body = await response.text();
  results.set(path, body);
  if (path.endsWith("/stream")) assert(body.includes("data:"));
  else if (path !== "/error") assert(JSON.parse(body).result);
}
const read = () =>
  JSON.parse(
    execFileSync(
      "px",
      ["span", "list", "--project", project, "--format", "raw", "--no-progress", "--limit", "500"],
      { encoding: "utf8" },
    ),
  );
let spans = [];
for (let attempt = 0; attempt < 30; attempt++) {
  spans = read().filter((s) => String(s.attributes["session.id"] ?? "").startsWith(run));
  if (spans.filter((s) => s.span_kind === "LLM").length === 7) break;
  await setTimeout(500);
}
assert.equal(spans.filter((s) => s.span_kind === "LLM").length, 7);
assert(!spans.some((s) => s.attributes["session.id"] === `${run}/suppressed`));
for (const path of cases.filter((p) => p !== "/suppressed")) {
  const group = spans.filter((s) => s.attributes["session.id"] === run + path);
  const llms = group.filter((s) => s.span_kind === "LLM");
  assert.equal(llms.length, 1, `One LLM span for ${path}`);
  const llm = llms[0];
  const attrs = llm.attributes;
  assert(
    group.some(
      (s) => s.context.span_id === llm.parent_id && s.context.trace_id === llm.context.trace_id,
    ),
    "LLM parent exported in same trace",
  );
  assert.equal(llm.name, "Workers AI.run");
  assert.equal(llm.status_code, path === "/error" ? "ERROR" : "UNSET");
  assert(attrs["llm.model_name"]);
  if (path === "/error") continue;
  if (path.startsWith("/masked")) {
    assert.equal(attrs["input.value"], "__REDACTED__");
    assert.equal(attrs["output.value"], "__REDACTED__");
    assert(
      !Object.keys(attrs).some(
        (k) => k.startsWith("llm.input_messages") || k.startsWith("llm.output_messages"),
      ),
    );
  } else {
    assert.equal(attrs["llm.input_messages.0.message.role"], "user");
    assert.equal(attrs["llm.output_messages.0.message.role"], "assistant");
    if (path === "/tools") {
      assert.equal(
        attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"],
        "get_weather",
      );
      assert.equal(
        JSON.parse(attrs["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"])
          .city,
        "Boston",
      );
      assert(attrs["llm.tools.0.tool.json_schema"]);
    } else {
      const expected = path.endsWith("/stream")
        ? results
            .get(path)
            .split("\n")
            .filter((l) => l.startsWith("data: {"))
            .map((l) => JSON.parse(l.slice(6)).response ?? "")
            .join("")
        : JSON.parse(results.get(path)).result.response;
      assert.equal(attrs["llm.output_messages.0.message.content"], expected);
      assert(expected.length > 0);
    }
  }
  assert(attrs["llm.token_count.prompt"] > 0);
  assert(attrs["llm.token_count.completion"] > 0);
  assert.equal(
    attrs["llm.token_count.total"],
    attrs["llm.token_count.prompt"] + attrs["llm.token_count.completion"],
  );
}
console.log(
  JSON.stringify({ run, project, spans: spans.length, llmSpans: 7, passed: true }, null, 2),
);
