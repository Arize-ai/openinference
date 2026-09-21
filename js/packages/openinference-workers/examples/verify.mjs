import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { setTimeout } from "node:timers/promises";

const [base = "http://localhost:8787", project = "openinference-workers-cloudflare"] =
  process.argv.slice(2);
const run = `verify-${randomUUID()}`;
const read = () =>
  JSON.parse(
    execFileSync(
      "px",
      ["span", "list", "--project", project, "--format", "raw", "--no-progress", "--limit", "500"],
      { encoding: "utf8" },
    ),
  );
const call = async (path, session, headers, counter = session) => {
  const response = await fetch(`${base}${path}?session=${session}&counter=${counter}`, { headers });
  return {
    status: response.status,
    body: response.ok ? await response.json() : await response.text(),
  };
};
const results = await Promise.all(
  Array.from({ length: 12 }, (_, i) => call("/", `${run}-${i}`, undefined, `${run}-counter`)),
);
assert(results.every((result) => result.status === 200));
assert.equal(new Set(results.map((result) => result.body.traceId)).size, 12);
assert.deepEqual(
  results.map((result) => result.body.result.count).sort((a, b) => a - b),
  Array.from({ length: 12 }, (_, i) => i + 1),
);
const suppressed = await call("/suppressed", `${run}-suppressed`);
assert.deepEqual(suppressed.body, { suppressed: true });
const error = await call("/error", `${run}-error`);
assert.equal(error.status, 500);
const explicitError = await call("/status-error", `${run}-explicit-error`);
assert.equal(explicitError.status, 500);
const remoteTraceId = randomUUID().replaceAll("-", "");
const remoteParentId = "0123456789abcdef";
const remote = await call("/", `${run}-remote`, {
  traceparent: `00-${remoteTraceId}-${remoteParentId}-01`,
  tracestate: "example=review",
});
assert.equal(remote.status, 200);
assert.equal(remote.body.traceId, remoteTraceId);
let spans;
for (let attempt = 0; attempt < 20; attempt++) {
  spans = read();
  if (
    results.every(
      (result) => spans.filter((s) => s.context.trace_id === result.body.traceId).length === 3,
    ) &&
    spans.some((s) => s.attributes["session.id"] === `${run}-error`) &&
    spans.some((s) => s.attributes["session.id"] === `${run}-explicit-error`) &&
    spans.filter((s) => s.context.trace_id === remoteTraceId).length === 3
  )
    break;
  await setTimeout(500);
}
for (const [i, result] of results.entries()) {
  const group = spans.filter((s) => s.context.trace_id === result.body.traceId);
  assert.equal(group.length, 3, "Expected exactly three application spans per request");
  const byName = Object.fromEntries(group.map((s) => [s.name, s]));
  const worker = byName["worker.request"];
  const object = byName["counter.request"];
  const tool = byName["counter.increment"];
  assert.equal(worker.parent_id, null);
  assert.equal(object.parent_id, worker.context.span_id);
  assert.equal(tool.parent_id, object.context.span_id);
  assert.equal(tool.span_kind, "TOOL");
  assert.equal(worker.attributes["session.id"], `${run}-${i}`);
  assert.equal(worker.attributes["user.id"], "example-user");
  assert.equal(worker.attributes["metadata.scenario"], "counter");
  assert.equal(worker.attributes["input.value"], "__REDACTED__");
  const tags = worker.attributes["tag.tags"];
  assert.deepEqual(typeof tags === "string" ? JSON.parse(tags) : tags, ["example"]);
  assert.equal(worker.status_code, "UNSET");
  assert.equal(result.body.result.traceId, result.body.traceId);
}
assert(!spans.some((s) => s.attributes["session.id"] === `${run}-suppressed`));
const failed = spans.find((s) => s.attributes["session.id"] === `${run}-error`);
assert.equal(failed.status_code, "ERROR");
assert(failed.events.some((e) => e.name === "exception"));
assert.equal(
  spans.find((s) => s.attributes["session.id"] === `${run}-explicit-error`).status_code,
  "ERROR",
);
const remoteSpans = spans.filter((s) => s.context.trace_id === remoteTraceId);
assert.equal(remoteSpans.find((s) => s.name === "worker.request").parent_id, remoteParentId);
assert.equal(
  remoteSpans.find((s) => s.name === "counter.request").parent_id,
  remoteSpans.find((s) => s.name === "worker.request").context.span_id,
);
console.log(
  JSON.stringify(
    {
      runtime: base,
      project,
      run,
      concurrentTraces: 12,
      durableObjectUpdates: "passed",
      spans: 41,
      parentage: "passed",
      context: "passed",
      masking: "passed",
      suppression: "passed",
      error: "passed",
      remoteParent: "passed",
      explicitError: "passed",
      traceIds: results.map((r) => r.body.traceId),
    },
    null,
    2,
  ),
);
