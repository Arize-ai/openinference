const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  LABELS,
  collectFacts,
  decide,
  renderComment,
  ruleLinks,
} = require("./pr-issue-policy");

const REPO = "Arize-ai/openinference";

function facts(overrides = {}) {
  return {
    state: "OPEN",
    event: "opened",
    senderIsAuthor: true,
    exempt: false,
    labels: [],
    linkedIssues: [{ ref: "#100", url: "https://example.test/issues/100" }],
    olderOpenPrs: [],
    ...overrides,
  };
}

test("closes an open community PR that links no issue", () => {
  assert.deepEqual(decide(facts({ linkedIssues: [] })), {
    kind: "close",
    rule: "missing-linked-issue",
  });
});

test("leaves maintainers and bots alone even without a linked issue", () => {
  assert.equal(decide(facts({ exempt: true, linkedIssues: [] })).kind, "noop");
});

test("leaves a PR carrying the skip label alone", () => {
  const verdict = decide(facts({ labels: [LABELS.skip], linkedIssues: [] }));
  assert.equal(verdict.kind, "noop");
});

test("keeps a linked PR when no older PR claims the issue", () => {
  assert.equal(decide(facts()).kind, "noop");
});

test("closes a newer PR for an issue that already has an open PR", () => {
  const older = {
    issueRef: "#100",
    number: 7,
    url: "u7",
    createdAt: "2026-01-02T00:00:00Z",
  };
  assert.deepEqual(decide(facts({ olderOpenPrs: [older] })), {
    kind: "close",
    rule: "duplicate",
    duplicateOf: older,
  });
});

test("reopens a PR the policy closed once its author links an issue", () => {
  const verdict = decide(
    facts({ state: "CLOSED", event: "edited", labels: [LABELS.missingIssue] }),
  );
  assert.deepEqual(verdict, { kind: "reopen" });
});

test("does not reopen a PR the policy closed while it still breaks a rule", () => {
  const verdict = decide(
    facts({
      state: "CLOSED",
      event: "edited",
      labels: [LABELS.missingIssue],
      linkedIssues: [],
    }),
  );
  assert.equal(verdict.kind, "noop");
});

test("does not reopen a PR a maintainer closed by hand", () => {
  assert.equal(
    decide(facts({ state: "CLOSED", event: "edited" })).kind,
    "noop",
  );
});

test("respects a maintainer reopening a PR the policy closed", () => {
  const verdict = decide(
    facts({ event: "reopened", senderIsAuthor: false, linkedIssues: [] }),
  );
  assert.equal(verdict.kind, "noop");
});

test("closes again when the author reopens without linking an issue", () => {
  const verdict = decide(facts({ event: "reopened", linkedIssues: [] }));
  assert.equal(verdict.rule, "missing-linked-issue");
});

function prNode({
  number = 50,
  createdAt = "2026-02-01T00:00:00Z",
  issues = [],
} = {}) {
  return {
    number,
    createdAt,
    state: "OPEN",
    labels: { nodes: [{ name: "triage" }] },
    closingIssuesReferences: {
      nodes: issues.map(({ number: issue, repo = REPO, prs = [] }) => ({
        number: issue,
        url: `https://github.com/${repo}/issues/${issue}`,
        repository: { nameWithOwner: repo },
        closedByPullRequestsReferences: {
          nodes: prs.map((pr) => ({
            url: `https://github.com/${pr.repo ?? REPO}/pull/${pr.number}`,
            repository: { nameWithOwner: pr.repo ?? REPO },
            state: "OPEN",
            ...pr,
          })),
        },
      })),
    },
  };
}

test("collects only older open PRs in this repo, oldest first", () => {
  const node = prNode({
    issues: [
      {
        number: 100,
        prs: [
          { number: 50, createdAt: "2026-02-01T00:00:00Z" },
          { number: 60, createdAt: "2026-03-01T00:00:00Z" },
          { number: 40, createdAt: "2026-01-15T00:00:00Z" },
          {
            number: 30,
            createdAt: "2026-01-20T00:00:00Z",
            repo: "Arize-ai/phoenix",
          },
          { number: 20, createdAt: "2026-01-10T00:00:00Z" },
        ],
      },
    ],
  });
  const { olderOpenPrs, linkedIssues, labels } = collectFacts(node, REPO);
  assert.deepEqual(
    olderOpenPrs.map((pr) => pr.number),
    [20, 40],
  );
  assert.deepEqual(linkedIssues, [
    { ref: "#100", url: `https://github.com/${REPO}/issues/100` },
  ]);
  assert.deepEqual(labels, ["triage"]);
});

test("ignores merged PRs, which the issue lookup still returns", () => {
  const node = prNode({
    issues: [
      {
        number: 100,
        prs: [
          { number: 20, createdAt: "2026-01-10T00:00:00Z", state: "MERGED" },
        ],
      },
    ],
  });
  assert.deepEqual(collectFacts(node, REPO).olderOpenPrs, []);
});

test("breaks a createdAt tie by PR number", () => {
  const node = prNode({
    issues: [
      {
        number: 100,
        prs: [
          { number: 49, createdAt: "2026-02-01T00:00:00Z" },
          { number: 51, createdAt: "2026-02-01T00:00:00Z" },
        ],
      },
    ],
  });
  assert.deepEqual(
    collectFacts(node, REPO).olderOpenPrs.map((pr) => pr.number),
    [49],
  );
});

test("names a cross-repo issue with its repository", () => {
  const node = prNode({ issues: [{ number: 9, repo: "Arize-ai/phoenix" }] });
  assert.equal(
    collectFacts(node, REPO).linkedIssues[0].ref,
    "Arize-ai/phoenix#9",
  );
});

const contributing = fs.readFileSync(
  path.join(__dirname, "..", "..", "CONTRIBUTING"),
  "utf8",
);

test("links each comment to the CONTRIBUTING line that states its rule", () => {
  const links = ruleLinks(contributing, REPO, "main");
  for (const [rule, prefix] of [
    ["missing-linked-issue", "- Link the issue"],
    ["duplicate", "- One PR per issue"],
  ]) {
    const line = Number(links[rule].split("#L")[1]);
    assert.ok(
      contributing.split("\n")[line - 1].startsWith(prefix),
      `${rule} link points at its rule`,
    );
  }
});

test("renders comments that name the rule and the way out", () => {
  const links = ruleLinks(contributing, REPO, "main");
  const missing = renderComment(
    { kind: "close", rule: "missing-linked-issue" },
    links,
  );
  assert.match(missing, /Fixes #/);
  assert.ok(missing.includes(links["missing-linked-issue"]));

  const duplicate = renderComment(
    {
      kind: "close",
      rule: "duplicate",
      duplicateOf: {
        issueRef: "#100",
        number: 7,
        url: "https://example.test/pull/7",
      },
    },
    links,
  );
  assert.ok(duplicate.includes("#100"));
  assert.ok(duplicate.includes("https://example.test/pull/7"));
  assert.ok(duplicate.includes(links.duplicate));
});

function fakeGithub(pr, permission = "read") {
  const writes = [];
  const record = (call) => async (args) => writes.push([call, args]);
  return {
    writes,
    graphql: async () => ({ repository: { pullRequest: pr } }),
    rest: {
      repos: {
        getCollaboratorPermissionLevel: async () => ({ data: { permission } }),
      },
      issues: {
        addLabels: record("addLabels"),
        removeLabel: record("removeLabel"),
        createComment: record("createComment"),
      },
      pulls: { update: record("update") },
    },
  };
}

function eventContext(number, action) {
  return {
    repo: { owner: "Arize-ai", repo: "openinference" },
    payload: {
      action,
      pull_request: { number },
      sender: { login: "contributor" },
    },
  };
}

const run = require("./pr-issue-policy");
const core = { info() {} };

test("enforcing labels, closes, then explains an unlinked PR", async () => {
  const github = fakeGithub({
    ...prNode(),
    author: { login: "contributor", __typename: "User" },
  });
  await run({
    github,
    context: eventContext(50, "opened"),
    core,
    enforce: true,
  });
  assert.deepEqual(
    github.writes.map(([call, args]) => [
      call,
      args.labels ?? args.state ?? "body",
    ]),
    [
      ["addLabels", [LABELS.missingIssue]],
      ["update", "closed"],
      ["createComment", "body"],
    ],
  );
});

test("enforcing reopens and unlabels once the author links an issue", async () => {
  const pr = {
    ...prNode({ issues: [{ number: 100 }] }),
    state: "CLOSED",
    labels: { nodes: [{ name: LABELS.missingIssue }] },
    author: { login: "contributor", __typename: "User" },
  };
  const github = fakeGithub(pr);
  await run({
    github,
    context: eventContext(50, "edited"),
    core,
    enforce: true,
  });
  assert.deepEqual(
    github.writes.map(([call, args]) => [call, args.state ?? args.name]),
    [
      ["update", "open"],
      ["removeLabel", LABELS.missingIssue],
    ],
  );
});

test("logging only never writes", async () => {
  const github = fakeGithub({
    ...prNode(),
    author: { login: "contributor", __typename: "User" },
  });
  const [result] = await run({
    github,
    context: eventContext(50, "opened"),
    core,
    enforce: false,
  });
  assert.equal(result.kind, "close");
  assert.deepEqual(github.writes, []);
});
