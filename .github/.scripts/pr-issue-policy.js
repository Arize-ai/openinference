// Enforces two CONTRIBUTING rules on pull requests from people without write
// access: every PR links an issue, and each issue has at most one open PR.
const fs = require("node:fs");
const path = require("node:path");

const LABELS = {
  missingIssue: "policy: no linked issue",
  duplicate: "policy: duplicate PR",
  skip: "skip issue policy",
};

const RULE_LABELS = {
  "missing-linked-issue": LABELS.missingIssue,
  duplicate: LABELS.duplicate,
};

const RULE_LINE_PREFIXES = {
  "missing-linked-issue": "- Link the issue",
  duplicate: "- One PR per issue",
};

const PR_QUERY = `
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        number
        url
        state
        createdAt
        author { login __typename }
        labels(first: 50) { nodes { name } }
        closingIssuesReferences(first: 20) {
          nodes {
            number
            url
            repository { nameWithOwner }
            closedByPullRequestsReferences(first: 50, includeClosedPrs: false) {
              nodes { number url state createdAt repository { nameWithOwner } }
            }
          }
        }
      }
    }
  }
`;

const OPEN_PRS_QUERY = `
  query($owner: String!, $repo: String!, $cursor: String) {
    repository(owner: $owner, name: $repo) {
      pullRequests(states: OPEN, first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { number }
      }
    }
  }
`;

function isOlder(a, b) {
  return (
    a.createdAt < b.createdAt ||
    (a.createdAt === b.createdAt && a.number < b.number)
  );
}

function collectFacts(prNode, repoFullName) {
  const linkedIssues = [];
  const olderOpenPrs = [];
  for (const issue of prNode.closingIssuesReferences.nodes) {
    const issueRepo = issue.repository.nameWithOwner;
    const issueRef =
      issueRepo === repoFullName
        ? `#${issue.number}`
        : `${issueRepo}#${issue.number}`;
    linkedIssues.push({ ref: issueRef, url: issue.url });
    for (const other of issue.closedByPullRequestsReferences.nodes) {
      const sameRepo = other.repository.nameWithOwner === repoFullName;
      if (sameRepo && other.state === "OPEN" && isOlder(other, prNode)) {
        olderOpenPrs.push({
          issueRef,
          number: other.number,
          url: other.url,
          createdAt: other.createdAt,
        });
      }
    }
  }
  olderOpenPrs.sort((a, b) => (isOlder(a, b) ? -1 : 1));
  return {
    state: prNode.state,
    labels: prNode.labels.nodes.map((label) => label.name),
    linkedIssues,
    olderOpenPrs,
  };
}

function findViolation({ linkedIssues, olderOpenPrs }) {
  if (linkedIssues.length === 0) return { rule: "missing-linked-issue" };
  if (olderOpenPrs.length > 0)
    return { rule: "duplicate", duplicateOf: olderOpenPrs[0] };
  return null;
}

function decide(facts) {
  if (facts.exempt)
    return { kind: "noop", reason: "author has write access or is a bot" };
  if (facts.labels.includes(LABELS.skip))
    return { kind: "noop", reason: `labeled "${LABELS.skip}"` };
  if (facts.event === "reopened" && !facts.senderIsAuthor) {
    return { kind: "noop", reason: "reopened by a maintainer" };
  }
  const violation = findViolation(facts);
  if (facts.state === "OPEN") {
    return violation
      ? { kind: "close", ...violation }
      : { kind: "noop", reason: "follows the policy" };
  }
  const closedByPolicy = Object.values(RULE_LABELS).some((label) =>
    facts.labels.includes(label),
  );
  if (facts.state === "CLOSED" && closedByPolicy && !violation)
    return { kind: "reopen" };
  return { kind: "noop", reason: `already ${facts.state.toLowerCase()}` };
}

function ruleLinks(contributingText, repoFullName, branch) {
  const lines = contributingText.split("\n");
  const links = {};
  for (const [rule, prefix] of Object.entries(RULE_LINE_PREFIXES)) {
    const index = lines.findIndex((line) => line.startsWith(prefix));
    if (index === -1)
      throw new Error(`CONTRIBUTING no longer states the "${prefix}" rule`);
    links[rule] =
      `https://github.com/${repoFullName}/blob/${branch}/CONTRIBUTING#L${index + 1}`;
  }
  return links;
}

function renderComment(verdict, links) {
  const footer = `\n\n<sub>This check runs automatically. A maintainer can add the \`${LABELS.skip}\` label to opt a PR out.</sub>`;
  if (verdict.rule === "missing-linked-issue") {
    return (
      `Thanks for the PR. It doesn't link an issue, which our [contributing guide requires](${links[verdict.rule]}), so we're closing it.\n\n` +
      "Add a closing keyword such as `Fixes #1234` to the description and this PR reopens on its own. " +
      "If no issue exists yet, open one first so a maintainer can confirm the change is wanted." +
      footer
    );
  }
  const { issueRef, url } = verdict.duplicateOf;
  return (
    `Thanks for the PR. ${issueRef} already has an open PR (${url}), and we keep [one PR per issue](${links[verdict.rule]}), so we're closing this one as a duplicate.\n\n` +
    `If your approach differs in a way that matters, say so on ${issueRef}.` +
    footer
  );
}

async function evaluate({ github, owner, repo, number, event, sender }) {
  const { repository } = await github.graphql(PR_QUERY, {
    owner,
    repo,
    number,
  });
  const pr = repository.pullRequest;
  let exempt = pr.author?.__typename === "Bot";
  if (!exempt && pr.author) {
    const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
      owner,
      repo,
      username: pr.author.login,
    });
    exempt = data.permission === "admin" || data.permission === "write";
  }
  const facts = {
    ...collectFacts(pr, `${owner}/${repo}`),
    event,
    senderIsAuthor: sender === pr.author?.login,
    exempt,
  };
  return { pr, verdict: decide(facts) };
}

async function apply({ github, owner, repo, pr, verdict, links }) {
  const issue = { owner, repo, issue_number: pr.number };
  if (verdict.kind === "close") {
    await github.rest.issues.addLabels({
      ...issue,
      labels: [RULE_LABELS[verdict.rule]],
    });
    await github.rest.pulls.update({
      owner,
      repo,
      pull_number: pr.number,
      state: "closed",
    });
    await github.rest.issues.createComment({
      ...issue,
      body: renderComment(verdict, links),
    });
  } else if (verdict.kind === "reopen") {
    await github.rest.pulls.update({
      owner,
      repo,
      pull_number: pr.number,
      state: "open",
    });
    for (const label of Object.values(RULE_LABELS)) {
      if (pr.labels.nodes.some((node) => node.name === label)) {
        await github.rest.issues.removeLabel({ ...issue, name: label });
      }
    }
  }
}

function describe(pr, verdict) {
  if (verdict.kind === "noop")
    return `#${pr.number}: no action (${verdict.reason})`;
  if (verdict.kind === "reopen") return `#${pr.number}: reopen`;
  const detail = verdict.duplicateOf
    ? ` of #${verdict.duplicateOf.number} for ${verdict.duplicateOf.issueRef}`
    : "";
  return `#${pr.number}: close, ${verdict.rule}${detail}`;
}

async function listOpenPrNumbers(github, owner, repo) {
  const numbers = [];
  let cursor = null;
  do {
    const { repository } = await github.graphql(OPEN_PRS_QUERY, {
      owner,
      repo,
      cursor,
    });
    const page = repository.pullRequests;
    numbers.push(...page.nodes.map((node) => node.number));
    cursor = page.pageInfo.hasNextPage ? page.pageInfo.endCursor : null;
  } while (cursor);
  return numbers;
}

module.exports = async ({
  github,
  context,
  core,
  enforce,
  contributingPath,
}) => {
  const { owner, repo } = context.repo;
  const branch = context.payload.repository?.default_branch ?? "main";
  const contributing = fs.readFileSync(
    contributingPath ?? path.join(__dirname, "..", "..", "CONTRIBUTING"),
    "utf8",
  );
  const links = ruleLinks(contributing, `${owner}/${repo}`, branch);

  const targets = context.payload.pull_request
    ? [
        {
          number: context.payload.pull_request.number,
          event: context.payload.action,
        },
      ]
    : (await listOpenPrNumbers(github, owner, repo)).map((number) => ({
        number,
        event: "sweep",
      }));

  const results = [];
  for (const { number, event } of targets) {
    const { pr, verdict } = await evaluate({
      github,
      owner,
      repo,
      number,
      event,
      sender: context.payload.sender?.login,
    });
    const line = describe(pr, verdict);
    core.info(enforce || verdict.kind === "noop" ? line : `${line} [log only]`);
    if (enforce) await apply({ github, owner, repo, pr, verdict, links });
    results.push({ number: pr.number, ...verdict });
  }
  return results;
};

Object.assign(module.exports, {
  LABELS,
  collectFacts,
  decide,
  renderComment,
  ruleLinks,
});
