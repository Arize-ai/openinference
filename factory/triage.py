"""Triage one OpenInference issue in the factory sandbox.

`start` checks that the issue can be triaged now and creates a Managed Agents session with the
issue-triager agent, which classifies the issue by the issue-triage skill, has subagents
reproduce it and locate its code path, and submits its triage through the `submit_triage`
custom tool. The session's GitHub token can only read. `watch` answers each submission, sending
back what is wrong with it until it is valid, and once the session ends labels the issue,
closes it if it duplicates an older open issue, decides whether it is a `factory-candidate`,
and posts the report on the issue. Only this
script writes to GitHub, as the factory's GitHub App, with a token no session sees. Agent,
environment and vault IDs come from claude-lock.json, which `ant apply` writes.
"""

import argparse
import json
import os
import re
import secrets
import subprocess
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import anthropic
from anthropic.types.beta.sessions import BetaManagedAgentsUserCustomToolResultEventParams
from dispatch import (
    ENVIRONMENT_LOCK_KEY,
    FACTORY_DIR,
    FACTORY_LABEL,
    FACTORY_LANGUAGE_LABELS,
    ISSUE_TYPE_LABELS,
    LOCK_FILE_PATH,
    NOT_READY_LABELS,
    STARTING_PERMISSIONS,
    comment_on_issue,
    format_cost,
    format_session_link,
    list_open_pr_urls,
    read_factory_login,
    read_lock_id,
    read_repo_permission,
    run_gh,
    wait_for_session,
    write_github_outputs,
)
from github_app import GitHubAppToken

TRIAGER_LOCK_KEY = "./agents/issue-triager.md"
READONLY_VAULT_LOCK_KEY = "./vaults/github-readonly.yaml"
SUBMIT_TOOL_NAME = "submit_triage"
CANDIDATE_LABEL = "factory-candidate"
NEEDS_INFORMATION_LABEL = "needs information"
DUPLICATE_LABEL = "duplicate"
# Ends every report, so a later run can count the earlier ones. Only the factory's count.
REPORT_MARKER = "<!-- factory-triage"
# The session reads the issue, searches issues and PRs, and clones the repo. Triage starts on
# issues anyone can open, so its sessions must not be able to push, label or comment.
SESSION_TOKEN_PERMISSIONS = {"contents": "read", "issues": "read", "pull_requests": "read"}
# This script labels the issue, closes a duplicate and posts the report, and needs nothing more.
REPORT_TOKEN_PERMISSIONS = {"issues": "write"}
# Sessions a reporter can start on one issue, by opening it and replying, failed ones
# included; a maintainer can always run triage again. Counted over SESSION_WINDOW.
MAX_SESSIONS_PER_ISSUE = 3
# Sessions started in a day before requesters without write access get none, since anyone can
# open issues.
MAX_SESSIONS_PER_DAY = 20
SESSION_WINDOW = timedelta(days=90)
# The names Factory Sync gives the triager and its vault. claude-lock.json lives on a branch the
# app's token can push, so the IDs it holds are checked against these before a session starts.
TRIAGER_AGENT_NAME = "openinference-issue-triager"
READONLY_VAULT_NAME = "factory-github-readonly"
# Invalid submissions answered before the dispatcher stops answering.
MAX_INVALID_SUBMISSIONS = 3
# The label families stages 1 to 5 of the issue-triage skill write. A submission may add only
# these, and only those that exist on the repo.
TRIAGE_LABEL_PATTERN = re.compile(
    r"bug|enhancement|documentation|cleanup|question|examples|security|new instrumentation"
    r"|instrumentation(: .+)?|language: (python|js|java)|c/[a-z-]+|size:[SML]"
)
CANDIDATE_SIZE_LABELS = {"size:S", "size:M"}
SUBMISSION_FIELD_TYPES = {
    "issue_type": str,
    "labels": list,
    "summary": str,
    "sufficiency": dict,
    "reproduction": dict,
    "confidence": dict,
    "open_questions": list,
    "maintainer_questions": list,
    "code_pointers": list,
    "shipped": dict,
    "duplicate_of": dict,
    "possible_duplicates": list,
    "needs_live_recording": bool,
    "sensitive_data": str,
}
SUFFICIENCY_QUESTIONS = {
    "current_behavior": "What happens now",
    "expected_behavior": "What should happen",
    "where_to_start": "Where to start",
}
SUFFICIENCY_GRADES = ("answered", "partial", "missing")
REPRODUCTION_STATUSES = ("confirmed", "not_confirmed", "not_attempted")
CONFIDENCE_LEVELS = ("high", "medium", "low")
SHIPPED_STATUSES = ("no", "partly", "fully")
# The verdict names whose move it is: a builder's, the reporter's, or a maintainer's, or that
# the issue was closed as a duplicate.
VERDICT_HEADINGS = {
    "ready": "Ready to work on",
    "needs-information": "Needs information",
    "needs-maintainer": "Needs a maintainer",
    "duplicate": "Duplicate",
}
REPO_ROOT = FACTORY_DIR.parent
LINE_SUFFIX_PATTERN = re.compile(r":\d+(-\d+)?$")
# Agent-written prose is escaped character by character: a backslash before the markdown
# punctuation that could open markup, a link or a table cell, and a word joiner where GitHub
# would find a mention or an issue reference in the rendered text. Code spans the agent wrote
# are fenced again by the dispatcher.
PROSE_ESCAPES = {
    **{character: "\\" + character for character in "\\`*_{}[]()<>!|~&"},
    "@": "@\u2060",
    "#": "#\u2060",
}
CODE_SPAN_PATTERN = re.compile(r"(?<![`\\])(`+)(?!`)(.+?)(?<!`)\1(?!`)")
# `@claude` anywhere in a comment, code included, starts claude.yml.
CLAUDE_MENTION_PATTERN = re.compile(r"@(?=claude)", re.IGNORECASE)
MAX_FIELD_CHARACTERS = 1000
MAX_SUMMARY_CHARACTERS = 2000
MAX_EVIDENCE_CHARACTERS = 3000
MAX_CODE_CHARACTERS = 6000
MAX_CELL_CHARACTERS = 300
# GitHub rejects a longer comment; a report over it drops its Investigation section.
MAX_COMMENT_CHARACTERS = 65536
# Items a submission may hold per list, so a report stays readable and under the limit.
MAX_LIST_ITEMS = {
    "labels": 20,
    "open_questions": 5,
    "maintainer_questions": 5,
    "code_pointers": 10,
    "possible_duplicates": 5,
}
README_URL = (
    "https://github.com/Arize-ai/openinference/blob/main/factory/README.md#triaging-an-issue"
)


def read_issue(*, issue_number: int) -> dict[str, Any]:
    return json.loads(
        run_gh(
            "issue",
            "view",
            str(issue_number),
            "--json",
            "number,state,author,labels,assignees,closedByPullRequestsReferences",
        )
    )


def read_repo_label_names() -> set[str]:
    return set(
        run_gh("label", "list", "--limit", "500", "--json", "name", "--jq", ".[].name").split("\n")
    ) - {""}


def read_triage_history(
    *, issue_number: int, author_login: str, factory_login: str
) -> tuple[list[str], Optional[str]]:
    """Return when the factory posted each triage report, and when the author last commented.

    Comments come from the REST API, whose logins carry the `[bot]` suffix that
    FACTORY_GITHUB_LOGIN does. Only the factory's comments count, so a marker someone else
    pastes changes nothing.
    """
    comment_lines = run_gh(
        "api",
        "--paginate",
        f"repos/{{owner}}/{{repo}}/issues/{issue_number}/comments",
        "--jq",
        ".[] | {login: .user.login, created_at, "
        f'is_report: ((.body // "") | contains("{REPORT_MARKER}"))}}',
    ).splitlines()
    comments = [json.loads(comment_line) for comment_line in comment_lines if comment_line]
    report_times = [
        comment["created_at"]
        for comment in comments
        if comment["login"] == factory_login and comment["is_report"]
    ]
    author_comment_times = [
        comment["created_at"] for comment in comments if comment["login"] == author_login
    ]
    return report_times, max(author_comment_times, default=None)


def list_triage_sessions(*, client: anthropic.Anthropic, agent_id: str) -> list[Any]:
    """Return the triage sessions from SESSION_WINDOW, archived ones included, oldest first."""
    created_after = datetime.now(timezone.utc) - SESSION_WINDOW
    sessions = client.beta.sessions.list(
        agent_id=agent_id, created_at_gte=created_after, include_archived=True, limit=100
    )
    return sorted(sessions, key=lambda session: (session.created_at, session.id))


def count_issue_sessions(*, sessions: list[Any], repo: str, issue_number: int) -> int:
    return sum(
        1
        for session in sessions
        if session.metadata.get("repo") == repo
        and session.metadata.get("issue") == str(issue_number)
    )


def count_sessions_in_day_before(*, sessions: list[Any], moment: datetime) -> int:
    """Count the sessions started in the day before `moment`."""
    return sum(
        1 for session in sessions if moment - timedelta(days=1) <= session.created_at < moment
    )


def read_triage_ids(*, client: anthropic.Anthropic, lock: dict[str, Any]) -> tuple[str, str]:
    """Return the triager's agent ID and the read-only vault's ID, checked against the workspace.

    A session in another lane could push a lock that maps the read-only vault to the one the
    bug lane fills with write tokens. Only the workspace's API key can rename an agent or a
    vault, so a lock whose IDs name anything else stops the run.
    """
    agent_id = read_lock_id(lock=lock, lock_key=TRIAGER_LOCK_KEY)
    vault_id = read_lock_id(lock=lock, lock_key=READONLY_VAULT_LOCK_KEY)
    agent_name = client.beta.agents.retrieve(agent_id).name
    vault_name = client.beta.vaults.retrieve(vault_id).display_name
    if agent_name != TRIAGER_AGENT_NAME or vault_name != READONLY_VAULT_NAME:
        raise SystemExit(
            f"::error::claude-lock.json maps the triager to {agent_name!r} and its vault to "
            f"{vault_name!r}, not {TRIAGER_AGENT_NAME!r} and {READONLY_VAULT_NAME!r}. Check "
            "the factory-state branch's history, and run Factory Sync to rewrite it."
        )
    return agent_id, vault_id


def find_admission_problem(
    *,
    issue: dict[str, Any],
    trigger: str,
    requester_permission: str,
    report_times: list[str],
    author_comment_time: Optional[str],
    issue_session_count: int,
    recent_session_count: int,
) -> Optional[str]:
    """Return why the issue shouldn't be triaged now, or None if it can be."""
    if issue["state"] != "OPEN":
        return "the issue is closed"
    label_names = {label["name"] for label in issue["labels"]}
    if "agent-in-progress" in label_names:
        return "claude-implement-issue.yml is working on it (`agent-in-progress`)"
    if FACTORY_LABEL in label_names:
        return "the factory is building it (`factory`)"
    # Two quick replies queue two runs; the first one's report covers both.
    if (
        trigger == "reply"
        and report_times
        and author_comment_time is not None
        and report_times[-1] > author_comment_time
    ):
        return "the latest triage report already covers the author's last reply"
    if requester_permission in STARTING_PERMISSIONS:
        return None
    if issue_session_count >= MAX_SESSIONS_PER_ISSUE:
        return f"it has had {issue_session_count} triage sessions; a maintainer can run it again"
    if recent_session_count >= MAX_SESSIONS_PER_DAY:
        return f"the factory started {recent_session_count} triage sessions in the last day"
    return None


def create_token(
    *,
    client: anthropic.Anthropic,
    repo: str,
    permissions: dict[str, str],
    vault_id: Optional[str],
) -> GitHubAppToken:
    return GitHubAppToken(
        client=client,
        repo=repo,
        vault_id=vault_id,
        app_client_id=os.environ["FACTORY_APP_CLIENT_ID"],
        app_private_key=os.environ["FACTORY_APP_PRIVATE_KEY"],
        permissions=permissions,
    )


def create_session(
    *,
    client: anthropic.Anthropic,
    repo: str,
    issue: dict[str, Any],
    agent_id: str,
    vault_id: str,
    github_token: str,
    budget_cents: int,
) -> Any:
    """Create the triage session, idle until send_kickoff, so its place in the daily cap can be
    checked after it exists."""
    lock = json.loads(LOCK_FILE_PATH.read_text())
    return client.beta.sessions.create(
        agent=agent_id,
        environment_id=read_lock_id(lock=lock, lock_key=ENVIRONMENT_LOCK_KEY),
        vault_ids=[vault_id],
        title=f"triage: {repo}#{issue['number']}",
        metadata={"repo": repo, "issue": str(issue["number"]), "lane": "triage"},
        budget={
            "type": "limit",
            "max_list_cost": {"amount": str(budget_cents), "currency": "USD"},
        },
        resources=[
            {
                "type": "github_repository",
                "url": f"https://github.com/{repo}",
                "authorization_token": github_token,
                "checkout": {"type": "branch", "name": "main"},
            }
        ],
    )


def send_kickoff(
    *,
    client: anthropic.Anthropic,
    session_id: str,
    repo: str,
    issue: dict[str, Any],
    report_count: int,
    factory_login: str,
) -> None:
    """Start the session on the issue, graded against rubrics/triage.md."""
    label_names = [label["name"] for label in issue["labels"]]
    # Only the issue number, its author's login and its labels go in the prompt; the agent
    # reads the untrusted title, body and comments itself through the GitHub MCP server.
    rerun_note = (
        f"This is a rerun: the factory, as {factory_login}, has posted {report_count} triage "
        "reports on it. "
        if report_count
        else ""
    )
    kickoff_text = (
        f"Triage GitHub issue #{issue['number']} in {repo}, opened by "
        f"{issue['author']['login']}. Its labels are: {', '.join(label_names) or 'none'}. "
        f"{rerun_note}Read it with issue_read, then submit your triage with {SUBMIT_TOOL_NAME}."
    )
    client.beta.sessions.events.send(
        session_id,
        events=[
            {"type": "user.message", "content": [{"type": "text", "text": kickoff_text}]},
            {
                "type": "user.define_outcome",
                "description": (
                    f"A triage of issue #{issue['number']} that the dispatcher recorded from "
                    f"{SUBMIT_TOOL_NAME}: labels by the issue-triage skill, a reproduction or "
                    "the reason there was none, a confidence level the evidence supports, and "
                    "specific open questions for the reporter when information is missing."
                ),
                "rubric": {
                    "type": "text",
                    "content": (FACTORY_DIR / "rubrics" / "triage.md").read_text(),
                },
                "max_iterations": 2,
            },
        ],
    )


def read_custom_tool_uses(
    *, client: anthropic.Anthropic, session_id: str, event_ids: list[str]
) -> list[Any]:
    """Return the custom tool calls among the events the session is blocked on."""
    pending_event_ids = set(event_ids)
    tool_uses = []
    for event in client.beta.sessions.events.list(
        session_id, types=["agent.custom_tool_use"], order="desc"
    ):
        if event.id in pending_event_ids:
            tool_uses.append(event)
            pending_event_ids.remove(event.id)
            if not pending_event_ids:
                break
    return tool_uses


def expect_confidence_level(*, submission: dict[str, Any]) -> str:
    """Return the confidence level the grading rules in agents/issue-triager.md give."""
    sufficiency = submission["sufficiency"]
    reproduction_status = submission["reproduction"]["status"]
    if (
        any(sufficiency[key] != "answered" for key in SUFFICIENCY_QUESTIONS)
        or reproduction_status == "not_confirmed"
    ):
        return "low"
    if (
        reproduction_status == "confirmed"
        and not submission["open_questions"]
        and not submission["maintainer_questions"]
    ):
        return "high"
    return "medium"


def is_repo_path(*, path: str) -> bool:
    """Whether the path names a file or directory in the checkout, outside `.git`."""
    resolved_path = (REPO_ROOT / path).resolve()
    repo_root = REPO_ROOT.resolve()
    return (
        resolved_path.exists()
        and resolved_path.is_relative_to(repo_root)
        and not resolved_path.is_relative_to(repo_root / ".git")
    )


def is_issue_number(value: Any) -> bool:
    return type(value) is int and value > 0


def find_shape_problems(*, submission: dict[str, Any]) -> list[str]:
    """Return the fields that are missing or of the wrong type, so later checks can read them."""
    problems = [
        f"`{field_name}` must be a {field_type.__name__}."
        for field_name, field_type in SUBMISSION_FIELD_TYPES.items()
        if not isinstance(submission.get(field_name), field_type)
    ]
    if problems:
        return problems
    enum_fields = [
        ("issue_type", submission["issue_type"], (*ISSUE_TYPE_LABELS, "other")),
        ("reproduction.status", submission["reproduction"].get("status"), REPRODUCTION_STATUSES),
        ("confidence.level", submission["confidence"].get("level"), CONFIDENCE_LEVELS),
        ("shipped.status", submission["shipped"].get("status"), SHIPPED_STATUSES),
        *(
            (f"sufficiency.{key}", submission["sufficiency"].get(key), SUFFICIENCY_GRADES)
            for key in SUFFICIENCY_QUESTIONS
        ),
    ]
    problems += [
        f"`{field_name}` must be one of {', '.join(allowed_values)}."
        for field_name, value, allowed_values in enum_fields
        if value not in allowed_values
    ]

    def read_item_values(*, field_name: str, keys: tuple[str, ...]) -> list[Any]:
        return [
            item.get(key) if isinstance(item, dict) else None
            for item in submission[field_name]
            for key in keys
        ]

    string_fields = {
        "labels": submission["labels"],
        "maintainer_questions": submission["maintainer_questions"],
        "open_questions": read_item_values(field_name="open_questions", keys=("question", "why")),
        "code_pointers": read_item_values(field_name="code_pointers", keys=("path", "note")),
        "reproduction": [
            submission["reproduction"].get(key) for key in ("evidence", "command", "test_code")
        ],
        "confidence.reason": [submission["confidence"].get("reason")],
        "shipped.evidence": [submission["shipped"].get("evidence")],
        "duplicate_of.evidence": [submission["duplicate_of"].get("evidence")],
    }
    problems += [
        f"Every value in `{field_name}` must be a string."
        for field_name, values in string_fields.items()
        if not all(isinstance(value, str) for value in values)
    ]
    problems += [
        f"`{field_name}` may hold at most {max_items} items; keep the strongest."
        for field_name, max_items in MAX_LIST_ITEMS.items()
        if len(submission[field_name]) > max_items
    ]
    if not all(is_issue_number(number) for number in submission["possible_duplicates"]):
        problems.append("`possible_duplicates` must hold issue numbers.")
    duplicate_number = submission["duplicate_of"].get("number")
    if not (
        is_issue_number(duplicate_number)
        or (type(duplicate_number) is int and duplicate_number == 0)
    ):
        problems.append("`duplicate_of.number` must be an issue number, or 0 for none.")
    return problems


def find_submission_problems(
    *, submission: dict[str, Any], repo_label_names: set[str]
) -> list[str]:
    """Return what is wrong with a submit_triage call, for the agent to fix; empty if valid.

    The agent's judgment goes into sufficiency, reproduction and the questions. These checks
    hold the rest to the grading rules and the repo, so every report means the same thing.
    """
    problems = find_shape_problems(submission=submission)
    if problems:
        return problems
    issue_type = submission["issue_type"]
    labels = submission["labels"]
    for label in labels:
        if label not in repo_label_names:
            problems.append(
                f"`{label}` is not a label on this repo; leave it out and recommend it in the "
                "summary."
            )
        elif not TRIAGE_LABEL_PATTERN.fullmatch(label):
            problems.append(f"`{label}` is not a label triage sets.")
    type_labels = [label for label in labels if label in ISSUE_TYPE_LABELS]
    if type_labels != ([] if issue_type == "other" else [issue_type]):
        problems.append(
            "`labels` must hold exactly one type label, the one `issue_type` names, or none "
            "for `other`."
        )
    size_labels = [label for label in labels if label.startswith("size:")]
    if len(size_labels) > 1 or (issue_type in ("bug", "enhancement") and not size_labels):
        problems.append(
            "Give a bug or an enhancement exactly one of `size:S`, `size:M` and `size:L`, and "
            "any other issue at most one."
        )
    unanswered = [
        question
        for key, question in SUFFICIENCY_QUESTIONS.items()
        if submission["sufficiency"][key] != "answered"
    ]
    if unanswered and not submission["open_questions"]:
        problems.append(
            f"{' and '.join(unanswered)} isn't answered, so ask the reporter for what is missing "
            "in `open_questions`."
        )
    reproduction = submission["reproduction"]
    if reproduction["status"] != "not_attempted" and not reproduction["command"].strip():
        problems.append("Give the `reproduction.command` that ran the test or probe.")
    if reproduction["status"] == "confirmed" and not reproduction["test_code"].strip():
        problems.append("Give the failing test or probe in `reproduction.test_code`.")
    expected_level = expect_confidence_level(submission=submission)
    if submission["confidence"]["level"] != expected_level:
        problems.append(
            f"By the grading rules, this is `{expected_level}` confidence: `high` needs all three "
            "sufficiency questions answered, a confirmed reproduction and nothing left to ask; "
            "`low` means a question isn't answered or the reproduction is not_confirmed. Fix "
            "the level and its reason, or the grades that led to it."
        )
    if submission["confidence"]["level"] == "low" and not (
        submission["open_questions"] or submission["maintainer_questions"]
    ):
        problems.append("Low confidence must say what is missing, as open or maintainer questions.")
    for code_pointer in submission["code_pointers"]:
        path = LINE_SUFFIX_PATTERN.sub("", code_pointer["path"]).lstrip("/")
        if not path or not is_repo_path(path=path):
            problems.append(f"`{code_pointer['path']}` is not a path on `main`.")
    if submission["shipped"]["status"] != "no" and not submission["shipped"]["evidence"].strip():
        problems.append("Name the file and line or CHANGELOG entry in `shipped.evidence`.")
    duplicate_of = submission["duplicate_of"]
    if duplicate_of["number"]:
        if submission["sufficiency"]["current_behavior"] != "answered":
            problems.append(
                "A duplicate reports the same behavior, so `duplicate_of` needs "
                "`sufficiency.current_behavior` answered; otherwise list the issue in "
                "`possible_duplicates`."
            )
        if submission["open_questions"] or submission["maintainer_questions"]:
            problems.append(
                "The dispatcher closes a `duplicate_of` issue, so nothing may be left to ask; with "
                "questions open, list the issue in `possible_duplicates` instead."
            )
        if not duplicate_of["evidence"].strip():
            problems.append("Quote what both issues report in `duplicate_of.evidence`.")
    return problems


def read_duplicate_target(*, issue_number: int) -> Optional[dict[str, Any]]:
    """Return the issue a submission names as the duplicate, or None if the repo has none."""
    try:
        output = run_gh(
            "api",
            f"repos/{{owner}}/{{repo}}/issues/{issue_number}",
            "--jq",
            "{number, state, is_pull_request: (.pull_request != null)}",
        )
    except subprocess.CalledProcessError:
        return None
    return json.loads(output)


def is_issue(*, target: Optional[dict[str, Any]]) -> bool:
    return target is not None and not target["is_pull_request"]


def find_duplicate_problem(
    *, issue_number: int, duplicate_number: int, target: Optional[dict[str, Any]]
) -> Optional[str]:
    """Return why this issue can't close as a duplicate of the other, or None if it can.

    The older open issue is the one to keep, so its discussion and any fix stay in one place.
    """
    if duplicate_number == issue_number:
        return "`duplicate_of` names this issue"
    if target is None or target["number"] != duplicate_number:
        return f"#{duplicate_number} is not an issue on this repo"
    if target["is_pull_request"]:
        return f"#{duplicate_number} is a pull request"
    if target["state"] != "open":
        return f"#{duplicate_number} is closed, so this may be a regression"
    if duplicate_number > issue_number:
        return f"#{duplicate_number} is newer than this issue, which is the one to keep"
    return None


def watch_session(
    *,
    client: anthropic.Anthropic,
    session_token: GitHubAppToken,
    session_id: str,
    issue_number: int,
    repo_label_names: set[str],
    watch_minutes: int,
) -> tuple[Optional[dict[str, Any]], str]:
    """Answer the session's submissions until it ends; return the last valid one and why it stopped.

    A valid call is answered Recorded and kept, an invalid one with its problems so the agent
    can fix them. The outcome grader may send the agent back to revise after it ends its turn,
    and a revised call replaces the kept one.
    """
    deadline = time.monotonic() + watch_minutes * 60
    submission: Optional[dict[str, Any]] = None
    invalid_submission_count = 0
    previous_idle_event_id: Optional[str] = None
    while True:
        session, idle_event = wait_for_session(
            client=client,
            app_token=session_token,
            session_id=session_id,
            previous_idle_event_id=previous_idle_event_id,
            deadline=deadline,
        )
        if session is None:
            return submission, f"still running after {watch_minutes} minutes"
        if idle_event is None:
            return submission, "terminated"
        if idle_event.stop_reason.type != "requires_action":
            return submission, idle_event.stop_reason.type
        previous_idle_event_id = idle_event.id
        result_events: list[BetaManagedAgentsUserCustomToolResultEventParams] = []
        for tool_use in read_custom_tool_uses(
            client=client, session_id=session_id, event_ids=idle_event.stop_reason.event_ids
        ):
            problems = (
                find_submission_problems(
                    submission=tool_use.input, repo_label_names=repo_label_names
                )
                if tool_use.name == SUBMIT_TOOL_NAME
                else [f"The dispatcher answers only {SUBMIT_TOOL_NAME}."]
            )
            if not problems:
                problems += [
                    f"#{number} in `possible_duplicates` is not an issue on this repo."
                    for number in tool_use.input["possible_duplicates"]
                    if not is_issue(target=read_duplicate_target(issue_number=number))
                ]
            if not problems and tool_use.input["duplicate_of"]["number"]:
                duplicate_number = tool_use.input["duplicate_of"]["number"]
                duplicate_problem = find_duplicate_problem(
                    issue_number=issue_number,
                    duplicate_number=duplicate_number,
                    target=read_duplicate_target(issue_number=duplicate_number),
                )
                if duplicate_problem is not None:
                    problems.append(
                        f"This can't close as a duplicate: {duplicate_problem}. Correct "
                        "`duplicate_of`, or set its number to 0 and list the issue in "
                        "`possible_duplicates` for a maintainer."
                    )
            if problems:
                invalid_submission_count += 1
                result_text = (
                    f"Not recorded. Fix these and call {SUBMIT_TOOL_NAME} again:\n"
                    + "\n".join(f"- {problem}" for problem in problems)
                )
                if invalid_submission_count == MAX_INVALID_SUBMISSIONS:
                    result_text += "\n\nThe dispatcher answers no more calls. End your turn."
            else:
                submission = tool_use.input
                result_text = (
                    "Recorded. The dispatcher labels the issue and posts this triage when you end "
                    "your turn."
                )
            result_events.append(
                {
                    "type": "user.custom_tool_result",
                    "custom_tool_use_id": tool_use.id,
                    "content": [{"type": "text", "text": result_text}],
                    "is_error": bool(problems),
                }
            )
        if not result_events:
            return submission, "blocked on an event the dispatcher can't answer"
        client.beta.sessions.events.send(session_id, events=result_events)
        if invalid_submission_count >= MAX_INVALID_SUBMISSIONS:
            return submission, f"stopped after {invalid_submission_count} invalid submissions"


def find_close_problem(
    *,
    issue: dict[str, Any],
    submission: dict[str, Any],
    duplicate_target: Optional[dict[str, Any]],
    outcome_result: str,
    open_pr_urls: list[str],
) -> Optional[str]:
    """Return why the issue shouldn't close as a duplicate, or None if it should.

    A wrong close shuts out a real report, so the grader must agree, nobody may be working on
    the issue, and the other issue, read again after the session, must still be open.
    """
    duplicate_problem = find_duplicate_problem(
        issue_number=issue["number"],
        duplicate_number=submission["duplicate_of"]["number"],
        target=duplicate_target,
    )
    if duplicate_problem is not None:
        return duplicate_problem
    if outcome_result != "satisfied":
        return f"the triage didn't meet its rubric (`{outcome_result}`)"
    if issue["assignees"]:
        return (
            f"it is assigned to {', '.join(assignee['login'] for assignee in issue['assignees'])}"
        )
    if open_pr_urls:
        return f"an open PR addresses it: {', '.join(open_pr_urls)}"
    return None


def decide_verdict(*, submission: dict[str, Any], closes_as_duplicate: bool) -> str:
    """Name whose move it is: the reporter's, a maintainer's, or a builder's, or none."""
    if closes_as_duplicate:
        return "duplicate"
    if submission["open_questions"]:
        return "needs-information"
    if (
        submission["maintainer_questions"]
        or submission["duplicate_of"]["number"]
        or submission["possible_duplicates"]
        or submission["shipped"]["status"] == "fully"
        or submission["issue_type"] in ("question", "other")
    ):
        return "needs-maintainer"
    return "ready"


def plan_label_changes(
    *, issue: dict[str, Any], submission: dict[str, Any], verdict: str
) -> tuple[list[str], list[str]]:
    """Return the labels to add and to remove, before the factory-candidate decision.

    Triage removes only a type label another type replaces and a `needs information` that no
    longer holds, so it never undoes what a maintainer set.
    """
    current_label_names = {label["name"] for label in issue["labels"]}
    wanted_label_names = set(submission["labels"])
    if verdict == "needs-information":
        wanted_label_names.add(NEEDS_INFORMATION_LABEL)
    if verdict == "duplicate":
        wanted_label_names.add(DUPLICATE_LABEL)
    removed_label_names = set()
    if submission["issue_type"] != "other":
        removed_label_names |= {
            label
            for label in ISSUE_TYPE_LABELS
            if label != submission["issue_type"] and label in current_label_names
        }
    if verdict != "needs-information" and NEEDS_INFORMATION_LABEL in current_label_names:
        removed_label_names.add(NEEDS_INFORMATION_LABEL)
    return sorted(wanted_label_names - current_label_names), sorted(removed_label_names)


def find_candidate_problem(
    *,
    issue: dict[str, Any],
    submission: dict[str, Any],
    verdict: str,
    label_names: set[str],
    open_pr_urls: list[str],
    outcome_result: str,
) -> Optional[str]:
    """Return why the factory shouldn't build the issue, or None if it is a factory candidate.

    The agent grades the evidence; this decides the nomination from those grades and the
    labels, with the bug lane's own admission rules, so a candidate is one the factory admits.
    """
    issue_type = submission["issue_type"]
    if verdict == "duplicate":
        return f"it duplicates #{submission['duplicate_of']['number']}"
    if verdict != "ready":
        return "it isn't ready to work on"
    if issue_type not in ("bug", "enhancement"):
        return f"the factory builds bugs and enhancements, not `{issue_type}` issues"
    if submission["reproduction"]["status"] != "confirmed":
        return (
            "the sandbox didn't reproduce it"
            if issue_type == "bug"
            else "the sandbox didn't show that `main` lacks it"
        )
    if submission["confidence"]["level"] != "high":
        return "confidence isn't high"
    if submission["needs_live_recording"]:
        return (
            "verifying a fix needs a new recording against a live provider, which the sandbox "
            "can't make"
        )
    if outcome_result != "satisfied":
        return f"the triage didn't meet its rubric (`{outcome_result}`)"
    for label_name, reason in NOT_READY_LABELS.items():
        if label_name in label_names:
            return reason
    if not label_names & CANDIDATE_SIZE_LABELS:
        return "it has no size label"
    language_labels = {label for label in label_names if label.startswith("language: ")}
    if not language_labels or not language_labels <= FACTORY_LANGUAGE_LABELS:
        return "the factory takes Python and TypeScript issues only"
    if issue_type == "enhancement" and label_names & {"c/semcov", "new instrumentation"}:
        return (
            "it changes the semantic conventions or adds an instrumentor, which a maintainer "
            "agrees on first"
        )
    if issue["assignees"]:
        return (
            f"it is assigned to {', '.join(assignee['login'] for assignee in issue['assignees'])}"
        )
    if open_pr_urls:
        return f"an open PR already addresses it: {', '.join(open_pr_urls)}"
    return None


def escape_prose(text: str) -> str:
    """Return text that GitHub renders as itself: no markup, link, mention or issue reference."""
    # A word joiner inside `://` and after `www` keeps GitHub from linking a bare URL.
    text = re.sub(r"(?i)\bwww\.", lambda match: match.group(0)[:3] + "\u2060.", text)
    text = text.replace("://", ":\u2060//")
    return "".join(PROSE_ESCAPES.get(character, character) for character in text)


def format_code_span(code: str, *, in_table: bool = False) -> str:
    """Fence one line of code with more backticks than it holds in a row, so it can't close."""
    code = code.replace("@", "@\u2060")
    if in_table:
        code = code.replace("|", "\\|")
    longest_backtick_run = max((len(run) for run in re.findall(r"`+", code)), default=0)
    fence = "`" * (longest_backtick_run + 1)
    padding = " " if code.startswith("`") or code.endswith("`") else ""
    return f"{fence}{padding}{code}{padding}{fence}"


def clean_text(
    text: str, *, max_characters: int = MAX_FIELD_CHARACTERS, in_table: bool = False
) -> str:
    """Return model-written text that is safe to post: one bounded line of plain text and code.

    The agent writes from untrusted issue text, so it could otherwise notify anyone, start a
    workflow, hide the rest of the report in an HTML comment, or put a link in a comment the
    factory signs. Everything outside its code spans is escaped, and the code spans are fenced
    again, so how GitHub would have parsed the original never matters.
    """
    text = " ".join(text.split())
    if len(text) > max_characters:
        text = text[:max_characters].rstrip() + " …"
    parts = []
    position = 0
    for match in CODE_SPAN_PATTERN.finditer(text):
        parts.append(escape_prose(text[position : match.start()]))
        parts.append(format_code_span(match.group(2), in_table=in_table))
        position = match.end()
    parts.append(escape_prose(text[position:]))
    return "".join(parts)


def clean_cell(text: str) -> str:
    return clean_text(text, max_characters=MAX_CELL_CHARACTERS, in_table=True)


def fence_code(code: str) -> str:
    """Fence code with more backticks than it holds in a row, so it can't close the fence.

    GitHub renders fenced code literally, so only a `@claude`, which starts claude.yml from
    anywhere in a comment, is neutralized; a decorator stays as written.
    """
    code = CLAUDE_MENTION_PATTERN.sub("@\u2060", code.strip()[:MAX_CODE_CHARACTERS])
    longest_backtick_run = max((len(run) for run in re.findall(r"`+", code)), default=0)
    fence = "`" * max(3, longest_backtick_run + 1)
    return f"{fence}\n{code}\n{fence}"


def format_label_changes(*, labels_added: list[str], labels_removed: list[str]) -> str:
    changes = []
    if labels_added:
        changes.append("added " + " ".join(f"`{label}`" for label in labels_added))
    if labels_removed:
        changes.append("removed " + " ".join(f"`{label}`" for label in labels_removed))
    return "; ".join(changes) or "unchanged"


def format_reproduction_result(*, submission: dict[str, Any]) -> tuple[str, str]:
    """Return the report's reproduction row: its name for the issue type, and its result."""
    status = submission["reproduction"]["status"]
    if submission["issue_type"] == "enhancement":
        row_name = "Missing on `main`"
        confirmed_result = "Yes, a probe shows it"
    else:
        row_name = "Reproduced"
        confirmed_result = "Yes, a test fails for the reported reason"
    return row_name, {
        "confirmed": confirmed_result,
        "not_confirmed": "No, see the investigation",
        "not_attempted": "Not attempted, see the investigation",
    }[status]


def format_maintainer_items(
    *, submission: dict[str, Any], close_problem: Optional[str]
) -> list[str]:
    """Return what a maintainer must decide: the agent's questions plus what the verdict read."""
    items = [clean_text(question) for question in submission["maintainer_questions"]]
    duplicate_of = submission["duplicate_of"]
    if duplicate_of["number"]:
        items.append(
            f"Whether this duplicates #{duplicate_of['number']}: "
            f"{clean_text(duplicate_of['evidence'])} The factory didn't close it: "
            f"{close_problem or 'it was not checked'}."
        )
    possible_duplicates = [
        number for number in submission["possible_duplicates"] if number != duplicate_of["number"]
    ]
    if possible_duplicates:
        duplicates = ", ".join(f"#{number}" for number in possible_duplicates)
        items.append(f"Whether this duplicates {duplicates}.")
    if submission["shipped"]["status"] == "fully":
        items.append(
            "Whether to close it: it looks shipped already "
            f"({clean_text(submission['shipped']['evidence'])})."
        )
    if not items:
        items.append(
            "How to answer this question."
            if submission["issue_type"] == "question"
            else "How to respond: it isn't a bug, an enhancement or a docs request."
        )
    return items


def format_factory_result(*, issue_type: str, candidate_problem: Optional[str]) -> str:
    if candidate_problem is not None:
        return "Not a candidate: " + candidate_problem.replace("|", "\\|")
    if issue_type == "bug":
        return (
            f"`{CANDIDATE_LABEL}`: a maintainer can add `{FACTORY_LABEL}` to have the factory "
            "open a PR"
        )
    return (
        f"`{CANDIDATE_LABEL}`: the factory has no feature lane yet, so a maintainer or a "
        "contributor builds it for now"
    )


def format_investigation(*, submission: dict[str, Any]) -> list[str]:
    """Return the collapsed details: the grades and the evidence behind the summary table."""
    sufficiency = " · ".join(
        f"**{question}:** {submission['sufficiency'][key]}"
        for key, question in SUFFICIENCY_QUESTIONS.items()
    )
    reproduction = submission["reproduction"]
    lines = [
        "<details><summary>Investigation</summary>",
        "",
        sufficiency,
        "",
        f"**Reproduction:** "
        f"{clean_text(reproduction['evidence'], max_characters=MAX_EVIDENCE_CHARACTERS)}",
    ]
    if reproduction["command"].strip():
        lines += ["", fence_code(reproduction["command"])]
    if reproduction["test_code"].strip():
        lines += [
            "",
            "<details><summary>Test</summary>",
            "",
            fence_code(reproduction["test_code"]),
            "",
            "</details>",
        ]
    if submission["code_pointers"]:
        lines += ["", "**Where to start:**", ""] + [
            f"- {format_code_span(code_pointer['path'])}: {clean_text(code_pointer['note'])}"
            for code_pointer in submission["code_pointers"]
        ]
    shipped = submission["shipped"]
    lines += [
        "",
        f"**Already shipped:** {shipped['status']}"
        + (f", {clean_text(shipped['evidence'])}" if shipped["evidence"].strip() else ""),
        "**Needs a live recording:** " + ("yes" if submission["needs_live_recording"] else "no"),
        "",
        "</details>",
    ]
    return lines


def format_report(
    *,
    issue: dict[str, Any],
    submission: dict[str, Any],
    verdict: str,
    labels_added: list[str],
    labels_removed: list[str],
    candidate_problem: Optional[str],
    close_problem: Optional[str],
    session_id: str,
    report_number: int,
    can_rerun: bool,
) -> str:
    """Render the issue comment from the submission, so every report has the same shape."""
    lines = [f"## Factory triage: {VERDICT_HEADINGS[verdict]}", ""]
    if submission["sensitive_data"].strip():
        lines += [
            "> [!WARNING]",
            f"> This issue looks like it contains {clean_cell(submission['sensitive_data'])}. "
            "If it is real, revoke or rotate it and edit it out of the issue.",
            "",
        ]
    if verdict == "needs-information":
        lines += [
            f"@{issue['author']['login']}, thanks for the report. Before anyone can work on "
            "it, we need to know:",
            "",
        ]
        for question_number, open_question in enumerate(submission["open_questions"], start=1):
            lines += [
                f"{question_number}. {clean_text(open_question['question'])}",
                f"   Why it matters: {clean_text(open_question['why'])}",
            ]
        next_step = (
            "the factory will triage the issue again"
            if can_rerun
            else "a maintainer will pick it up"
        )
        lines += ["", f"Reply here, and {next_step}.", ""]
    if verdict == "duplicate":
        duplicate_of = submission["duplicate_of"]
        lines += [
            f"@{issue['author']['login']}, thanks for the report. It describes the same problem "
            f"as #{duplicate_of['number']}, so the factory closed it as a duplicate to keep the "
            f"discussion and the fix in one place. Add anything #{duplicate_of['number']} "
            "doesn't cover there.",
            "",
            f"**Why it matches:** {clean_text(duplicate_of['evidence'])}",
            "",
            "If this is a different problem, reply here with what sets it apart, and a "
            "maintainer can reopen it.",
            "",
        ]
    lines += [clean_text(submission["summary"], max_characters=MAX_SUMMARY_CHARACTERS), ""]
    if verdict == "needs-maintainer":
        lines += ["A maintainer needs to decide:", ""]
        lines += [
            f"{item_number}. {item}"
            for item_number, item in enumerate(
                format_maintainer_items(submission=submission, close_problem=close_problem),
                start=1,
            )
        ]
        lines.append("")
    confidence = submission["confidence"]
    reproduction_row_name, reproduction_result = format_reproduction_result(submission=submission)
    label_changes = format_label_changes(labels_added=labels_added, labels_removed=labels_removed)
    factory_result = format_factory_result(
        issue_type=submission["issue_type"], candidate_problem=candidate_problem
    )
    lines += [
        "| Check | Result |",
        "|---|---|",
        f"| Type | `{submission['issue_type']}` |",
        f"| {reproduction_row_name} | {reproduction_result} |",
        f"| Enough information | {confidence['level'].capitalize()} confidence: "
        f"{clean_cell(confidence['reason'])} |",
        f"| Labels | {label_changes} |",
        f"| Factory | {factory_result} |",
        "",
    ]
    footer = [
        "",
        f"<sub>Triage report {report_number} · session "
        f"{format_session_link(session_id=session_id)} · "
        f"[How factory triage works]({README_URL})</sub>",
        "",
        f"{REPORT_MARKER} verdict={verdict} -->",
    ]
    report = "\n".join(lines + format_investigation(submission=submission) + footer)
    if len(report) > MAX_COMMENT_CHARACTERS:
        too_long = "The investigation is too long for a comment; the session has it."
        report = "\n".join(lines + [too_long] + footer)
    return report


def post_triage(
    *,
    issue: dict[str, Any],
    submission: dict[str, Any],
    outcome_result: str,
    session_id: str,
    report_number: int,
    can_rerun: bool,
    report_token: GitHubAppToken,
) -> None:
    """Label the issue from the submission, decide `factory-candidate`, and post the report.

    An issue that duplicates an older open one closes as its duplicate before the report is
    posted, so the report never says it closed when it didn't.
    """
    open_pr_urls = list_open_pr_urls(issue=issue)
    duplicate_number = submission["duplicate_of"]["number"]
    close_problem = (
        find_close_problem(
            issue=issue,
            submission=submission,
            duplicate_target=read_duplicate_target(issue_number=duplicate_number),
            outcome_result=outcome_result,
            open_pr_urls=open_pr_urls,
        )
        if duplicate_number
        else None
    )
    closes_as_duplicate = bool(duplicate_number) and close_problem is None
    verdict = decide_verdict(submission=submission, closes_as_duplicate=closes_as_duplicate)
    labels_added, labels_removed = plan_label_changes(
        issue=issue, submission=submission, verdict=verdict
    )
    current_label_names = {label["name"] for label in issue["labels"]}
    candidate_problem = find_candidate_problem(
        issue=issue,
        submission=submission,
        verdict=verdict,
        label_names=(current_label_names - set(labels_removed)) | set(labels_added),
        open_pr_urls=open_pr_urls,
        outcome_result=outcome_result,
    )
    if candidate_problem is None and CANDIDATE_LABEL not in current_label_names:
        labels_added.append(CANDIDATE_LABEL)
    if candidate_problem is not None and CANDIDATE_LABEL in current_label_names:
        labels_removed.append(CANDIDATE_LABEL)
    print(
        f"verdict={verdict} added={labels_added} removed={labels_removed} "
        f"candidate_problem={candidate_problem} close_problem={close_problem}"
    )
    label_args = [
        *(arg for label in labels_added for arg in ("--add-label", label)),
        *(arg for label in labels_removed for arg in ("--remove-label", label)),
    ]
    if label_args:
        run_gh("issue", "edit", str(issue["number"]), *label_args, token=report_token.refresh())
    if closes_as_duplicate:
        run_gh(
            "issue",
            "close",
            str(issue["number"]),
            "--duplicate-of",
            str(duplicate_number),
            token=report_token.refresh(),
        )
    comment_on_issue(
        issue_number=issue["number"],
        body=format_report(
            issue=issue,
            submission=submission,
            verdict=verdict,
            labels_added=labels_added,
            labels_removed=labels_removed,
            candidate_problem=candidate_problem,
            close_problem=close_problem,
            session_id=session_id,
            report_number=report_number,
            can_rerun=can_rerun,
        ),
        token=report_token.refresh(),
    )


def run_start(*, args: argparse.Namespace) -> None:
    issue = read_issue(issue_number=args.issue)
    factory_login = read_factory_login()
    missing_label_names = {
        CANDIDATE_LABEL,
        NEEDS_INFORMATION_LABEL,
        DUPLICATE_LABEL,
    } - read_repo_label_names()
    if missing_label_names:
        raise SystemExit(
            f"Create the {', '.join(sorted(missing_label_names))} label(s) first; see Bootstrap "
            "in factory/README.md."
        )
    report_times, author_comment_time = read_triage_history(
        issue_number=args.issue,
        author_login=issue["author"]["login"],
        factory_login=factory_login,
    )
    client = anthropic.Anthropic()
    agent_id, vault_id = read_triage_ids(client=client, lock=json.loads(LOCK_FILE_PATH.read_text()))
    sessions = list_triage_sessions(client=client, agent_id=agent_id)
    requester_permission = read_repo_permission(login=args.requested_by)
    admission_problem = find_admission_problem(
        issue=issue,
        trigger=args.trigger,
        requester_permission=requester_permission,
        report_times=report_times,
        author_comment_time=author_comment_time,
        issue_session_count=count_issue_sessions(
            sessions=sessions, repo=args.repo, issue_number=args.issue
        ),
        recent_session_count=count_sessions_in_day_before(
            sessions=sessions, moment=datetime.now(timezone.utc)
        ),
    )
    print(
        f"trigger={args.trigger} requested_by={args.requested_by} reports={len(report_times)} "
        f"admission_problem={admission_problem}"
    )
    # Anyone can open an issue, so a skipped triage says why in the job log, not on the issue.
    if args.dry_run or admission_problem is not None:
        write_github_outputs(started="false")
        return
    session_token = create_token(
        client=client, repo=args.repo, permissions=SESSION_TOKEN_PERMISSIONS, vault_id=vault_id
    )
    session = create_session(
        client=client,
        repo=args.repo,
        issue=issue,
        agent_id=agent_id,
        vault_id=vault_id,
        github_token=session_token.refresh(),
        budget_cents=args.budget_cents,
    )
    print(f"session_id={session.id}")
    # Jobs for a burst of issues all pass the count above before any creates a session, so the
    # cap holds only for sessions that existed first: one past it is archived before it starts.
    if requester_permission not in STARTING_PERMISSIONS:
        earlier_sessions = [
            earlier_session
            for earlier_session in list_triage_sessions(client=client, agent_id=agent_id)
            if (earlier_session.created_at, earlier_session.id) < (session.created_at, session.id)
        ]
        earlier_session_count = count_sessions_in_day_before(
            sessions=earlier_sessions, moment=session.created_at
        )
        if earlier_session_count >= MAX_SESSIONS_PER_DAY:
            client.beta.sessions.archive(session.id)
            print(f"Archived it: {earlier_session_count} triage sessions started before it today.")
            write_github_outputs(started="false")
            return
    send_kickoff(
        client=client,
        session_id=session.id,
        repo=args.repo,
        issue=issue,
        report_count=len(report_times),
        factory_login=factory_login,
    )
    write_github_outputs(started="true", session_id=session.id)


def print_untrusted(text: str) -> None:
    """Print text that may quote the issue, so no line of it runs as a workflow command."""
    stop_token = secrets.token_hex(16)
    print(f"::stop-commands::{stop_token}")
    print(text)
    print(f"::{stop_token}::")


def run_watch(*, args: argparse.Namespace) -> None:
    client = anthropic.Anthropic()
    factory_login = read_factory_login()
    agent_id, vault_id = read_triage_ids(client=client, lock=json.loads(LOCK_FILE_PATH.read_text()))
    session_token = create_token(
        client=client, repo=args.repo, permissions=SESSION_TOKEN_PERMISSIONS, vault_id=vault_id
    )
    session_token.follow_session(session_id=args.session_id)
    submission, stop_reason = watch_session(
        client=client,
        session_token=session_token,
        session_id=args.session_id,
        issue_number=args.issue,
        repo_label_names=read_repo_label_names(),
        watch_minutes=args.watch_minutes,
    )
    session = client.beta.sessions.retrieve(args.session_id)
    # A session this job stops following would otherwise run on to its spend cap.
    if session.status == "running":
        client.beta.sessions.events.send(args.session_id, events=[{"type": "user.interrupt"}])
    latest_outcome = session.outcome_evaluations[-1] if session.outcome_evaluations else None
    outcome_result = latest_outcome.result if latest_outcome else "not graded"
    # How the run went is for the factory's maintainers, so it goes to the job log.
    print(f"stop_reason={stop_reason} rubric={outcome_result} cost={format_cost(session=session)}")
    if latest_outcome is not None and outcome_result != "satisfied" and latest_outcome.explanation:
        print("Grader explanation:")
        print_untrusted(latest_outcome.explanation)
    if submission is None:
        raise SystemExit(
            f"::error::Session {args.session_id} ended without a valid {SUBMIT_TOOL_NAME} call "
            f"(stop reason: {stop_reason}), so nothing was posted."
        )
    # Labels and assignees may have changed while the session ran.
    issue = read_issue(issue_number=args.issue)
    label_names = {label["name"] for label in issue["labels"]}
    if issue["state"] != "OPEN" or label_names & {"agent-in-progress", FACTORY_LABEL}:
        print("The issue was closed or an agent took it during triage; nothing was posted.")
        return
    report_times, _ = read_triage_history(
        issue_number=args.issue, author_login=issue["author"]["login"], factory_login=factory_login
    )
    issue_session_count = count_issue_sessions(
        sessions=list_triage_sessions(client=client, agent_id=agent_id),
        repo=args.repo,
        issue_number=args.issue,
    )
    post_triage(
        issue=issue,
        submission=submission,
        outcome_result=outcome_result,
        session_id=args.session_id,
        report_number=len(report_times) + 1,
        can_rerun=issue_session_count < MAX_SESSIONS_PER_ISSUE,
        report_token=create_token(
            client=client, repo=args.repo, permissions=REPORT_TOKEN_PERMISSIONS, vault_id=None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="owner/name, e.g. Arize-ai/openinference")
    parser.add_argument("--issue", type=int, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Check the issue and start its triage")
    start_parser.add_argument(
        "--requested-by", required=True, help="Login of whoever opened, labeled or dispatched it"
    )
    start_parser.add_argument(
        "--trigger",
        choices=["label", "reply", "dispatch"],
        required=True,
        help="label: `triage` was added; reply: the author answered; dispatch: by hand",
    )
    start_parser.add_argument(
        "--budget-cents", type=int, default=1000, help="Session spend cap, USD cents"
    )
    start_parser.add_argument("--dry-run", action="store_true", help="Print the checks and exit")

    watch_parser = subparsers.add_parser("watch", help="Answer the session and post its triage")
    watch_parser.add_argument("--session-id", required=True)
    watch_parser.add_argument("--watch-minutes", type=int, default=75)

    args = parser.parse_args()
    if args.command == "start":
        run_start(args=args)
    else:
        run_watch(args=args)


if __name__ == "__main__":
    main()
