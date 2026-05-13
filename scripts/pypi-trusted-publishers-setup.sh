#!/usr/bin/env bash
# Add GitHub Actions trusted publishing to each OpenInference Python project on PyPI.
#
# Requires: agent-browser (https://github.com/stevekinney/agent-browser), jq
#
# What it does:
#   - Reads Python package names from release-please-config.json.
#   - Opens each existing PyPI project's Publishing settings page.
#   - In default mode, fills and submits PyPI's GitHub trusted publisher form.
#   - In REMOVE_EXISTING=1 mode, removes matching GitHub publishers only and
#     does not add a replacement.
#
# Safety notes:
#   - This automates the current PyPI web UI, so review the output and spot-check
#     project Publishing pages after a run.
#   - REMOVE_EXISTING=1 is destructive. It removes only GitHub publisher rows
#     matching the configured owner, repository, workflow, and environment. Test
#     it with LIMIT_PACKAGE first.
#   - PyPI trusted publisher settings must match .github/workflows/publish.yaml:
#     owner, repository, workflow filename, and optional GitHub environment.
#
# Usage:
#   1) Log in once (opens a headed browser — complete MFA in the window):
#        ./scripts/pypi-trusted-publishers-setup.sh --login
#   2) Register publishers (reuses the same named session / cookies):
#        ./scripts/pypi-trusted-publishers-setup.sh
#   3) Test one project:
#        LIMIT_PACKAGE=openinference-instrumentation-google-genai ./scripts/pypi-trusted-publishers-setup.sh
#   4) Remove existing publishers only (does not re-add):
#        REMOVE_EXISTING=1 LIMIT_PACKAGE=openinference-instrumentation-google-genai ./scripts/pypi-trusted-publishers-setup.sh
#   5) Skip known packages:
#        EXCLUDE_PACKAGES="pkg-one,pkg-two" ./scripts/pypi-trusted-publishers-setup.sh
#
# PyPI pre-filled publishing URLs:
#   https://docs.pypi.org/trusted-publishers/adding-a-publisher/
#
# The script includes PyPI's pre-fill query parameters in the project URL, but
# it also fills #github-publisher-form directly using the current PyPI form field
# names. This keeps submission scoped to the GitHub publisher form.
#
# Environment variables:
#   AGENT_BROWSER_SESSION                 Named browser session to reuse.
#   LIMIT_PACKAGE                         Process one PyPI project only.
#   EXCLUDE_PACKAGES                      Comma- or space-separated PyPI projects to skip.
#   REMOVE_EXISTING                       Remove existing publishers only.
#   AGENT_BROWSER_WAIT_TIMEOUT_MS         Browser wait timeout in milliseconds.
#   PACKAGE_SLEEP_SECONDS                 Delay between projects.
#   GITHUB_TRUSTED_PUBLISHER_OWNER        GitHub repository owner for PyPI.
#   GITHUB_TRUSTED_PUBLISHER_REPO         GitHub repository name for PyPI.
#   GITHUB_TRUSTED_PUBLISHER_WORKFLOW     Workflow filename for PyPI.
#   GITHUB_TRUSTED_PUBLISHER_ENVIRONMENT  Optional GitHub environment for PyPI.
#
# Session: PyPI login is tied to agent-browser's named session (cookies in that browser
# context). Daemon control files live under ~/.agent-browser/ unless AGENT_BROWSER_SOCKET_DIR is set.
# Optional JSON snapshots of storage may be written under ~/.agent-browser/sessions/
# when agent-browser saves/restores state (e.g. on close or via state save/load).
#
# Missing or inaccessible project pages are recorded as failures, but the script
# keeps processing the remaining packages and exits non-zero at the end.
#
set -euo pipefail

SESSION_NAME="${AGENT_BROWSER_SESSION:-pypi-trusted-publishers}"
LIMIT_PACKAGE="${LIMIT_PACKAGE:-}"
EXCLUDE_PACKAGES="${EXCLUDE_PACKAGES:-}"
REMOVE_EXISTING="${REMOVE_EXISTING:-}"
AGENT_BROWSER_WAIT_TIMEOUT_MS="${AGENT_BROWSER_WAIT_TIMEOUT_MS:-5000}"
PACKAGE_SLEEP_SECONDS="${PACKAGE_SLEEP_SECONDS:-1}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Must match .github/workflows/publish.yaml and the GitHub repository.
# Optional: if the publish-python job uses `environment: <name>`, set the same
# name here (and in PyPI). Leave unset only when the workflow job has no
# top-level `environment:` key — mismatches will break OIDC publishing.
GITHUB_OWNER="${GITHUB_TRUSTED_PUBLISHER_OWNER:-Arize-ai}"
GITHUB_REPO="${GITHUB_TRUSTED_PUBLISHER_REPO:-openinference}"
WORKFLOW_FILE="${GITHUB_TRUSTED_PUBLISHER_WORKFLOW:-publish.yaml}"
GITHUB_ENV_NAME="${GITHUB_TRUSTED_PUBLISHER_ENVIRONMENT:-}"

if ! command -v agent-browser >/dev/null 2>&1; then
  echo "agent-browser not found in PATH" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq not found in PATH" >&2
  exit 1
fi

if [[ "${1:-}" == "--login" ]]; then
  echo "Opening PyPI login in headed mode (session: ${SESSION_NAME})."
  echo "Sign in, complete any MFA, then run this script again without --login."
  exec agent-browser --session-name "$SESSION_NAME" --headed open "https://pypi.org/account/login/"
fi

urlencode() {
  jq -nr --arg value "$1" '$value|@uri'
}

PYPI_QUERY="provider=github"
PYPI_QUERY+="&owner=$(urlencode "$GITHUB_OWNER")"
PYPI_QUERY+="&repository=$(urlencode "$GITHUB_REPO")"
PYPI_QUERY+="&workflow_filename=$(urlencode "$WORKFLOW_FILE")"
if [[ -n "$GITHUB_ENV_NAME" ]]; then
  PYPI_QUERY+="&environment=$(urlencode "$GITHUB_ENV_NAME")"
fi

wait_for_page() {
  local pkg="$1"
  local action="$2"

  if ! AGENT_BROWSER_DEFAULT_TIMEOUT="$AGENT_BROWSER_WAIT_TIMEOUT_MS" \
    agent-browser --session-name "$SESSION_NAME" wait --load domcontentloaded >/dev/null 2>&1; then
    echo "Warning: timed out waiting for PyPI page after ${action} for ${pkg}; continuing with current URL check." >&2
  fi
}

count_matching_publishers() {
  local publisher_values="$1"

  AGENT_BROWSER_DEFAULT_TIMEOUT="$AGENT_BROWSER_WAIT_TIMEOUT_MS" \
    agent-browser --session-name "$SESSION_NAME" eval --stdin <<EVALEOF
(() => {
  const expected = ${publisher_values};
  const repoSlug = \`\${expected.owner}/\${expected.repository}\`;
  const matchesExpectedPublisher = (row) => {
    const cells = row.querySelectorAll("td");
    if (cells.length < 3 || cells[0].textContent.trim() !== "GitHub") {
      return false;
    }
    const details = cells[1].textContent.replace(/\s+/g, " ").trim();
    const repoMatches = Array.from(cells[1].querySelectorAll("a"))
      .some((link) => link.textContent.trim() === repoSlug);
    const workflowMatches = details.includes(\`Workflow: \${expected.workflow_filename}\`);
    const environmentMatches = expected.environment
      ? details.includes(\`Environment name: \${expected.environment}\`)
      : details.includes("Environment name: (Any)");
    return repoMatches && workflowMatches && environmentMatches;
  };

  return Array.from(document.querySelectorAll("table.table--publisher-list tbody tr"))
    .filter(matchesExpectedPublisher).length;
})()
EVALEOF
}

remove_existing_publishers() {
  local pkg="$1"
  local removed=0
  local max_removals=20
  local matching_publisher_count
  local publisher_values

  publisher_values="$(
    jq -nc \
      --arg owner "$GITHUB_OWNER" \
      --arg repository "$GITHUB_REPO" \
      --arg workflow_filename "$WORKFLOW_FILE" \
      --arg environment "$GITHUB_ENV_NAME" \
      '{owner: $owner, repository: $repository, workflow_filename: $workflow_filename, environment: $environment}'
  )"

  while ((removed < max_removals)); do
    matching_publisher_count="$(count_matching_publishers "$publisher_values" 2>/dev/null || true)"
    if [[ ! "$matching_publisher_count" =~ ^[0-9]+$ ]]; then
      wait_for_page "$pkg" "checking publisher removal result"
      matching_publisher_count="$(count_matching_publishers "$publisher_values" 2>/dev/null || true)"
    fi
    if [[ ! "$matching_publisher_count" =~ ^[0-9]+$ ]]; then
      echo "Could not check existing publishers for ${pkg}; page may still be navigating." >&2
      return 1
    fi
    if ((matching_publisher_count == 0)); then
      if ((removed == 0)); then
        echo "No matching existing publishers found for ${pkg}"
      else
        echo "Removed ${removed} matching existing publisher(s) for ${pkg}"
      fi
      return 0
    fi

    if ! AGENT_BROWSER_DEFAULT_TIMEOUT="$AGENT_BROWSER_WAIT_TIMEOUT_MS" \
      agent-browser --session-name "$SESSION_NAME" eval --stdin >/dev/null 2>&1 <<EVALEOF
(() => {
  const expected = ${publisher_values};
  const repoSlug = \`\${expected.owner}/\${expected.repository}\`;
  const matchesExpectedPublisher = (row) => {
    const cells = row.querySelectorAll("td");
    if (cells.length < 3 || cells[0].textContent.trim() !== "GitHub") {
      return false;
    }
    const details = cells[1].textContent.replace(/\s+/g, " ").trim();
    const repoMatches = Array.from(cells[1].querySelectorAll("a"))
      .some((link) => link.textContent.trim() === repoSlug);
    const workflowMatches = details.includes(\`Workflow: \${expected.workflow_filename}\`);
    const environmentMatches = expected.environment
      ? details.includes(\`Environment name: \${expected.environment}\`)
      : details.includes("Environment name: (Any)");
    return repoMatches && workflowMatches && environmentMatches;
  };

  const row = Array.from(document.querySelectorAll("table.table--publisher-list tbody tr"))
    .find(matchesExpectedPublisher);
  if (!row) {
    throw new Error("Matching GitHub publisher row not found");
  }

  const removeLink = row.querySelector('a.button--danger[href^="#remove-publisher-"]');
  if (!removeLink || !removeLink.hash) {
    throw new Error("Matching GitHub publisher remove link not found");
  }

  const modal = document.getElementById(removeLink.hash.slice(1));
  const sourceForm = modal && modal.querySelector("form");
  const csrf = modal && modal.querySelector('input[name="csrf_token"]');
  const publisherId = modal && modal.querySelector('input[name="publisher_id"]');
  if (!modal || !sourceForm || !csrf || !publisherId) {
    throw new Error("Matching GitHub publisher removal inputs not found");
  }

  // PyPI's modal markup sits inside a table body, so browsers repair the parsed
  // DOM by closing the form before its hidden inputs. Build an equivalent form
  // explicitly instead of relying on the repaired modal form structure.
  const form = document.createElement("form");
  form.method = "POST";
  form.action = sourceForm.action;
  for (const source of [csrf, publisherId]) {
    const input = document.createElement("input");
    input.type = "hidden";
    input.name = source.name;
    input.value = source.value;
    form.appendChild(input);
  }
  document.body.appendChild(form);
  HTMLFormElement.prototype.submit.call(form);
})()
EVALEOF
    then
      echo "Found a matching publisher for ${pkg}, but could not submit its removal form." >&2
      return 1
    fi

    ((removed += 1))
    echo "Removed matching publisher ${removed} for ${pkg}"
    wait_for_page "$pkg" "removing publisher"
  done

  echo "Stopped after removing ${max_removals} matching publishers for ${pkg}; refusing to continue." >&2
  return 1
}

submit_github_publisher() {
  local pkg="$1"
  local form_values

  form_values="$(
    jq -nc \
      --arg owner "$GITHUB_OWNER" \
      --arg repository "$GITHUB_REPO" \
      --arg workflow_filename "$WORKFLOW_FILE" \
      --arg environment "$GITHUB_ENV_NAME" \
      '{owner: $owner, repository: $repository, workflow_filename: $workflow_filename, environment: $environment}'
  )"

  if ! AGENT_BROWSER_DEFAULT_TIMEOUT="$AGENT_BROWSER_WAIT_TIMEOUT_MS" \
    agent-browser --session-name "$SESSION_NAME" eval --stdin >/dev/null 2>&1 <<EVALEOF
(() => {
  const values = ${form_values};
  const form = document.querySelector("#github-publisher-form");
  if (!form) {
    throw new Error("GitHub publisher form not found");
  }

  for (const [name, value] of Object.entries(values)) {
    const field = form.elements.namedItem(name);
    if (!field) {
      throw new Error(\`GitHub publisher field not found: \${name}\`);
    }
    field.value = value;
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
  }

  const submit = form.querySelector('input[type="submit"][value="Add"], button[type="submit"]');
  if (!submit) {
    throw new Error("GitHub publisher submit button not found");
  }

  if (form.requestSubmit) {
    form.requestSubmit(submit);
  } else {
    submit.click();
  }
})()
EVALEOF
  then
    echo "Could not submit GitHub publisher form for ${pkg}" >&2
    return 1
  fi

  wait_for_page "$pkg" "submitting GitHub publisher form"
  echo "Submitted GitHub publisher form for ${pkg}"
}

PACKAGES=()
while IFS= read -r line; do
  [[ -n "$line" ]] && PACKAGES+=("$line")
done < <(
  jq -r '.packages | to_entries[] | select(.key | startswith("python/")) | .value["package-name"] | sub("^python-"; "")' \
    "$REPO_ROOT/release-please-config.json" | sort
)

if [[ -n "$LIMIT_PACKAGE" ]]; then
  FILTERED_PACKAGES=()
  for pkg in "${PACKAGES[@]}"; do
    if [[ "$pkg" == "$LIMIT_PACKAGE" ]]; then
      FILTERED_PACKAGES+=("$pkg")
    fi
  done
  if ((${#FILTERED_PACKAGES[@]} == 0)); then
    echo "No Python package named '${LIMIT_PACKAGE}' found in release-please-config.json." >&2
    exit 1
  fi
  PACKAGES=("${FILTERED_PACKAGES[@]}")
fi

EXCLUDED_PACKAGES=()
if [[ -n "$EXCLUDE_PACKAGES" ]]; then
  EXCLUDE_PACKAGE_SET=" ${EXCLUDE_PACKAGES//,/ } "
  FILTERED_PACKAGES=()
  for pkg in "${PACKAGES[@]}"; do
    if [[ "$EXCLUDE_PACKAGE_SET" == *" $pkg "* ]]; then
      EXCLUDED_PACKAGES+=("$pkg")
    else
      FILTERED_PACKAGES+=("$pkg")
    fi
  done
  if ((${#FILTERED_PACKAGES[@]} == 0)); then
    PACKAGES=()
  else
    PACKAGES=("${FILTERED_PACKAGES[@]}")
  fi
fi

echo "Using trusted publisher: ${GITHUB_OWNER}/${GITHUB_REPO} workflow ${WORKFLOW_FILE}"
if [[ -n "$GITHUB_ENV_NAME" ]]; then
  echo "GitHub environment (must match publish-python job): ${GITHUB_ENV_NAME}"
else
  echo "GitHub environment: (none — job should not use environment:)"
fi
if [[ -n "$REMOVE_EXISTING" ]]; then
  echo "Mode: remove existing publishers only (will not add publishers)"
fi
if ((${#EXCLUDED_PACKAGES[@]} > 0)); then
  echo "Excluded projects: ${#EXCLUDED_PACKAGES[@]}"
  printf '  - %s\n' "${EXCLUDED_PACKAGES[@]}"
fi
echo "Projects: ${#PACKAGES[@]} (PyPI account needs 2FA enabled to add publishers)"
echo ""

if ((${#PACKAGES[@]} == 0)); then
  echo "No projects to process after filters."
  exit 0
fi

FAILED_PACKAGES=()

for i in "${!PACKAGES[@]}"; do
  pkg="${PACKAGES[$i]}"
  url="https://pypi.org/manage/project/${pkg}/settings/publishing/?${PYPI_QUERY}"
  echo "=== ${pkg} ==="
  echo "Opening publishing page for ${pkg}"
  agent-browser --session-name "$SESSION_NAME" open "$url" >/dev/null
  wait_for_page "$pkg" "opening publishing page"
  current="$(agent-browser --session-name "$SESSION_NAME" get url)"
  if [[ "$current" != *"/manage/project/${pkg}/"* ]]; then
    echo "Not on project publishing page (got: $current). Run with --login first." >&2
    FAILED_PACKAGES+=("$pkg")
  elif [[ -n "$REMOVE_EXISTING" ]]; then
    if ! remove_existing_publishers "$pkg"; then
      FAILED_PACKAGES+=("$pkg")
    fi
  else
    if ! submit_github_publisher "$pkg"; then
      FAILED_PACKAGES+=("$pkg")
    fi
  fi

  echo ""
  if ((i < ${#PACKAGES[@]} - 1)) && [[ "$PACKAGE_SLEEP_SECONDS" != "0" ]]; then
    echo "Sleeping ${PACKAGE_SLEEP_SECONDS}s before next project"
    sleep "$PACKAGE_SLEEP_SECONDS"
  fi
done

if ((${#FAILED_PACKAGES[@]} > 0)); then
  echo "Failed to process trusted publisher for ${#FAILED_PACKAGES[@]} project(s):" >&2
  printf '  - %s\n' "${FAILED_PACKAGES[@]}" >&2
  echo "Review each failed project's Publishing page for errors, duplicate publishers, or UI changes." >&2
  exit 1
fi

if [[ -n "$REMOVE_EXISTING" ]]; then
  echo "Done. Existing publishers were removed where present; no publishers were added."
else
  echo "Done. Review each project's Publishing page for errors or duplicate publishers."
fi
