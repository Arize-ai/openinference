#!/usr/bin/env bash
#
# Unit test for the alert-slicing logic in
# .github/workflows/claude-dependabot-security.yml
#
# The workflow embeds its selection/slug/dedup logic in `run:` blocks that can't
# be exercised without a full GitHub Actions run. This script mirrors that exact
# logic against synthetic fixtures so the riskiest part (which manifest gets
# picked, branch-slug uniqueness, dedup prefix matching) can be tested locally
# and in CI. Keep it in sync when editing the workflow.
#
# Usage: scripts/test-dependabot-slice.sh
set -euo pipefail

sha1() { if command -v sha1sum >/dev/null 2>&1; then sha1sum; else shasum; fi; }

# --- logic under test (mirrors the workflow) ---------------------------------

# Pick the manifest the workflow would target: highest severity (then lowest
# alert number) among alerts that HAVE a patched version.
select_manifest() {
  jq -r '
    def rank: {"critical":0,"high":1,"medium":2,"low":3}[.] // 4;
    [ .[] | select(.first_patched_version != null) ]
    | sort_by((.severity | rank), .number)
    | (.[0].manifest_path // "")
  ' "$1"
}

# Branch-/dedup-safe slug: distinguishing path tail + content hash.
make_slug() {
  local manifest="$1" path_hash path_tail
  path_hash=$(printf '%s' "$manifest" | sha1 | cut -c1-8)
  path_tail=$(printf '%s' "$manifest" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//' \
    | tail -c 40 \
    | sed -E 's/^-+//')
  printf '%s-%s' "$path_tail" "$path_hash"
}

# --- tiny assert harness -----------------------------------------------------

PASS=0 FAIL=0
check() { # check <description> <actual> <expected>
  if [ "$2" = "$3" ]; then PASS=$((PASS+1)); echo "ok   - $1";
  else FAIL=$((FAIL+1)); echo "FAIL - $1"; echo "        expected: $3"; echo "        actual:   $2"; fi
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# --- fixtures ----------------------------------------------------------------

# A mixed queue: an unpatchable critical (must be skipped for selection), a
# patchable critical in a lockfile, and lower-severity pip alerts.
cat > "$TMP/alerts.json" <<'JSON'
[
  {"number": 672, "severity": "critical", "ecosystem": "pip", "package": "guardrails-ai",
   "manifest_path": "python/instrumentation/openinference-instrumentation-guardrails/test-requirements.txt",
   "first_patched_version": null},
  {"number": 681, "severity": "critical", "ecosystem": "npm", "package": "vitest",
   "manifest_path": "js/pnpm-lock.yaml", "first_patched_version": "4.1.0"},
  {"number": 669, "severity": "medium", "ecosystem": "npm", "package": "ws",
   "manifest_path": "js/pnpm-lock.yaml", "first_patched_version": "8.20.1"},
  {"number": 509, "severity": "high", "ecosystem": "pip", "package": "pydantic-ai",
   "manifest_path": "python/instrumentation/openinference-instrumentation-pydantic-ai/test-requirements.txt",
   "first_patched_version": "1.56.0"}
]
JSON

# Queue where the ONLY alert has no patch -> nothing auto-fixable.
cat > "$TMP/nopatch.json" <<'JSON'
[
  {"number": 1, "severity": "critical", "ecosystem": "pip", "package": "x",
   "manifest_path": "a/b.txt", "first_patched_version": null}
]
JSON

# --- tests -------------------------------------------------------------------

# 1. Selection skips the unpatchable critical and picks the patchable critical.
target=$(select_manifest "$TMP/alerts.json")
check "selection skips no-patch critical, picks patchable critical" \
  "$target" "js/pnpm-lock.yaml"

# 2. Slice includes every alert in the chosen manifest.
count=$(jq --arg m "$target" '[.[] | select(.manifest_path==$m)] | length' "$TMP/alerts.json")
check "slice pulls all alerts for the chosen manifest" "$count" "2"

# 3. No auto-fixable alerts -> empty target (workflow then exits cleanly).
check "all-unpatchable queue yields empty target" \
  "$(select_manifest "$TMP/nopatch.json")" ""

# 4. Two files in the SAME package differ only by suffix -> distinct slugs.
s1=$(make_slug "python/instrumentation/openinference-instrumentation-beeai/pyproject.toml")
s2=$(make_slug "python/instrumentation/openinference-instrumentation-beeai/test-requirements.txt")
if [ "$s1" != "$s2" ]; then PASS=$((PASS+1)); echo "ok   - same-package manifests get distinct slugs";
else FAIL=$((FAIL+1)); echo "FAIL - same-package manifests collided: $s1"; fi

# 5. Slug is deterministic (dedup across days depends on this).
check "slug is deterministic" \
  "$(make_slug js/pnpm-lock.yaml)" "$(make_slug js/pnpm-lock.yaml)"

# 6. Dedup prefix matches the same slice and not a different one.
slug=$(make_slug js/pnpm-lock.yaml)
SLICE_PREFIX="claude/dependabot-security-${slug}--"
export SLICE_PREFIX
prs=$(printf '[{"url":"u1","headRefName":"%s999-1"},{"url":"u2","headRefName":"claude/dependabot-security-other--5-1"}]' "$SLICE_PREFIX")
hit=$(printf '%s' "$prs" | jq -r '[.[] | select(.headRefName | startswith(env.SLICE_PREFIX))][0].url // empty')
check "dedup prefix matches same slice only" "$hit" "u1"

# --- summary -----------------------------------------------------------------
echo
echo "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
