#!/usr/bin/env bash
# Render the spans of a Phoenix project as a tree, or list attribute keys, values, or errors.
#
# Usage:
#   span_tree.sh <project-or-spans.json> [tree|keys|values|errors] [-- <extra px span list args>]
#
# <project-or-spans.json> is a Phoenix project name, or a path to a JSON file saved with
# `px span list --project <p> --format raw --no-progress > <p>.spans.json`.
#
# Modes:
#   tree    name [KIND] status, nested by parent (default)
#   keys    sorted attribute keys per span
#   values  sorted key=value per span, minus volatile keys (output.value, llm.output_messages.*, llm.token_count.*, llm.finish_reason)
#   errors  ERROR spans with status_message and exception.message
#
# Examples:
#   span_tree.sh openai-chat
#   span_tree.sh openai-chat keys
#   span_tree.sh openai-chat errors -- --last-n-minutes 5
#   span_tree.sh openai-chat tree -- --limit 500 --trace-id <id> --endpoint http://localhost:6006
#
# Exit codes: 0 ok, 1 no spans / px failure, 2 usage.
# A span count is printed to stderr in every mode so it never pollutes a diff.
# Requires: jq, and px on PATH (falls back to `npx @arizeai/phoenix-cli`). Honors PHOENIX_HOST.
set -euo pipefail

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

[[ $# -ge 1 ]] || usage
project=$1; shift
mode=tree
if [[ $# -ge 1 && $1 != "--" ]]; then mode=$1; shift; fi
[[ ${1:-} == "--" ]] && shift
case $mode in tree|keys|values|errors) ;; *) echo "unknown mode: $mode" >&2; usage ;; esac

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

if [[ -f $project ]]; then
  spans=$(cat "$project")
else
  if command -v px >/dev/null; then px=(px); else px=(npx --yes @arizeai/phoenix-cli); fi
  err=$(mktemp)
  if ! spans=$("${px[@]}" span list --project "$project" --format raw --no-progress "$@" 2>"$err"); then
    msg=$(cat "$err"); rm -f "$err"
    if grep -qiE 'fetch failed|ECONNREFUSED|unreachable|timed out' <<<"$msg"; then
      echo "Phoenix unreachable at ${PHOENIX_HOST:-http://localhost:6006} (or --endpoint). Start it or fix PHOENIX_HOST. px said: $msg" >&2
    elif grep -qiE '404|not found|resolve project' <<<"$msg"; then
      echo "project '$project' does not exist in Phoenix (${PHOENIX_HOST:-http://localhost:6006}): no spans were ever received. Did the example export, and is the project name right?" >&2
    else
      echo "px failed: $msg" >&2
    fi
    exit 1
  fi
  rm -f "$err"
fi

if ! count=$(jq length <<<"$spans" 2>/dev/null); then
  echo "px did not return JSON. Output was: ${spans:0:300}" >&2; exit 1
fi
if [[ $count == 0 ]]; then
  echo "project '$project' exists but has no spans (did the example flush before exit?)" >&2
  exit 1
fi

case $mode in
  tree)
    jq -r '
      . as $spans
      | ($spans | map(.context.span_id)) as $ids
      | def indent($d): [range($d)] | map("  ") | join("");
        def line($s; $d):
          indent($d) + $s.name + " [" + ($s.span_kind // "?") + "] " + ($s.status_code // "UNSET")
          + (if ($s.status_message // "") != "" then "  -- " + $s.status_message else "" end);
        def kids($pid): [$spans[] | select(.parent_id == $pid)] | sort_by(.start_time) | .[];
        def render($s; $d): line($s; $d), (kids($s.context.span_id) | render(.; $d + 1));
        [$spans[] | .parent_id as $p | select($p == null or (($ids | index($p)) == null))]
        | sort_by(.start_time) | .[] | render(.; 0)
    ' <<<"$spans"
    ;;
  keys)
    jq -r '
      sort_by(.start_time) | .[]
      | "\(.name) [\(.span_kind // "?")]", ((.attributes // {}) | keys[] | "    " + .)
    ' <<<"$spans"
    ;;
  values)
    jq -r '
      def volatile: test("^(output\\.value|llm\\.output_messages\\..*|llm\\.token_count\\..*|llm\\.finish_reason)$");
      sort_by(.start_time) | .[]
      | "\(.name) [\(.span_kind // "?")] \(.status_code // "UNSET")",
        ((.attributes // {}) | to_entries | sort_by(.key) | .[] | select(.key | volatile | not)
          | "    \(.key)=\(.value | tojson | .[0:200])")
    ' <<<"$spans"
    ;;
  errors)
    jq -r --argjson total "$count" '
      [.[] | select(.status_code == "ERROR")]
      | if length == 0 then "no ERROR spans (\($total) checked)" else
        .[] | "\(.name) [\(.span_kind // "?")]\n    status_message: \(.status_message // "")\n    exception.message: \(.attributes["exception.message"] // "")"
        end
    ' <<<"$spans"
    ;;
esac
echo "-- $count span(s) in '$project'" >&2
