#!/usr/bin/env bash
set -euo pipefail

files=()
for f in "$@"; do
  case "$f" in
    js/*)
      files+=("${f#js/}")
      ;;
  esac
done

if [ ${#files[@]} -eq 0 ]; then
  exit 0
fi

cd js
pnpm exec oxlint --fix -- "${files[@]}"
