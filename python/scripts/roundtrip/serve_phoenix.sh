#!/usr/bin/env bash
# Starts a throwaway Phoenix in the background for the round trip and waits until it answers.
# Usage: serve_phoenix.sh [working dir, default a fresh temp dir]. Prints the log path on failure.
# Reuses a Phoenix already answering on PHOENIX_HOST (default http://localhost:6006).
set -uo pipefail
PHOENIX_VERSION=21.0.0
host=${PHOENIX_HOST:-http://localhost:6006}
curl -sf "$host/healthz" >/dev/null && { echo "phoenix: already up at $host"; exit 0; }
dir=${1:-$(mktemp -d)}
mkdir -p "$dir"
# The server reads PHOENIX_HOST as its bind address, not the URL px and the harness read it as.
env -u PHOENIX_HOST PHOENIX_WORKING_DIR="$dir" \
  nohup uvx --python 3.12 "arize-phoenix==$PHOENIX_VERSION" serve >"$dir/phoenix.log" 2>&1 &
for _ in $(seq 1 90); do
  curl -sf "$host/healthz" >/dev/null && { echo "phoenix: $PHOENIX_VERSION up at $host, log $dir/phoenix.log"; exit 0; }
  sleep 2
done
echo "phoenix: did not answer at $host after 180s; last lines of $dir/phoenix.log:"
tail -n 20 "$dir/phoenix.log"
exit 1
