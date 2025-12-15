#!/usr/bin/env bash
set -e

MODE="${1:-}"

if [ "$MODE" != "base" ] && [ "$MODE" != "new" ]; then
    echo "Usage: $0 {base|new}"
    exit 1
fi

if [ "$MODE" = "base" ]; then
    # Run baseline tests (existing tests that should pass before the fix)
    pytest python/instrumentation/openinference-instrumentation-agno/tests/test_workflow_instrumentation.py -v
elif [ "$MODE" = "new" ]; then
    # Run new tests (should fail before fix, pass after fix)
    pytest python/instrumentation/openinference-instrumentation-agno/tests/test_team_context_isolation.py -v
fi
