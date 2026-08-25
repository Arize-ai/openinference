import os

# Disable agno's telemetry API calls before any Workflow/Agent/Team is
# constructed. Telemetry posts to os-api.agno.com with a 60s httpx timeout;
# background-run tests await the workflow task and time out at 5s whenever
# that endpoint hangs in CI.
os.environ["AGNO_TELEMETRY"] = "false"
