"""Trace Claude or Codex into Agno's SQLite tracing database.

Install this instrumenter checkout and an Agno version with external adapters,
then install/authenticate the chosen harness SDK. Run, for example:

    python examples/external_agents.py claude
    python examples/external_agents.py codex

No Phoenix or Arize account is needed. SQLite is used for this local demo.
"""

import argparse
import asyncio

from agno.agents.base import BaseExternalAgent
from agno.db.sqlite import SqliteDb
from agno.tracing import setup_tracing


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", choices=("claude", "codex"))
    parser.add_argument("--db", default="external-agent-traces.db")
    parser.add_argument(
        "--prompt",
        default="Use your shell tool to print the current working directory, then report it.",
    )
    args = parser.parse_args()
    db = SqliteDb(db_file=args.db)
    # This registers AgnoInstrumentor and Agno's DatabaseSpanExporter together.
    setup_tracing(db=db)
    agent: BaseExternalAgent
    if args.harness == "claude":
        from agno.agents.claude import ClaudeAgent

        agent = ClaudeAgent(name="Traced Claude", db=db, allowed_tools=["Bash"])
    else:
        from agno.agents.codex import CodexAgent

        agent = CodexAgent(name="Traced Codex", db=db, model="gpt-5.6-luna")
    await agent.aprint_response(args.prompt, stream=True, user_id="local-test-user")
    print("Traces stored in:", args.db)


if __name__ == "__main__":
    asyncio.run(main())
