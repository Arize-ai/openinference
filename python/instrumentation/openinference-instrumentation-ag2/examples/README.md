# OpenInference AG2 Examples

Runnable examples that send AG2 traces to a local [Phoenix](https://github.com/Arize-ai/phoenix)
instance at `http://localhost:6006`. Each example sets its own Phoenix project through the
`openinference.project.name` resource attribute, so traces stay separated as you work
through them.

## Setup

```shell
pip install arize-phoenix
phoenix serve            # in a separate terminal
pip install -r requirements.txt
export OPENAI_API_KEY=<your-key>
```

## Examples

| Example | Phoenix project | Requires a key | What it traces |
| --- | --- | --- | --- |
| [`no_llm_multi_agent.py`](no_llm_multi_agent.py) | `ag2-no-llm-multi-agent` | No | A two-agent chat and a tool call, using canned replies |
| [`sessions_and_metadata.py`](sessions_and_metadata.py) | `ag2-sessions-and-metadata` | No | Two chats grouped into one Phoenix session, with user, metadata, and tags |
| [`conversable_agent_run.py`](conversable_agent_run.py) | `ag2-conversable-agent` | Yes | The quickstart agent driven by `run()` and `response.process()` |
| [`tool_calling.py`](tool_calling.py) | `ag2-tool-calling` | Yes | An LLM-driven tool call, split across a calling and an executing agent |
| [`group_chat.py`](group_chat.py) | `ag2-group-chat` | Yes | An `AutoPattern` group chat where a manager routes between specialists |
| [`sequential_chats.py`](sequential_chats.py) | `ag2-sequential-chats` | Yes | A queue of chats passing carryover forward via `initiate_chats` |
| [`structured_output.py`](structured_output.py) | `ag2-structured-output` | Yes | An agent replying with schema-validated JSON via `response_format` |
| [`async_tool_calling.py`](async_tool_calling.py) | `ag2-async-tool-calling` | Yes | The async paths: `a_initiate_chat`, `a_generate_reply`, `a_execute_function` |

If you have no API key handy, start with `no_llm_multi_agent.py` — it runs offline and is
the quickest way to confirm traces are reaching Phoenix.

```shell
python no_llm_multi_agent.py
```

`group_chat.py` produces the most detailed trace, including the manager's speaker
selection:

```
_User.initiate_chat [AGENT]
  chat_manager.generate_reply [AGENT]
    finance_bot.generate_reply [AGENT]
      ChatCompletion [LLM]
    checking_agent.initiate_chat [AGENT]
      speaker_selection_agent.generate_reply [AGENT]
        ChatCompletion [LLM]
    summary_bot.generate_reply [AGENT]
      ChatCompletion [LLM]
```

## Span kinds

| AG2 method | Span name | OpenInference span kind |
| --- | --- | --- |
| `initiate_chat` / `a_initiate_chat` (also used by `run` and `initiate_chats`) | `<agent>.initiate_chat` | `AGENT` |
| `generate_reply` / `a_generate_reply` | `<agent>.generate_reply` | `AGENT` |
| `execute_function` / `a_execute_function` | `<tool>` | `TOOL` |

The `LLM` spans come from instrumenting the OpenAI client alongside AG2, which the
LLM-backed examples do with `OpenAIInstrumentor`.

To send traces to Phoenix Cloud instead, point the exporter at your collector endpoint
and add your API key as described in the
[Phoenix docs](https://docs.arize.com/phoenix/tracing/how-to-tracing/setup-tracing).
