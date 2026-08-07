# OpenInference AG2 Examples

Runnable examples that send AG2 traces to a local [Phoenix](https://github.com/Arize-ai/phoenix)
instance at `http://localhost:6006`.

## Setup

```shell
pip install arize-phoenix
phoenix serve            # in a separate terminal
pip install -r requirements.txt
```

## Examples

| Example | Requires an API key | What it traces |
| --- | --- | --- |
| [`no_llm_multi_agent.py`](no_llm_multi_agent.py) | No | A two-agent chat and a tool call, using canned replies |
| [`sessions_and_metadata.py`](sessions_and_metadata.py) | No | Two chats grouped into one Phoenix session, with user, metadata, and tags |
| [`openai_tool_calling.py`](openai_tool_calling.py) | `OPENAI_API_KEY` | An LLM-driven tool call, with OpenAI spans nested under the agent spans |
| [`async_tool_calling.py`](async_tool_calling.py) | `OPENAI_API_KEY` | The async paths: `a_initiate_chat`, `a_generate_reply`, and `a_execute_function` |

Start with `no_llm_multi_agent.py` — it runs offline and is the quickest way to confirm
traces are reaching Phoenix.

```shell
python no_llm_multi_agent.py
```

## Span kinds

| AG2 method | Span name | OpenInference span kind |
| --- | --- | --- |
| `initiate_chat` / `a_initiate_chat` | `<agent>.initiate_chat` | `AGENT` |
| `generate_reply` / `a_generate_reply` | `<agent>.generate_reply` | `AGENT` |
| `execute_function` / `a_execute_function` | `<tool>` | `TOOL` |

To send traces to Phoenix Cloud instead, point the exporter at your collector endpoint
and add your API key as described in the
[Phoenix docs](https://docs.arize.com/phoenix/tracing/how-to-tracing/setup-tracing).
