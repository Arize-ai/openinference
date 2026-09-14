# Qwen-Agent Examples

Traces from these examples are exported to a local [Arize Phoenix](https://github.com/Arize-ai/phoenix) instance.

## Setup

```shell
pip install -r requirements.txt
phoenix serve          # in a separate terminal; UI at http://localhost:6006
```

## `assistant_dashscope.py`

An `Assistant` with a tool, running on DashScope. Requires `DASHSCOPE_API_KEY`.

```shell
export DASHSCOPE_API_KEY=...
python assistant_dashscope.py
```

This is the backend that surfaces token usage to Qwen-Agent, so the `LLM` span
carries `llm.token_count.*`.

## `assistant_openai_compatible.py`

The same agent against any OpenAI-compatible server — a local
[Ollama](https://ollama.com) by default (`ollama pull qwen3`), or a vLLM server
or DashScope's compatible-mode endpoint via `QWEN_MODEL_SERVER`.

```shell
python assistant_openai_compatible.py
```

Qwen-Agent discards the usage block on this backend, so the example also enables
`OpenAIInstrumentor`; the nested OpenAI-SDK span carries the token counts.

## `multi_agent_router.py`

A `Router` delegating to two specialised assistants, showing nested agent spans.
Requires `DASHSCOPE_API_KEY`.
