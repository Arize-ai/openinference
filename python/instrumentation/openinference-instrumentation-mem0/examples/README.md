# mem0 Examples

Each example exports OpenInference spans to a local OTLP collector at `http://localhost:6006`
(for example `uvx arize-phoenix serve`, or `docker run -p 6006:6006 arizephoenix/phoenix:latest`).

```shell
pip install -r requirements.txt
python memory_offline.py
```

| Example | Project | Needs API key | What it traces |
| --- | --- | --- | --- |
| [`memory_offline.py`](memory_offline.py) | `mem0-offline` | No | `add`, `search`, `get_all`, `get`, `update`, `history`, `delete`, `delete_all` and one failing call, with a stub LLM and embedder and an in-memory Qdrant |
| [`memory_openai.py`](memory_openai.py) | `mem0-openai` | `OPENAI_API_KEY` | mem0's default OpenAI-backed `Memory`, with the OpenAI calls nested under each mem0 span |
