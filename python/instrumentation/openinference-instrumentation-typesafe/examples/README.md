# TypeSafe AI Examples

Each example exports OpenInference spans to a local OTLP collector at `http://localhost:6006`
(for example `uvx arize-phoenix serve`) and needs `TYPESAFE_API_KEY` set.

```shell
pip install -r requirements.txt
python system_one.py
```

| Example | Project | Needs API key | What it traces |
| --- | --- | --- | --- |
| [`system_one.py`](system_one.py) | `typesafe-system-one` | Yes | One `TypeSafeClient.system_one` call asking a Noul, a Choice, and a Score over structured state |
| [`async_system_one.py`](async_system_one.py) | `typesafe-async-system-one` | Yes | Two concurrent `AsyncTypeSafeClient.system_one` calls |
| [`context_attributes.py`](context_attributes.py) | `typesafe-context-attributes` | Yes | Session, user, metadata, and tag propagation, plus one suppressed call |
