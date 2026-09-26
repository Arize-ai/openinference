# TypeSafe AI Examples

All examples need `TYPESAFE_API_KEY` set.

## Raw HTTP input/output gallery

Run [`raw_http.py`](raw_http.py) from this directory:

```shell
uv run raw_http.py
```

The script declares its own `httpx` dependency and makes four sequential requests to the
[TypeSafe HTTP API](https://docs.typesafe.ai/api). It prints each labeled request and full JSON
response, without the TypeSafe SDK or a tracing setup.

| Scenario | Questions | Input/output details to explore |
| --- | --- | --- |
| Content moderation | Three Noul judgments | Plain text; overlapping judgments distinguish a quoted threat from the author's own harassment |
| Support routing | Two Choice classifications | Conversation array; structured instructions and criteria; competing categories and fallback options |
| Answer quality | Three Scores | Answer plus reference data; two-, three-, and four-level rubrics; structured instructions and level descriptions |
| Refund review | Noul + Choice + Score | Nested order, policy, and history; eligibility, recommended action, and urgency on the same state |

Classification uses `type: "choice"` in the API. Match each question to its answer by its key:
Noul exposes a probability of yes; Choice exposes a selected label, confidence, and category
probabilities; Score exposes a weighted score, confidence, probabilities, and a level legend.
Keep the input criteria alongside these outputs when exploring renderings such as probability
bars, category distributions, or labeled score scales. Live model judgments can vary between runs.

## Instrumented SDK examples

These examples export OpenInference spans to a local OTLP collector at `http://localhost:6006`
(for example `uvx arize-phoenix serve`).

```shell
uv run --with-requirements requirements.txt system_one.py
```

| Example | Project | Needs API key | What it traces |
| --- | --- | --- | --- |
| [`system_one.py`](system_one.py) | `typesafe-system-one` | Yes | One `TypeSafeClient.system_one` call asking a Noul, a Choice, and a Score over structured state |
| [`async_system_one.py`](async_system_one.py) | `typesafe-async-system-one` | Yes | Two concurrent `AsyncTypeSafeClient.system_one` calls |
| [`context_attributes.py`](context_attributes.py) | `typesafe-context-attributes` | Yes | Session, user, metadata, and tag propagation, plus one suppressed call |
