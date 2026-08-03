---
"@arizeai/openinference-genai": patch
---

fix(openinference-genai): keep input message indexes aligned when `gen_ai.system_instructions` carries no renderable parts

`mapSystemInstructions` emitted the synthetic system message at `llm.input_messages.0` whenever the `gen_ai.system_instructions` attribute was a non-empty string, while `mapInputMessages` only shifted its own indexes by one when that attribute parsed into at least one text part. For an instructions payload that is present but yields no parts (for example `[]`, or parts that are all non-text), the two disagreed: the system message claimed index 0 while the first real input message was also written at index 0, so the merged span reported the user's content under `message.role = "system"`.

Both mappings now key off the same parsed-parts predicate. The raw attribute is still preserved at `metadata.gen_ai.system_instructions` in every case, so nothing is lost.
