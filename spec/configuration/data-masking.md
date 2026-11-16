# Data Masking

Prompts, completions, retrieved documents, and media frequently contain personal or sensitive information. The
[trace configuration](./trace-config.md) settings let an application keep that content out of its traces. This page
describes what each setting hides.

## Redacted Content

When content is hidden by a setting, its value is replaced with the placeholder `"__REDACTED__"`, or the attribute is
omitted. The constant placeholder lets consumers tell that content was intentionally hidden rather than missing or
empty.

## Inputs and Outputs

- `OPENINFERENCE_HIDE_INPUTS` hides `input.value`, `input.images`, `input.documents`, all input messages, and the tool
  definitions advertised to the LLM (`llm.tools.*`).
- `OPENINFERENCE_HIDE_OUTPUTS` hides `output.value`, `output.images`, `output.documents`, and all output messages.

## Messages

- Input messages are hidden if either `OPENINFERENCE_HIDE_INPUTS` or `OPENINFERENCE_HIDE_INPUT_MESSAGES` is true.
- Output messages are hidden if either `OPENINFERENCE_HIDE_OUTPUTS` or `OPENINFERENCE_HIDE_OUTPUT_MESSAGES` is true.

When messages are not already hidden, finer settings hide parts of them:

- `OPENINFERENCE_HIDE_INPUT_TEXT` replaces text content in input messages with `"__REDACTED__"`, including message
  document `document.content`.
- `OPENINFERENCE_HIDE_OUTPUT_TEXT` does the same for output messages.
- `OPENINFERENCE_HIDE_INPUT_IMAGES` replaces image URLs in input messages with `"__REDACTED__"`.

## LLM Request Details

- `OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS` hides `llm.invocation_parameters`, independent of input and output
  hiding.
- `OPENINFERENCE_HIDE_LLM_TOOLS` hides the tool definitions advertised to the LLM (`llm.tools.*`). They are also hidden
  when `OPENINFERENCE_HIDE_INPUTS` is true.
- `OPENINFERENCE_HIDE_PROMPTS` hides the prompts of a completions API call (`llm.prompts.*`).
- `OPENINFERENCE_HIDE_CHOICES` hides the choices of a completions API call (`llm.choices.*`).

## Embeddings

- `OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS` replaces each `embedding.embeddings.N.embedding.vector` value with
  `"__REDACTED__"`, so vector data is not included in traces. `OPENINFERENCE_HIDE_EMBEDDING_VECTORS` is a deprecated
  name for the same setting.
- `OPENINFERENCE_HIDE_EMBEDDINGS_TEXT` replaces each `embedding.embeddings.N.embedding.text` value with
  `"__REDACTED__"`, so the input text is not included in traces.

## Retrieval Documents

`OPENINFERENCE_HIDE_RETRIEVAL_DOCUMENTS` replaces `retrieval.documents.*.document.content` and
`retrieval.documents.*.document.metadata` values with `"__REDACTED__"`. `OPENINFERENCE_HIDE_OUTPUTS` implies it.
Only the Python SDK supports this setting today.

## Images and Documents

The image settings apply to the span-kind-independent `input.images` and `output.images` attributes the same way they
apply to message-content images:

- `OPENINFERENCE_HIDE_INPUT_IMAGES` removes all `input.images.*` attributes.
- `OPENINFERENCE_HIDE_INPUTS` removes `input.images.*` and `OPENINFERENCE_HIDE_OUTPUTS` removes `output.images.*`, just
  as they hide `input.value` and `output.value`.

### Base64 Size Limit

`OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` (default `32000`) limits the length of a base64 data URI recorded in an image
`image.url` or a document `document.url`, in messages and in `<input|output>.images` and `<input|output>.documents`.
The setting keeps its image-specific name but sets the limit for both.

- A data URI longer than the limit is replaced with `"__REDACTED__"`.
- If a [blob uploader](./blob-upload.md) is configured, an over-limit image or document is uploaded instead, and the
  attribute records the destination URI.
- Hide settings take precedence: hidden content is never uploaded.
- Audio and video data URIs are not size-gated, and the shared masking does not yet hide audio or video.

### SDK Support

Support for the image and document settings differs by SDK: Python enforces the size limit and can externalize to a
blob uploader; JavaScript enforces the size limit and redacts oversized images to `"__REDACTED__"`; Java and Go apply
the hide flags only.

## Feedback

[Annotation and evaluation](../feedback/annotations.md#encoding-and-privacy) explanations and metadata can contain
sensitive data and MUST follow the same masking and access policies as other span content.
