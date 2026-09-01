---
"@arizeai/openinference-core": minor
---

Apply the existing image privacy controls to the span-level `input.images` / `output.images` attributes: `hideInputs` and `hideInputImages` remove input images, `hideOutputs` removes output images, and `base64ImageMaxLength` redacts oversized base64 payloads recorded under `<input|output>.images.[i].image.url`.
