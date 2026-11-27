---
"@arizeai/openinference-instrumentation-openai": patch
---

fix(openai): keep the mask at its request index in `images.edit` spans when the source image cannot be read, so the mask is no longer recorded as the source image.
