# Audio and video demo scripts

These scripts prove the proposed OpenInference audio and video keys. They do not call live vendor APIs. Each one builds a `current` span (what instrumentors emit today) and a `future` span (the keys in [vendor_comparison.md](../vendor_comparison.md)), then exports both to Phoenix.

New constants that are not in every pinned `openinference-semantic-conventions` release live as string literals in `common.py`.

## Prerequisites

Start an isolated Phoenix collector. Do not use a Phoenix on port 6006 that already has operator data.

```bash
export PHOENIX_COLLECTOR_ENDPOINT=http://127.0.0.1:<phoenix-port>
```

If you skip export, set `AUDIO_VIDEO_DEMO_SKIP_PHOENIX=1`. The scripts still assert the in-memory attribute dicts.

## Run

```bash
uv run --script internal_docs/specs/audio_video/scripts/gemini_video_demo.py
uv run --script internal_docs/specs/audio_video/scripts/openai_chat_audio_demo.py
```

Each script prints PASS or FAIL and exits non-zero on failure.

Open the Phoenix UI for that collector and compare the `current` and `future` spans.

## What each script proves

| Script | Current span | Future span |
|---|---|---|
| `gemini_video_demo.py` | LLM span with text contents only. Video URI may sit in `input.value`. | `message_content.type=video` plus `video.video.url`. No MIME type attribute. GenAI `uri` part with `modality: video`. |
| `openai_chat_audio_demo.py` | LLM span with text, no `input_audio` contents. | Input audio data URI on `message.contents`. No MIME type attribute. GenAI `blob` part with `modality: audio`, MIME read from the data URI prefix. |

## Layout

```
scripts/
├── README.md
├── common.py
├── gemini_video_demo.py
└── openai_chat_audio_demo.py
```
