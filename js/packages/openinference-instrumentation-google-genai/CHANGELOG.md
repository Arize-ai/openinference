# @arizeai/openinference-instrumentation-google-genai

## 0.1.0

### Minor Changes

- Initial release of Google Gen AI SDK instrumentation for `@google/genai`
- Full OpenInference semantic conventions compliance
- Comprehensive coverage of SDK features:
  - **Models module**: `generateContent`, `generateContentStream`, `generateImages`
  - **Chats module**: `Chat.sendMessage`, `Chat.sendMessageStream`
- Instance-based instrumentation by wrapping `GoogleGenAI` constructor
- Support for text generation and streaming
- Support for multimodal content (text, images)
- Support for image generation
- Support for chat sessions with history management
- Support for function/tool calling
- Token usage tracking (prompt, completion, total, cached)
- Trace configuration for masking sensitive data
- Context attribute propagation
- Suppress tracing support
- Compatible with both Gemini Developer API and Vertex AI
- Example applications for basic usage, function calling, and multimodal features

### Implementation Details

This instrumentation works by wrapping the `GoogleGenAI` constructor to intercept instance creation. When a new `GoogleGenAI` instance is created, the instrumentation automatically wraps methods on:
- `ai.models.*` - Content generation methods
- `ai.chats.*` - Chat session methods

This approach ensures all method calls on any `GoogleGenAI` instance are automatically traced without requiring manual setup beyond registering the instrumentation.
