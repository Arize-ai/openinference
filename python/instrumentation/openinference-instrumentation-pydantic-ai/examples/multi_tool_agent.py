from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor

from pydantic_ai import Agent, RunContext

# OpenTelemetry setup
endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
exporter = OTLPSpanExporter(endpoint=endpoint)
trace.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
tracer_provider.add_span_processor(BatchSpanProcessor(exporter))


# Define dependencies type
class ApiDeps:
    def __init__(self, user_id: str, api_key: str):
        self.user_id = user_id
        self.api_key = api_key


# Create agent with system prompt that encourages multiple tool usage
agent = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=ApiDeps,
    system_prompt=(
        "You are a technical documentation assistant. When users ask about API differences "
        "or technical comparisons, you should:\n"
        "1. Search multiple sources to gather comprehensive information\n"
        "2. Use parallel tool calls when possible for efficiency\n"
        "3. Combine results to provide a complete answer\n"
        "Always use multiple tools when the query requires different types of information."
    ),
    instrument=True,
)


@agent.tool
def api_search(ctx: RunContext[ApiDeps], query: str) -> str:
    """
    Search API documentation for technical information.

    Args:
        query: The search query for API documentation
    """
    print(f"[API_SEARCH] Searching for: {query}")

    # Simulate API documentation search
    results = {
        "REST vs GraphQL API architectural differences": {
            "REST": "Representational State Transfer uses HTTP methods and endpoints",
            "GraphQL": "Query language with single endpoint and flexible data fetching",
            "key_differences": ["endpoint structure", "data fetching", "versioning"]
        },
        "REST GraphQL API design patterns": {
            "REST_patterns": ["Resource-based URLs", "HATEOAS", "Richardson Maturity Model"],
            "GraphQL_patterns": ["Schema-first design", "Resolvers", "DataLoader"]
        }
    }

    # Find matching results
    for key, value in results.items():
        if any(term in query.lower() for term in key.lower().split()):
            return f"API Documentation Results: {value}"

    return f"No specific results found for: {query}"


@agent.tool
def generic_search(ctx: RunContext[ApiDeps], query: str) -> str:
    """
    Search general web sources for technical information.

    Args:
        query: The search query for general web search
    """
    print(f"[GENERIC_SEARCH] Searching for: {query}")

    # Simulate web search results
    results = {
        "REST GraphQL API design patterns": [
            "Best practices for REST API design",
            "GraphQL schema design considerations",
            "Performance comparison between REST and GraphQL"
        ],
        "API architectural differences": [
            "When to use REST vs GraphQL",
            "Scalability considerations",
            "Client-side caching strategies"
        ]
    }

    for key, value in results.items():
        if any(term in query.lower() for term in key.lower().split()):
            return f"Web Search Results: {', '.join(value)}"

    return f"General information about: {query}"


@agent.tool
def code_examples_search(ctx: RunContext[ApiDeps], technology: str) -> str:
    """
    Search for code examples and implementations.

    Args:
        technology: The technology to find code examples for
    """
    print(f"[CODE_EXAMPLES] Searching examples for: {technology}")

    examples = {
        "REST": "GET /users/{id}, POST /users, PUT /users/{id}, DELETE /users/{id}",
        "GraphQL": "query { user(id: 1) { name email posts { title } } }",
        "API": "Common API implementation patterns and best practices"
    }

    for key, value in examples.items():
        if key.lower() in technology.lower():
            return f"Code Examples: {value}"

    return f"No code examples found for: {technology}"


@agent.tool
def get_user_context(ctx: RunContext[ApiDeps]) -> str:
    """
    Get the current user's context and preferences.
    """
    print(f"[USER_CONTEXT] Getting context for user: {ctx.deps.user_id}")
    return f"User: {ctx.deps.user_id}, Experience Level: Intermediate"


# Example 1: Query that triggers multiple tool calls
def example_multiple_tools():
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Multiple Tool Calls")
    print("=" * 80 + "\n")

    deps = ApiDeps(user_id="dev_user_123", api_key="secret_key")

    result = agent.run_sync(
        "What's the difference between REST and GraphQL? I need comprehensive information.",
        deps=deps
    )

    print("\n--- AGENT RESPONSE ---")
    print(result.output)
    print("\n--- TOOL CALLS MADE ---")
    for msg in result.all_messages():
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if part.part_kind == 'tool-call':
                    print(f"Tool: {part.tool_name}, Args: {part.args}")


# Example 2: Complex query requiring multiple data sources
def example_complex_query():
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Complex Query with Multiple Data Sources")
    print("=" * 80 + "\n")

    deps = ApiDeps(user_id="senior_dev_456", api_key="secret_key")

    result = agent.run_sync(
        "I need to compare REST and GraphQL APIs. Show me architectural differences, "
        "design patterns, and code examples for both.",
        deps=deps
    )

    print("\n--- AGENT RESPONSE ---")
    print(result.output)
    print("\n--- ALL MESSAGES ---")
    for i, msg in enumerate(result.all_messages()):
        print(f"\nMessage {i + 1}: {msg.role}")
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if part.part_kind == 'tool-call':
                    print(f"  [TOOL CALL] {part.tool_name}({part.args})")
                elif part.part_kind == 'tool-return':
                    print(f"  [TOOL RESULT] {part.content[:100]}...")


# Example 3: Streaming with multiple tool calls
async def example_streaming_multiple_tools():
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Streaming with Multiple Tool Calls")
    print("=" * 80 + "\n")

    deps = ApiDeps(user_id="dev_user_789", api_key="secret_key")

    async with agent.run_stream(
            "Explain REST vs GraphQL with examples and best practices",
            deps=deps
    ) as response:
        print("\n--- STREAMING RESPONSE ---")
        async for chunk in response.stream_text():
            print(chunk, end='', flush=True)

        print("\n\n--- TOOL USAGE ---")
        result = await response.get_result()
        for msg in result.all_messages():
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if part.part_kind == 'tool-call':
                        print(f"Tool Used: {part.tool_name}")


if __name__ == "__main__":
    # Run synchronous examples
    example_multiple_tools()
