"""
LangGraph Multi-Agent Example with OpenInference Tracing

This example demonstrates a multi-agent system using LangGraph with:
- A supervisor agent that routes between workers
- Search agent using Tavily
- Web scraper agent

Requires environment variables:
- OPENAI_API_KEY: Your OpenAI API key
- TAVILY_API_KEY: Your Tavily API key
- ARIZE_SPACE_ID: Your Arize space ID (from space settings)
- ARIZE_API_KEY: Your Arize API key (from space settings)
"""

import os
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Dict, List, Optional

from arize.otel import register
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing_extensions import TypedDict

from openinference.instrumentation.langchain import LangChainInstrumentor

# Verify API keys are set (should be set in your environment)
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not os.environ.get("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY environment variable is not set")

# Setup Arize OTel tracing
# Get your space_id and api_key from your Arize account space settings page
tracer_provider = register(
    space_id=os.environ.get("ARIZE_SPACE_ID"),  # Get from Arize space settings
    api_key=os.environ.get("ARIZE_API_KEY"),  # Get from Arize space settings
    project_name="langgraph-multi-agent",  # Name your project
)

# Import the automatic instrumentor from OpenInference
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Initialize tools
tavily_tool = TavilySearchResults(max_results=5)

# Set up temporary directory for file operations
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)


@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        (
            "Dictionary where key is the line number (1-indexed) and value is "
            "the text to be inserted at that line."
        ),
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


# Warning: This executes code locally, which can be unsafe when not sandboxed
repl = PythonREPL()


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"


def make_supervisor_node(llm: BaseChatModel, members: list[str]):
    """Create a supervisor node that routes between workers."""
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""

        next: str  # Will be one of the options

    def supervisor_node(state: MessagesState) -> Command:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto)

    return supervisor_node


# Initialize LLM
llm = ChatOpenAI(model="gpt-4o")

# Create agents
search_agent = create_react_agent(llm, tools=[tavily_tool])


def search_node(state: MessagesState) -> Command:  # type: ignore
    """Search node that uses Tavily to search the web."""
    result = search_agent.invoke(state)
    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="search")]},
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


web_scraper_agent = create_react_agent(llm, tools=[scrape_webpages])


def web_scraper_node(state: MessagesState) -> Command:  # type: ignore
    """Web scraper node that scrapes web pages."""
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [HumanMessage(content=result["messages"][-1].content, name="web_scraper")]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


# Create supervisor node
research_supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])

# Build the graph
research_builder = StateGraph(MessagesState)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)

research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()


if __name__ == "__main__":
    # Generate a session ID for this conversation
    # Use the same session_id across multiple runs to group them as a session
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}\n")
    
    # Import context setting utilities
    from openinference.instrumentation import using_attributes
    
    # Run the graph with session_id
    # The session_id will be attached to all spans in this trace
    with using_attributes(session_id=session_id):
        for s in research_graph.stream(
            {"messages": [("user", "what is braintrust?")]},
            {"recursion_limit": 100},
        ):
            print(s)
            print("---")
