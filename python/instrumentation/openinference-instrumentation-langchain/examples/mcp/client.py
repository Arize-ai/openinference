# Create server parameters for stdio connection
import asyncio

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from phoenix.otel import register

from openinference.instrumentation.mcp import MCPInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(project_name="langchain-mcp-adapters")
MCPInstrumentor().instrument(tracer_provider=tracer_provider)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


async def main():
    model = ChatOpenAI(model="gpt-4")

    server_params = StdioServerParameters(
        command="python",
        # Make sure to update to the full absolute path to your math_server.py file
        args=["./math_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            print(agent_response)


if __name__ == "__main__":
    asyncio.run(main())
