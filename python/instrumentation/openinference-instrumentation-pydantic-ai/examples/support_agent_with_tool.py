"""
Example of a Pydantic AI agent with instrumentation.

This example demonstrates:
1. Creating a structured AI agent for customer support
2. Adding custom tools and system prompts
3. Using dependencies injection for database access
4. Instrumenting the agent for observability

Requirements:
- pydantic-ai
- opentelemetry-api
- opentelemetry-sdk
- opentelemetry-exporter-otlp-proto-http
- openinference-instrumentation-pydantic-ai
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

# OpenTelemetry imports for observability
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor

resource = Resource.create(
    {
        "service.name": "banking-support-agent",
        "service.version": "1.0.0",
        "openinference.project.name": "banking-support-agent",
    }
)

# Set up the OpenTelemetry tracer provider
tracer_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(tracer_provider)

# Configure the exporter to send telemetry to Phoenix
otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:6006/v1/traces")
exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
tracer_provider.add_span_processor(BatchSpanProcessor(exporter))


class DatabaseConn:
    """Simulated database connection for demonstration purposes.

    In a production environment, this would be replaced with an actual
    connection to a database system like PostgreSQL, MongoDB, etc.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> Optional[str]:
        """Retrieve a customer's name by their ID."""
        # Simulated database lookup
        customer_data = {123: "John Smith", 456: "Jane Doe", 789: "Michael Johnson"}
        return customer_data.get(id)

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        """Retrieve a customer's account balance.
        Args:
            id: The customer's unique identifier
            include_pending: Whether to include pending transactions
        Returns:
            The customer's current balance
        Raises:
            ValueError: If the customer is not found
        """
        # Simulated database lookup with different results based on parameters
        if id not in (123, 456, 789):
            raise ValueError(f"Customer with ID {id} not found")

        balances = {
            123: {"settled": 100.00, "pending": 23.45},
            456: {"settled": 5432.10, "pending": 0.00},
            789: {"settled": 250.75, "pending": -50.25},
        }

        balance = balances[id]["settled"]
        if include_pending:
            balance += balances[id]["pending"]

        return balance


@dataclass
class SupportDependencies:
    """Dependencies required by the support agent.

    This class follows the dependency injection pattern, allowing
    for easy testing and component replacement.
    """

    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    """Structured output format for the support agent responses."""

    support_advice: str = Field(description="Personalized advice and information for the customer")
    block_card: bool = Field(
        description="Whether the customer's card should be blocked as a security measure"
    )
    risk: int = Field(description="Risk assessment level of the query (0-10)", ge=0, le=10)


# Create the AI agent with instrumentation
support_agent = Agent(
    "openai:gpt-4o",
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt=(
        "You are a support agent for Secure Banking Inc. Your role is to provide "
        "helpful support to customers while assessing the security risk of their queries. "
        "Always be professional, concise, and empathetic in your responses."
    ),
    instrument=True,
)


@support_agent.system_prompt
async def add_customer_context(ctx: RunContext[SupportDependencies]) -> str:
    """Dynamically add customer information to the system prompt.

    This function enriches the agent's system prompt with customer-specific
    information retrieved from the database.
    """
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    if customer_name:
        return f"The customer's name is {customer_name}. Address them by name in your response."
    else:
        return "Unable to retrieve customer name. Address them generically."


@support_agent.tool
async def customer_balance(ctx: RunContext[SupportDependencies], include_pending: bool) -> str:
    """Tool to retrieve the customer's current account balance.

    This function provides the agent with the ability to look up
    balance information when needed to answer customer queries.

    Args:
        ctx: The run context containing dependencies
        include_pending: Whether to include pending transactions

    Returns:
        A formatted string with the customer's balance
    """
    try:
        balance = await ctx.deps.db.customer_balance(
            id=ctx.deps.customer_id,
            include_pending=include_pending,
        )
        return f"${balance:.2f}"
    except ValueError as e:
        return f"Error retrieving balance: {str(e)}"


async def handle_customer_query(customer_id: int, query: str) -> SupportOutput:
    """Process a customer query and return a structured response.
    This function demonstrates how to use the agent in a real application.
    Args:
        customer_id: The ID of the customer making the query
        query: The text of the customer's question or request

    Returns:
        A structured response with advice and risk assessment
    """
    # Initialize dependencies
    deps = SupportDependencies(customer_id=customer_id, db=DatabaseConn())

    # Process the query with the AI agent
    result = await support_agent.run(query, deps=deps)

    # Return the structured output
    return result.output


async def main() -> None:
    """
    Example usage of the support agent.
    """

    # Example 1: Balance inquiry (low risk)
    result1 = await handle_customer_query(123, "What is my current account balance?")
    print("\nQuery: What is my current account balance?")
    print(f"Support Advice: {result1.support_advice}")
    print(f"Block Card: {result1.block_card}")
    print(f"Risk Level: {result1.risk}/10")

    # Example 2: Lost card (high risk)
    result2 = await handle_customer_query(123, "I just lost my credit card!")
    print("\nQuery: I just lost my credit card!")
    print(f"Support Advice: {result2.support_advice}")
    print(f"Block Card: {result2.block_card}")
    print(f"Risk Level: {result2.risk}/10")

    # Example 3: General question (medium risk)
    result3 = await handle_customer_query(
        456, "Can someone access my account if they know my email?"
    )
    print("\nQuery: Can someone access my account if they know my email?")
    print(f"Support Advice: {result3.support_advice}")
    print(f"Block Card: {result3.block_card}")
    print(f"Risk Level: {result3.risk}/10")


if __name__ == "__main__":
    asyncio.run(main())
