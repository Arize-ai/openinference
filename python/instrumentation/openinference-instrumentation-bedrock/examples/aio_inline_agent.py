import asyncio
import time

import aioboto3
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.bedrock import BedrockInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

BedrockInstrumentor().instrument(tracer_provider=tracer_provider)

FOUNDATION_MODEL_NAME = "anthropic.claude-3-sonnet-20240229-v1:0"
HAIKU_FOUNDATION_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"

KNOWLEDGE_BASE_ID = "<KnowledgeBaseID>"
ACTION_GROUP_ARN = "ActionGroupLambdaARN"
AGENT_ALIAS_ARN = "CollaborationAgentAliasARN"


async def call_agent(params, region_name="us-east-1"):
    params["sessionId"] = f"default-session1_{int(time.time())}"
    session = aioboto3.session.Session()
    async with session.client("bedrock-agent-runtime", region_name) as client:
        response = await client.invoke_inline_agent(**params)
        async for event in response["completion"]:
            if "chunk" in event:
                print(event)
                chunk_data = event["chunk"]
                if "bytes" in chunk_data:
                    output_text = chunk_data["bytes"].decode("utf8")
                    print(output_text)
            elif "trace" in event:
                print(event)


def simple_agent():
    attributes = dict(
        foundationModel="anthropic.claude-3-5-sonnet-20240620-v1:0",
        instruction="You are a helpful assistant and need to help the user with your knowledge.",
        inputText="who is US President in 2001?",
        sessionId="default_session_id2",
        enableTrace=True,
    )
    asyncio.run(call_agent(attributes))


def code_gen_agent():
    attributes = dict(
        foundationModel=FOUNDATION_MODEL_NAME,
        instruction="You are a helpful assistant and need to help the user with their query.",
        inputText="Generate a python function to add two numbers.",
        actionGroups=[
            {
                "actionGroupName": "code_execution",
                "parentActionGroupSignature": "AMAZON.CodeInterpreter",
            }
        ],
        enableTrace=True,
    )
    asyncio.run(call_agent(attributes))


def full_processing_agent():
    attributes = dict(
        foundationModel=FOUNDATION_MODEL_NAME,
        instruction="You are a helpful assistant and need to help the user with their query.",
        inputText="Who is srinivas ramanujan? give short story about him",
        enableTrace=True,
        promptOverrideConfiguration={
            "promptConfigurations": [
                {
                    "foundationModel": HAIKU_FOUNDATION_MODEL,
                    "inferenceConfiguration": {
                        "maximumLength": 2048,
                        "temperature": 0,
                        "topK": 250,
                        "topP": 0,
                    },
                    "parserMode": "DEFAULT",
                    "promptCreationMode": "DEFAULT",
                    "promptState": "ENABLED",
                    "promptType": "PRE_PROCESSING",
                },
                {
                    "foundationModel": HAIKU_FOUNDATION_MODEL,
                    "inferenceConfiguration": {
                        "maximumLength": 2048,
                        "temperature": 0,
                        "topK": 250,
                        "topP": 0,
                    },
                    "parserMode": "DEFAULT",
                    "promptCreationMode": "DEFAULT",
                    "promptState": "ENABLED",
                    "promptType": "POST_PROCESSING",
                },
            ]
        },
    )
    asyncio.run(call_agent(attributes))


def knowledge_base_agent():
    attributes = dict(
        foundationModel=FOUNDATION_MODEL_NAME,
        instruction="You are a helpful assistant and need to help the user with their query "
        "using knowledge base.",
        inputText="What is Task Decomposition?",
        knowledgeBases=[
            {
                "description": "Task Decomposition Knowledge Base",
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "retrievalConfiguration": {"vectorSearchConfiguration": {}},
            }
        ],
        enableTrace=True,
    )
    asyncio.run(call_agent(attributes, "ap-south-1"))


def action_group():
    attributes = dict(
        foundationModel=FOUNDATION_MODEL_NAME,
        actionGroups=[
            {
                "actionGroupName": "action_group_quick_start_6gq19",
                "actionGroupExecutor": {"lambda": ACTION_GROUP_ARN},
                "functionSchema": {
                    "functions": [
                        {
                            "name": "add_two_numbers",
                            "description": "This function adds the two numbers and returns the Sum"
                            " of two numbers, It takes the input as two numbers",
                            "parameters": {
                                "n1": {
                                    "description": "First Number for sum",
                                    "required": True,
                                    "type": "number",
                                },
                                "n2": {
                                    "description": "Second number for Sum",
                                    "required": True,
                                    "type": "number",
                                },
                            },
                            "requireConfirmation": "DISABLED",
                        },
                        {
                            "name": "get_time",
                            "description": "This function returns the current time of the system",
                            "parameters": {},
                            "requireConfirmation": "DISABLED",
                        },
                    ]
                },
            }
        ],
        instruction="You are a helpful assistant and need to help the user with their query"
        " using Tools.",
        inputText="What is the sum of 10 and 20?",
        enableTrace=True,
    )
    asyncio.run(call_agent(attributes, "us-east-1"))


def multi_agent_colab():
    attributes = dict(
        foundationModel=FOUNDATION_MODEL_NAME,
        instruction="You are MasterAgent. When a user request arrives, determine whether to use"
        " SimpleSupervisor for math or research tasks, LoggingAgent for audit, or "
        "fallback logic. Invoke collaborators concurrently, enforce guardrails, and "
        "merge their outputs into a cohesive response.",
        inputText="What is the sum of 10 and 20?",
        enableTrace=True,
        agentCollaboration="SUPERVISOR",
        collaboratorConfigurations=[
            {
                "collaboratorName": "SimpleSupervisor",
                "collaboratorInstruction": "You are SimpleSupervisor. Split user requests into "
                "either math or research tasks. Invoke MathSolverAgent "
                "for any calculation, and WebResearchAgent for any "
                "fact-finding. Consolidate both outputs into a single"
                " response.",
                "agentAliasArn": AGENT_ALIAS_ARN,
                "relayConversationHistory": "DISABLED",
            },
        ],
    )
    asyncio.run(call_agent(attributes, "us-east-1"))


if __name__ == "__main__":
    # simple_agent()
    code_gen_agent()
    # full_processing_agent()
    # knowledge_base_agent()
    # action_group()
    # multi_agent_colab()
