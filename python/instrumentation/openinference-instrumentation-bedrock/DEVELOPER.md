# Amazon Bedrock Agent Developer Guide

## Creating Agents using Amazon Bedrock

Amazon Bedrock Agents allow developers to build intelligent, multi-step applications using foundation models (FMs) integrated with knowledge bases, tools (action groups), and memory. This guide walks through the full setup process, including knowledge base, and code interpreter support.

---

### Prerequisites

- AWS account with Bedrock access in your region
- IAM permissions for:
  - Bedrock
  - Lambda (for tools)
  - Open Search or Pinecone (for vector stores)
---

## Steps to Create an Agent in Amazon Bedrock

### 1. **Define the Agent Purpose**

Clarify the tasks your agent will perform:
- Answer FAQs
- Perform task automation
- Query documents
- Run calculations (with code interpreter)

---

### 2. **Create the Agent**

1. Go to **Amazon Bedrock > Agents**
2. Click **Create agent**
3. Fill out:
   - Agent name
   - Instruction prompt
   - Foundation model (e.g., `claude 3.5 sonnet`)
   - Idle session TTL

---
### 3. **Configure Action Groups Using Lambda for Tools**

Action groups are used to link Lambda functions (or other tools) to the agent's workflow. This allows the agent to call an API or execute a Lambda function during a session to perform tasks like querying databases, making external API calls, or processing complex logic.

Here’s how to configure an **Action Group** using Lambda:

#### **Create Your Lambda Function**
Before attaching the Lambda function to an agent, make sure you have the Lambda function already created.

- Go to the **AWS Lambda Console**.
- Click **Create function**.
- Choose **Author from scratch**, and provide:
  - **Function name**
  - **Runtime** (e.g., Python 3.x, Node.js)
  - **Execution role** (Lambda execution role with permissions to access the required services)
- Add your Lambda code that performs the desired task (e.g., querying an external API, data processing).

Example of a simple Lambda function (Python):
```python
import json

def lambda_handler(event, context):
    order_id = event.get('orderId')
    # Simulating order status lookup
    order_status = "Shipped" if order_id == "1234" else "Not Found"
    return {
        'statusCode': 200,
        'body': json.dumps({'orderStatus': order_status})
    }
```

#### **Create an Action Group**
Once your Lambda function is ready, you can create an Action Group in your Bedrock Agent.

**Console Steps:**
1. Go to **Amazon Bedrock > Agents**.
2. Open the agent's details page.
3. In the **Action Groups** tab, click **Add**.
4. Configure the action group:
   - **Name**: Descriptive name for the action (e.g., `OrderStatusLookup`).
   - **Description**: Short description of the task.
   - **Lambda Function**: Select the existing lambda function

5. Configure the **Input Parameters**:
   - Define which parameters the Lambda function expects (e.g., `orderId`).
   - You can use placeholders or variables from the agent's session.

6. Save the **Action Group**.

#### **Link the Action Group to Agent**
Once the Action Group is created, you need to link it to your agent.

1. In your agent's **Action Groups** tab, click **Attach Action Group**.
2. Select the newly created action group (e.g., `OrderStatusLookup`).
3. Optionally, specify **utterances** (phrases the agent will recognize to trigger the action).
4. Save the configuration.

#### **Invoke the Lambda Action Group**
Now that the Action Group is set up, the agent can invoke the Lambda function whenever it matches the condition or intent.

### 4. **Configure and Attach a Knowledge Base**

####  Create Knowledge Base
1. Go to **Bedrock > Knowledge bases**
2. Click **Create knowledge base**
3. Provide Knowledge Base Details
   1. Name
   2. Description
   3. IAM Permissions
   4. Data source(e.g., Web Crawler/Amazon S3)
4. Configure Data source
   1. Provide Name, description, source urls
5. Configure data storage and processing
   1. Choose Embeddings Model
   2. Choose Vector Database(Choose existing vector store)
   3. Choose Pinecone from Dropdown
   4. Provide Pinecone API Details
6. Review and Create knowledge base


#### Attach to Agent
- In agent settings, open the **Knowledge Bases** tab
- Click **Attach Knowledge Base**
- Add Knowledge base instructions and Select the knowledge base from dropdown
---

### 5. **Enable Code Interpreter**

The **code interpreter** lets the agent execute Python code to solve analytical and logic-based problems.

#### Steps to Enable Code Interpreter:

**Console Steps:**
1. Go to **Amazon Bedrock > Agents**.
2. Open your agent and go to **Agent settings**.
3. Scroll to **Tools** section.
4. Enable:
   - **Enable code interpreter (built-in tool for reasoning and calculations)**
5. Save changes.
---

### 6. **Test the Agent**

```python
import boto3
session = boto3.session.Session()
client = session.client("bedrock-agent-runtime", "us-east-1")
response = client.invoke_agent(
    agentId='your-agent-id',
    sessionId='demo-session',
    inputText='Can you calculate the average order value from $120, $150, and $90?',
    enableTrace=True
)
for idx, event in enumerate(response["completion"]):
    if "chunk" in event:
        print(event)
        chunk_data = event["chunk"]
        if "bytes" in chunk_data:
            output_text = chunk_data["bytes"].decode("utf8")
            print(output_text)
    elif "trace" in event:
        print(event["trace"]["trace"])
```

---





#### Trace Events Covered
When tracing is enabled, the system will capture and return traces for the following stages of agent execution:

- `preProcessing`
- `guardRail`
- `orchestration`
- `postProcessing`
- `failureTrace`

#### Agent Configuration & Traceable Contexts

The agent can be configured to capture trace data during the following contexts:

**1. PreProcessing**  
This stage evaluates user input—e.g., checking if it's safe or non-harmful. It involves an LLM invocation to perform this evaluation. If `preProcessing` is enabled at the agent level and tracing is turned on, these LLM calls will be captured and returned in the trace.

**2. Orchestration**  
When orchestration is enabled, the agent determines which AWS services (via ActionGroups such as Lambda, API Gateway, Code Interpreter, or Knowledge Bases) to invoke based on the user input.  
- After preProcessing passes, the agent sends the input to the LLM.
- The LLM decides the next step (e.g., calling an ActionGroup or Knowledge Base).
- The agent then invokes the corresponding tool.

Traces captured during orchestration will include:
- `invocationInput`: Input to the selected Agent Tool
- `invocationOutput`: Output returned from the tool

These responses are then forwarded to the LLM for further processing (another LLM call), which will also be traced.

**3. PostProcessing**  
After the LLM generates a final response, it is passed through a post-processing phase—again, an LLM invocation—to validate or refine the output before returning it to the user.

#### Model Invocation Traces

Each LLM interaction—whether during preProcessing, orchestration, or postProcessing—will include:
- `modelInvocationInput`
- `modelInvocationOutput`

These will be exposed in the trace data returned to the user.

#### Instrumentation and Logging

Our internal instrumentor will capture and log all relevant trace data to the **Phonix UI**, enabling detailed visibility into the end-to-end agent execution flow.