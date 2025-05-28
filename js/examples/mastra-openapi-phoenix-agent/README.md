# Phoenix Agent

A chatbot agent that helps you with Arize Phoenix.

It can be used to:

- Answer questions about Phoenix directly from the [Documentation](https://docs.arize.com/phoenix)
- Generate code snippets for instrumenting you TypeScript or Python code to send traces to Phoenix
- Directly read and perform actions on your live Phoenix instance
  - Create and update Prompts, Datasets, and Experiments with natural language

## Getting Started

### Prerequisites

- nvm (Node Version Manager)
- A running Phoenix Instance (local or remote)
  - See [Phoenix Environments overview](https://docs.arize.com/phoenix/environments) for instructions on how to get started in various runtime environments

### Setup

1. Install dependencies

```bash
cd mastra-openapi-phoenix-agent
# switch to the correct node version, v23.6.0 or higher is required
# install nvm first if you don't have it
nvm use
# install dependencies
npm install
```

2. Create a `.env.development` file and set the following environment variables:

```bash
PHOENIX_API_URL=http://localhost:6006
PHOENIX_API_KEY=your_api_key
OPENAI_API_KEY=your_openai_api_key
```

3. Run the agent (in dev mode)

```bash
npm run dev
```

4. Talk to the agent!

```bash
open http://localhost:4111/agents/phoenixAgent/chat
```

5. With Phoenix running, you will also see traces appear in the Phoenix UI under the `phoenix-agent` project.

## How it works

The agent is built with [Mastra](https://github.com/mastra-ai/mastra/tree/main), an open-source framework for building agents.

Check out their [documentation](https://mastra.ai/en/docs) for information on how to deploy and extend the Phoenix Agent.
