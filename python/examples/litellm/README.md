# LiteLLM Proxy Server

Use [LiteLLM Proxy](https://docs.litellm.ai/docs/simple_proxy) to Log OpenAI, Azure, Vertex, Bedrock (100+ LLMs) to Arize

Use LiteLLM Proxy for:
- Calling 100+ LLMs OpenAI, Azure, Vertex, Bedrock/etc. in the OpenAI ChatCompletions & Completions format
- Automatically Log all requests to Arize AI
- Proving a central self hosted server for calling LLMs + logging to Arize 


## Step 1. Create a Config for LiteLLM proxy

LiteLLM Requires a config with all your models define - we will call this file `litellm_config.yaml`

[Detailed docs on how to setup litellm config - here](https://docs.litellm.ai/docs/proxy/configs)

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/fake
      api_key: fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/

litellm_settings:
  success_callback: ["arize"] # 👈 Set Arize AI as a callback

environment_variables: # 👈 Set Arize AI env vars
    ARIZE_SPACE_KEY: "d0*****"
    ARIZE_API_KEY: "141a****"
```

Step 2. Start LiteLLM proxy

```shell
docker run \
    -v $(pwd)/litellm_config.yaml:/app/config.yaml \
    -p 4000:4000 \
    ghcr.io/berriai/litellm:main-latest \
    --config /app/config.yaml --detailed_debug
```

Step 3. Test it - Make /chat/completions request to LiteLLM proxy

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, Claude gm!"}
    ]
}'
```

## Expected output on Arize AI below:

<img width="1283" alt="Xnapper-2024-07-23-17 07 34" src="https://github.com/user-attachments/assets/7460bc2b-7f4f-4ec4-b966-2bf33a26ded5">


## Additional Resources
- [LiteLLM Arize AI docs](https://docs.litellm.ai/docs/observability/arize_integration)

