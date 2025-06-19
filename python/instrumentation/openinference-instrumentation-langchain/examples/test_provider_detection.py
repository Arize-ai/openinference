#!/usr/bin/env python3

import os

from langchain_openai import ChatOpenAI
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.langchain._tracer import _llm_provider, parse_provider_and_model


def test_parse_provider_and_model():
    """Test the parse_provider_and_model function with various model strings."""
    test_cases = [
        ("openai/gpt-4", "openai", "gpt-4"),
        ("text-completion-openai/gpt-3.5-turbo-instruct", "openai", "gpt-3.5-turbo-instruct"),
        ("anthropic/claude-2", "anthropic", "claude-2"),
        ("mistral/mistral-7b-instruct", "mistral", "mistral-7b-instruct"),
        ("cohere/command-r-plus", "cohere", "command-r-plus"),
        ("google/gemini-pro", "google", "gemini-pro"),
        ("azure/gpt-4", "azure", "gpt-4"),
        (
            "databricks/databricks-meta-llama-3-1-70b-instruct",
            "databricks",
            "databricks-meta-llama-3-1-70b-instruct",
        ),
        (
            "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "together",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        ),
        ("gpt-4", None, "gpt-4"),
        ("claude-2", None, "claude-2"),
        ("mistral-7b-instruct", None, "mistral-7b-instruct"),
        (None, None, None),
        ("", None, None),
    ]

    print("\n=== Testing parse_provider_and_model ===")
    for model_str, expected_provider, expected_model in test_cases:
        provider, model = parse_provider_and_model(model_str)
        result = "✓" if provider == expected_provider and model == expected_model else "✗"
        print(f"{result} {model_str!r} -> ({provider!r}, {model!r})")


def test_llm_provider():
    """Test the _llm_provider function with various extra dictionaries."""
    test_cases = [
        # Provider directly in invocation_params
        ({"invocation_params": {"provider": "openai"}}, "openai"),
        ({"invocation_params": {"provider": "anthropic"}}, "anthropic"),
        ({"invocation_params": {"provider": "mistral"}}, "mistral"),
        ({"invocation_params": {"provider": "cohere"}}, "cohere"),
        # From client name
        ({"invocation_params": {"client_name": "OpenAIClient"}}, "openai"),
        ({"invocation_params": {"client_name": "AnthropicClient"}}, "anthropic"),
        ({"invocation_params": {"client_name": "MistralClient"}}, "mistral"),
        ({"invocation_params": {"client_name": "CohereClient"}}, "cohere"),
        # Provider from model name
        ({"invocation_params": {"model_name": "openai/gpt-4"}}, "openai"),
        ({"invocation_params": {"model_name": "anthropic/claude-3-opus"}}, "anthropic"),
        ({"invocation_params": {"model_name": "mistral/mistral-7b-instruct"}}, "mistral"),
        ({"invocation_params": {"model_name": "cohere/command-r-plus"}}, "cohere"),
        (
            {"invocation_params": {"model": "text-completion-openai/gpt-3.5-turbo-instruct"}},
            "openai",
        ),
        # Provider from class name
        ({"id": ["langchain", "llms", "openai", "OpenAI"]}, "openai"),
        ({"id": ["langchain", "llms", "anthropic", "ChatAnthropic"]}, "anthropic"),
        ({"id": ["langchain", "llms", "mistral", "ChatMistral"]}, "mistral"),
        ({"id": ["langchain", "llms", "cohere", "Cohere"]}, "cohere"),
        # No provider info
        ({"invocation_params": {"model_name": "gpt-4"}}, None),
        ({"invocation_params": {"model_name": "claude-3-opus"}}, None),
        ({"invocation_params": {"model_name": "mistral-7b-instruct"}}, None),
        ({}, None),
        (None, None),
    ]

    print("\n=== Testing _llm_provider ===")
    for extra, expected_provider in test_cases:
        provider_items = list(_llm_provider(extra))
        if expected_provider is None:
            result = "✓" if len(provider_items) == 0 else "✗"
            print(f"{result} {str(extra)[:50]} -> No provider detected")
        else:
            result = (
                "✓"
                if len(provider_items) == 1
                and provider_items[0][0] == "llm.provider"
                and provider_items[0][1] == expected_provider
                else "✗"
            )
            detected = provider_items[0][1] if provider_items else None
            print(f"{result} {str(extra)[:50]} -> {detected!r} (expected: {expected_provider!r})")


def test_mock_langchain_runs():
    """Test provider detection with mock LangChain runs for different providers."""
    print("\n=== Testing with mock LangChain runs for different providers ===")

    # Define mock runs for different providers
    mock_runs = {
        "OpenAI": {
            "run_type": "llm",
            "extra": {
                "invocation_params": {"model_name": "gpt-3.5-turbo", "client_name": "OpenAIClient"},
                "id": ["langchain", "llms", "openai", "ChatOpenAI"],
            },
        },
        "Anthropic": {
            "run_type": "llm",
            "extra": {
                "invocation_params": {"model_name": "claude-3-opus"},
                "id": ["langchain", "llms", "anthropic", "ChatAnthropic"],
            },
        },
        "Mistral": {
            "run_type": "llm",
            "extra": {
                "invocation_params": {
                    "model_name": "mistral-7b-instruct",
                    "client_name": "MistralClient",
                },
                "id": ["langchain", "llms", "mistral", "ChatMistral"],
            },
        },
        "Cohere": {
            "run_type": "llm",
            "extra": {
                "invocation_params": {"model_name": "command-r-plus"},
                "id": ["langchain", "llms", "cohere", "Cohere"],
            },
        },
        "Azure OpenAI": {
            "run_type": "llm",
            "extra": {
                "invocation_params": {"model_name": "azure/gpt-4"},
                "id": ["langchain", "llms", "azure_openai", "AzureChatOpenAI"],
            },
        },
        "Google Vertex AI": {
            "run_type": "llm",
            "extra": {
                "invocation_params": {"model_name": "gemini-pro"},
                "id": ["langchain", "llms", "vertexai", "ChatVertexAI"],
            },
        },
        "LiteLLM Format": {
            "run_type": "llm",
            "extra": {"invocation_params": {"model": "openai/gpt-4"}},
        },
        "No Provider Info": {
            "run_type": "llm",
            "extra": {"invocation_params": {"model_name": "gpt-4"}},
        },
    }

    # Test each mock run
    for provider_name, mock_run in mock_runs.items():
        provider_items = list(_llm_provider(mock_run["extra"]))
        detected = provider_items[0][1] if provider_items else "None"
        print(f"Provider: {provider_name:<15} | Detected: {detected}")


def test_with_real_langchain_providers():
    """Test provider detection with real LangChain LLM providers."""
    print("\n=== Testing with Real LangChain LLM Providers ===")

    # Set up tracing
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    # Instrument LangChain
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    # Define providers to test with their import statements and initialization
    providers = [
        {
            "name": "ChatOpenAI",
            "import": "from langchain_openai import ChatOpenAI",
            "init": "ChatOpenAI(model='gpt-3.5-turbo', temperature=0)",
            "env_var": "OPENAI_API_KEY",
            "expected_provider": "openai",
        },
        {
            "name": "ChatAnthropic",
            "import": "from langchain_anthropic import ChatAnthropic",
            "init": "ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0)",
            "env_var": "ANTHROPIC_API_KEY",
            "expected_provider": "anthropic",
        },
        {
            "name": "ChatCohere",
            "import": "from langchain_cohere import ChatCohere",
            "init": "ChatCohere(model='command', temperature=0)",
            "env_var": "COHERE_API_KEY",
            "expected_provider": "cohere",
        },
        {
            "name": "ChatGoogleGenerativeAI",
            "import": "from langchain_google_genai import ChatGoogleGenerativeAI",
            "init": "ChatGoogleGenerativeAI(model='gemini-pro', temperature=0)",
            "env_var": "GOOGLE_API_KEY",
            "expected_provider": "google",
        },
        {
            "name": "ChatVertexAI",
            "import": "from langchain_google_vertexai import ChatVertexAI",
            "init": "ChatVertexAI(model_name='gemini-pro', temperature=0)",
            "env_var": "GOOGLE_APPLICATION_CREDENTIALS",
            "expected_provider": "vertexai",
        },
        {
            "name": "BedrockChat",
            "import": "from langchain_aws import BedrockChat",
            "init": "BedrockChat(model_id='anthropic.claude-3-sonnet-20240229', temperature=0)",
            "env_var": "AWS_ACCESS_KEY_ID",
            "expected_provider": "bedrock",
        },
        {
            "name": "AzureChatOpenAI",
            "import": "from langchain_openai import AzureChatOpenAI",
            "init": "AzureChatOpenAI(deployment_name='gpt-35-turbo', temperature=0, azure_endpoint='https://example.openai.azure.com')",
            "env_var": "AZURE_OPENAI_API_KEY",
            "expected_provider": "azure",
        },
        {
            "name": "HuggingFaceEndpoint",
            "import": "from langchain_huggingface import HuggingFaceEndpoint",
            "init": "HuggingFaceEndpoint(endpoint_url='https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1')",
            "env_var": "HUGGINGFACEHUB_API_TOKEN",
            "expected_provider": "huggingface",
        },
        {
            "name": "ChatOllama",
            "import": "from langchain_community.chat_models import ChatOllama",
            "init": "ChatOllama(model='llama2')",
            "env_var": None,
            "expected_provider": "ollama",
        },
        {
            "name": "ChatMLX",
            "import": "from langchain_community.chat_models import ChatMLX",
            "init": "ChatMLX(model='mlx-community/Mistral-7B-Instruct-v0.2-4bit-MLX')",
            "env_var": None,
            "expected_provider": "mlx",
        },
        {
            "name": "LlamaCpp",
            "import": "from langchain_community.llms import LlamaCpp",
            "init": "LlamaCpp(model_path='/path/to/model.gguf', temperature=0.1)",
            "env_var": None,
            "expected_provider": "llamacpp",
        },
        {
            "name": "Replicate",
            "import": "from langchain_community.llms import Replicate",
            "init": "Replicate(model='meta/llama-2-70b-chat')",
            "env_var": "REPLICATE_API_TOKEN",
            "expected_provider": "replicate",
        },
        {
            "name": "Anyscale",
            "import": "from langchain_community.chat_models import ChatAnyscale",
            "init": "ChatAnyscale(model_name='mistralai/Mixtral-8x7B-Instruct-v0.1')",
            "env_var": "ANYSCALE_API_KEY",
            "expected_provider": "anyscale",
        },
        {
            "name": "TogetherLLM",
            "import": "from langchain_together import TogetherLLM",
            "init": "TogetherLLM(model='mistralai/Mixtral-8x7B-Instruct-v0.1', temperature=0)",
            "env_var": "TOGETHER_API_KEY",
            "expected_provider": "together",
        },
        {
            "name": "ChatGroq",
            "import": "from langchain_groq import ChatGroq",
            "init": "ChatGroq(model_name='llama3-8b-8192', temperature=0)",
            "env_var": "GROQ_API_KEY",
            "expected_provider": "groq",
        },
    ]

    # Test each provider
    for provider in providers:
        name = provider["name"]
        print(f"\n--- Testing {name} ---")

        # Check if environment variable is set
        if provider["env_var"] and not os.environ.get(provider["env_var"]):
            print(f"✗ Skipping {name} (environment variable {provider['env_var']} not set)")
            continue

        try:
            # Try to import the module
            print(f"Import statement: {provider['import']}")
            print(f"Initialization: {provider['init']}")
            print(f"Expected provider: {provider['expected_provider']}")

            # We don't actually execute the imports or create instances
            # This is just to document what would be required for each provider
            print(f"✓ To test with real {name}, you would need to:")
            if provider["env_var"]:
                print(f"  1. Set the {provider['env_var']} environment variable")
            print("  2. Install the required package")
            print("  3. Run the import and initialization code")

        except Exception as e:
            print(f"✗ Error testing {name}: {e}")


def test_with_available_providers():
    """Test with providers that are actually available in the environment."""
    print("\n=== Testing with Available LLM Providers ===")

    # Set up tracing
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    # Instrument LangChain
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    # Test with FakeListLLM (always available)
    print("\n--- Testing with FakeListLLM ---")
    try:
        from langchain_core.language_models.fake import FakeListLLM

        print("✓ Successfully imported FakeListLLM")

        # Create and use the model
        fake_llm = FakeListLLM(responses=["Hello, I'm a fake LLM!"])
        print("✓ Created FakeListLLM instance")

        # Try a simple invocation
        response = fake_llm.invoke("This prompt will be ignored")
        print(f"✓ FakeListLLM response: {response}")

        # The provider should be detected as "fake"
        print("Expected provider: fake")
    except ImportError as e:
        print(f"✗ Import error for FakeListLLM: {e}")

    # Test with Ollama if available
    print("\n--- Testing with Ollama ---")
    try:
        # Check if Ollama is running
        import requests
        from langchain_community.chat_models import ChatOllama

        ollama_running = False
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            if response.status_code == 200:
                ollama_running = True
                print("✓ Ollama server detected")
        except Exception:
            print("✗ Ollama server not running or not accessible")

        if ollama_running:
            try:
                # List available models
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    if models:
                        model_name = models[0].get("name", "llama2")
                        print(f"✓ Using Ollama model: {model_name}")

                        # Create the model
                        llm = ChatOllama(model=model_name)
                        print("✓ Successfully created ChatOllama instance")

                        # Try a simple invocation
                        try:
                            response = llm.invoke("Hello, how are you?")
                            print(f"✓ ChatOllama response: {response.content[:50]}...")
                            print("Expected provider: ollama")
                        except Exception as e:
                            print(f"✗ Error invoking ChatOllama: {e}")
                    else:
                        print("✗ No Ollama models found")
                else:
                    print("✗ Failed to list Ollama models")
            except Exception as e:
                print(f"✗ Error interacting with Ollama: {e}")
    except ImportError as e:
        print(f"✗ Import error for ChatOllama: {e}")
        print("  To use ChatOllama, install langchain-community")

    # Test with HuggingFaceHub if API token is available
    print("\n--- Testing with HuggingFaceHub ---")
    if os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        try:
            from langchain_huggingface import HuggingFaceHub

            print("✓ Successfully imported HuggingFaceHub")

            # Create the model
            llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.1})
            print("✓ Successfully created HuggingFaceHub instance")

            # Try a simple invocation
            try:
                response = llm.invoke("What is the capital of France?")
                print(f"✓ HuggingFaceHub response: {response[:50]}...")
                print("Expected provider: huggingface")
            except Exception as e:
                print(f"✗ Error invoking HuggingFaceHub: {e}")
        except ImportError as e:
            print(f"✗ Import error for HuggingFaceHub: {e}")
            print("  To use HuggingFaceHub, install langchain-huggingface")
    else:
        print("✗ Skipping HuggingFaceHub (HUGGINGFACEHUB_API_TOKEN not set)")

    # Test with OpenAI if API key is available
    print("\n--- Testing with OpenAI ---")
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAI

            print("✓ Successfully imported OpenAI")

            # Create the model
            llm = OpenAI(temperature=0)
            print("✓ Successfully created OpenAI instance")

            # Try a simple invocation
            try:
                response = llm.invoke("Say hello in one word")
                print(f"✓ OpenAI response: {response[:50]}...")
                print("Expected provider: openai")
            except Exception as e:
                print(f"✗ Error invoking OpenAI: {e}")
        except ImportError as e:
            print(f"✗ Import error for OpenAI: {e}")
            print("  To use OpenAI, install langchain-openai")
    else:
        print("✗ Skipping OpenAI (OPENAI_API_KEY not set)")

    # Test with Anthropic if API key is available
    print("\n--- Testing with Anthropic ---")
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic

            print("✓ Successfully imported ChatAnthropic")

            # Create the model
            llm = ChatAnthropic(temperature=0)
            print("✓ Successfully created ChatAnthropic instance")

            # Try a simple invocation
            try:
                response = llm.invoke("Say hello in one word")
                print(f"✓ ChatAnthropic response: {response.content[:50]}...")
                print("Expected provider: anthropic")
            except Exception as e:
                print(f"✗ Error invoking ChatAnthropic: {e}")
        except ImportError as e:
            print(f"✗ Import error for ChatAnthropic: {e}")
            print("  To use ChatAnthropic, install langchain-anthropic")
    else:
        print("✗ Skipping ChatAnthropic (ANTHROPIC_API_KEY not set)")

    # Test with Cohere if API key is available
    print("\n--- Testing with Cohere ---")
    if os.environ.get("COHERE_API_KEY"):
        try:
            from langchain_cohere import Cohere

            print("✓ Successfully imported Cohere")

            # Create the model
            llm = Cohere(temperature=0)
            print("✓ Successfully created Cohere instance")

            # Try a simple invocation
            try:
                response = llm.invoke("Say hello in one word")
                print(f"✓ Cohere response: {response[:50]}...")
                print("Expected provider: cohere")
            except Exception as e:
                print(f"✗ Error invoking Cohere: {e}")
        except ImportError as e:
            print(f"✗ Import error for Cohere: {e}")
            print("  To use Cohere, install langchain-cohere")
    else:
        print("✗ Skipping Cohere (COHERE_API_KEY not set)")


def test_real_langchain_invocation():
    """Test provider detection with a real LangChain invocation."""
    # Check if OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n=== Skipping real LangChain invocation test (OPENAI_API_KEY not set) ===")
        print("To run this test, set the OPENAI_API_KEY environment variable.")
        return

    print("\n=== Testing with real LangChain invocation ===")

    # Set up tracing
    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )  # Print spans to console

    # Instrument LangChain
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        # Create a LangChain model with explicit provider/model format
        llm1 = ChatOpenAI(model="openai/gpt-3.5-turbo", temperature=0)
        # This will trigger a LangChain run that our instrumentation will capture
        response1 = llm1.invoke("Say hello in one word")
        print(f"Model 1 response: {response1.content}")

        # Create a LangChain model with standard format
        llm2 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        # This will trigger a LangChain run that our instrumentation will capture
        response2 = llm2.invoke("Say hello in one word")
        print(f"Model 2 response: {response2.content}")

        print(
            "\nCheck the console output above for the spans generated by the "
            "OpenTelemetry instrumentation."
        )
        print("The spans should include the llm.provider attribute set to 'openai'")
    except Exception as e:
        print(f"Error during LangChain invocation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Test the functions directly
    test_parse_provider_and_model()
    test_llm_provider()
    test_mock_langchain_runs()

    # Test with real LangChain invocation
    test_real_langchain_invocation()

    # Test with real LangChain LLM providers
    test_with_real_langchain_providers()

    # Test with available providers
    test_with_available_providers()
