from dsp import LM


# src: https://dspy-docs.vercel.app/docs/deep-dive/language_model_clients/custom-lm-client
class CustomLM(LM):
    """A Fake LM to test instrumentation"""

    def __init__(self, model, api_key, **kwargs):
        self.model = model
        self.api_key = api_key
        self.provider = "default"
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }
        self.history = []

    def basic_request(self, prompt, **kwargs):
        response = {"content": [{"text": "This is a test"}]}
        self.history.append(
            {
                "prompt": prompt,
                "response": "This is a test",
                "kwargs": kwargs,
            }
        )

        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)

        completions = [result["text"] for result in response["content"]]

        return completions
