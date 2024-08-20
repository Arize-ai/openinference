from dspy import LM


class CustomLM(LM):
    def __init__(self, model):
        self.model = model

    def predict(self, inputs):
        return self.model.predict(inputs)

    def train(self, inputs, outputs):
        return self.model.train(inputs, outputs)

    def save(self, path):
        return self.model.save(path)

    def load(self, path):
        return self.model.load(path)

    def __str__(self):
        return str(self.model)
