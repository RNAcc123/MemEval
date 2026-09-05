"""Deterministic generation backend for offline runner checks."""

from memeval.generation.traced import GenerationResponse


class FakeGenerationBackend:
    name = "fake-llm"

    def complete(self, messages, model, parameters):
        return GenerationResponse("[fake] " + messages[-1]["content"].split("Question:\n", 1)[-1].strip())
