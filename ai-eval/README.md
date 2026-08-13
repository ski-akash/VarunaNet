# ai-eval/

Evaluation for the LLM layer specifically — separate from the model benchmarks in `benchmarks/`, because "is the chatbot any good" is a different question from "is the segmentation model any good."

This folder holds:
- A golden set of at least 50 questions (spanning all four AI features) with known-correct answers.
- Scoring code for tool-selection accuracy, parameter-extraction accuracy, answer correctness, and groundedness rate (what fraction of numbers the model states can actually be traced back to a tool call).
- Adversarial test cases: questions about regions with no data, ambiguous date ranges, and prompt-injection attempts typed into the chat box.
- Cost and latency reporting per query, with and without the semantic cache.

The project's hard rule is that the LLM can never state a number it didn't get from a tool call — this folder is what proves that rule actually holds, instead of just being a comment in a system prompt.
