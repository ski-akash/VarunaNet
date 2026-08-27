# ai-eval/

Evaluation for the LLM layer specifically — separate from the model benchmarks in `benchmarks/`, because "is the chatbot any good" is a different question from "is the segmentation model any good."

This folder holds:
- A golden set of at least 50 questions (spanning all four AI features) with known-correct answers.
- Scoring code for tool-selection accuracy, parameter-extraction accuracy, answer correctness, and groundedness rate (what fraction of numbers the model states can actually be traced back to a tool call).
- Adversarial test cases: questions about regions with no data, ambiguous date ranges, and prompt-injection attempts typed into the chat box.
- Cost and latency reporting per query, with and without the semantic cache.

The project's hard rule is that the LLM can never state a number it didn't get from a tool call — this folder is what proves that rule actually holds, instead of just being a comment in a system prompt.

## Built so far

**`src/groundedness.ts`** — the actual enforcement mechanism spec section 6.1 calls for:
"an automated test suite asserts that numeric spans in model output appear in the tool
results for that turn." Built ahead of the agent loop (blocked on an LLM provider key
this environment doesn't have) because the check itself needs no real model response —
only text and tool results, both of which tests supply directly, adversarial cases
included.

- `extractNumericClaims(text)` — pulls every standalone number out of a response
  (decimals, percentages, comma-grouped thousands). Masks out ISO-8601 timestamps and
  this project's chip ids (`India_900498`) *before* extraction, not just at each
  match's boundary — a timestamp's colon-separated segments (`08:49:38`) each look like
  a standalone number on their own, so boundary-only exclusion still lets fragments of
  an already-excluded span through.
- `checkGroundedness(responseText, toolResults)` — literal-appearance check, matching
  the spec's own wording: a claim is grounded only if its digits occur verbatim in the
  turn's tool results (JSON-stringified), not merely "something close." A model saying
  "about 1250 hectares" when a tool returned `1251.4` fails this check — imprecise
  rounding presented as a measured figure is exactly the failure mode this rule exists
  to catch, not something to wave through on a fuzzy-match technicality.
- **Real bug this surfaced while writing tests, not eyeballing the regex**: the first
  version of the number pattern only matched runs of ≤3 digits or comma-grouped
  thousands (`1,251`), so a plain `5000` (no comma) silently extracted as zero numeric
  claims — the adversarial "fabricated 5000 hectares" test would have quietly reported
  the claim as fully grounded (vacuously, since it was never even seen), the opposite of
  what a groundedness checker exists to catch. Fixed by allowing an unrestricted-length
  digit run before the optional comma-triplet suffix.

11 tests passing (`npm test`, needs no external services).

**Not built yet:** the golden set of ≥50 questions, tool-selection/parameter-extraction
accuracy scoring, adversarial prompt-injection cases, and cost/latency reporting — all
of Phase 6 proper, waiting on an LLM provider key and the agent loop itself
(`api/src/tools/registry.ts` has the provider-agnostic tool implementations this will
score against).
