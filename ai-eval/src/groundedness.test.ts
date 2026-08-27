import { test } from "node:test";
import assert from "node:assert/strict";
import { extractNumericClaims, checkGroundedness } from "./groundedness.js";

const REAL_TOOL_RESULT = {
  tool: "get_worst_affected",
  queriedAt: "2026-08-27T08:49:38.000Z",
  data: [{ name: "Golaghat", flooded_hectares: 1251.4, flooded_percent: 0.37 }],
};

test("extractNumericClaims finds a plain decimal", () => {
  const claims = extractNumericClaims("Golaghat has 1251.4 flooded hectares.");
  assert.deepEqual(
    claims.map((c) => c.value),
    [1251.4],
  );
});

test("extractNumericClaims finds a percentage and strips the % for its value", () => {
  const claims = extractNumericClaims("0.37% of Golaghat is flooded.");
  assert.deepEqual(
    claims.map((c) => c.value),
    [0.37],
  );
  assert.equal(claims[0].text, "0.37%");
});

test("extractNumericClaims handles comma thousands grouping", () => {
  const claims = extractNumericClaims("Total flooded area: 1,251.4 hectares.");
  assert.deepEqual(
    claims.map((c) => c.value),
    [1251.4],
  );
});

test("extractNumericClaims does NOT treat an ISO date as a numeric claim", () => {
  const claims = extractNumericClaims("This scene was processed at 2026-08-27T08:49:38Z.");
  assert.deepEqual(claims, []);
});

test("extractNumericClaims does NOT treat digits inside a chip id as a numeric claim", () => {
  const claims = extractNumericClaims("The scene India_900498 was processed.");
  assert.deepEqual(claims, []);
});

test("checkGroundedness: a real figure from the actual tool result is grounded", () => {
  const result = checkGroundedness(
    "Golaghat has 1251.4 flooded hectares (0.37% of the district).",
    [REAL_TOOL_RESULT],
  );

  assert.equal(result.groundednessRate, 1);
  assert.deepEqual(result.grounded, [true, true]);
});

test("checkGroundedness: a fabricated figure is caught, not waved through", () => {
  // The tool result says 1251.4 -- a model claiming 5000 hectares said
  // something the tool never produced, which is exactly the failure mode
  // spec section 6.1 exists to prevent.
  const result = checkGroundedness("Golaghat has 5000 flooded hectares.", [REAL_TOOL_RESULT]);

  assert.equal(result.groundednessRate, 0);
  assert.deepEqual(result.grounded, [false]);
});

test("checkGroundedness: an imprecise paraphrase of a real number fails, not passes on a near-match", () => {
  // "about 1250" is not what the tool said (1251.4) -- rounding off a
  // figure and reporting the rounded value as if it were the measured one
  // is exactly the kind of confident-sounding imprecision this check
  // should refuse to launder as grounded.
  const result = checkGroundedness("Golaghat has about 1250 flooded hectares.", [REAL_TOOL_RESULT]);

  assert.equal(result.groundednessRate, 0);
});

test("checkGroundedness: prose with no numeric claims is trivially fully grounded", () => {
  const result = checkGroundedness("Golaghat is affected by flooding.", [REAL_TOOL_RESULT]);

  assert.equal(result.claims.length, 0);
  assert.equal(result.groundednessRate, 1);
});

test("checkGroundedness: a mix of grounded and fabricated claims reports the real fraction", () => {
  const result = checkGroundedness(
    "Golaghat has 1251.4 flooded hectares, up from a fabricated 900 hectares last week.",
    [REAL_TOOL_RESULT],
  );

  assert.equal(result.claims.length, 2);
  assert.deepEqual(result.grounded, [true, false]);
  assert.equal(result.groundednessRate, 0.5);
});

test("checkGroundedness: an out-of-scope refusal with no tool results and no numbers is fully grounded", () => {
  const result = checkGroundedness(
    "I don't have data for that region -- I can't answer without a real query result.",
    [],
  );

  assert.equal(result.groundednessRate, 1);
});
