/**
 * The enforcement mechanism spec section 6.1 requires as a hard rule, not
 * an aspiration: "An automated test suite asserts that numeric spans in
 * model output appear in the tool results for that turn." This module is
 * that test suite's core logic -- independent of which LLM eventually
 * produces the text, since the check is purely: did every number the
 * model said trace back to something a tool actually returned this turn.
 *
 * Built ahead of the agent loop itself (blocked on an LLM provider key)
 * because the check doesn't need a real model response to prove out --
 * only text and tool results, both of which can be supplied directly in
 * tests, adversarial cases included.
 */

export interface NumericClaim {
  /** The exact substring matched in the response text, e.g. "1,251.4" or "37%". */
  text: string;
  /** Parsed numeric value, commas stripped, % sign stripped (not divided by 100 -- the raw digits are what must appear in the tool result, not a reinterpreted value). */
  value: number;
  index: number;
}

export interface GroundednessCheck {
  claims: NumericClaim[];
  /** One verdict per claim, same order as `claims`. */
  grounded: boolean[];
  groundednessRate: number;
}

// Chip/scene-id-shaped tokens (e.g. "India_900498") are masked OUT of
// numeric-claim extraction below, deliberately -- a real chip id appearing
// in real tool output is not a "numeric claim" to fuzzy-match, it's an
// identifier. That exemption has a real cost, caught live: a model can
// state a completely FABRICATED scene id and the numeric checker alone
// never flags it, since the id was never extracted as a claim at all.
// Checked separately here, literally (the id string itself must appear
// verbatim in the tool results), which is exactly why the exemption above
// doesn't reopen a hole -- a fabricated id fails this check even though it
// skips the numeric one.
const IDENTIFIER_PATTERN = /\b[A-Za-z]+_\d+\b/g;

export interface IdentifierClaim {
  text: string;
  index: number;
}

export function extractIdentifierClaims(text: string): IdentifierClaim[] {
  return Array.from(text.matchAll(IDENTIFIER_PATTERN), (m) => ({ text: m[0], index: m.index }));
}

// ISO-8601 timestamps ("2026-08-27T08:49:38.000Z") and this project's chip
// ids ("India_900498") both contain digit runs that are not numeric claims
// about a quantity. Masking them out before number extraction is more
// robust than trying to exclude them at each digit-run's boundary with
// lookaround -- a timestamp's colon- and dot-separated segments each look
// like a standalone number on their own (e.g. "49" in "08:49:38"), so
// boundary-only exclusion still lets fragments of a masked-out span
// through. Replaced with spaces, not deleted, so surrounding numbers keep
// their own word boundaries.
const ISO_TIMESTAMP_PATTERN = /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?/g;
const CHIP_ID_PATTERN = /\b[A-Za-z]+_\d+\b/g;

function maskNonClaimDigits(text: string): string {
  return text
    .replace(ISO_TIMESTAMP_PATTERN, (m) => " ".repeat(m.length))
    .replace(CHIP_ID_PATTERN, (m) => " ".repeat(m.length));
}

// Matches a standalone number: optional minus, an unrestricted-length
// digit run (covers both "5000" and, via the optional comma-triplet group,
// "5,000"), optional decimal part, optional trailing percent sign. The
// word-boundary lookaround still guards against matching digits embedded
// in an identifier that maskNonClaimDigits didn't already remove.
const NUMBER_PATTERN = /(?<![\w-])-?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\w-])/g;

export function extractNumericClaims(text: string): NumericClaim[] {
  const masked = maskNonClaimDigits(text);
  const claims: NumericClaim[] = [];
  for (const match of masked.matchAll(NUMBER_PATTERN)) {
    const raw = match[0];
    const numeric = raw.replace(/,/g, "").replace(/%$/, "");
    const value = Number(numeric);
    if (Number.isNaN(value)) continue;
    claims.push({ text: raw, value, index: match.index });
  }
  return claims;
}

/**
 * Literal-appearance check, matching spec section 6.1's own wording
 * ("numeric spans in model output appear in the tool results"): a claim
 * is grounded if its digits -- with or without the response's own comma
 * grouping, with or without a trailing "%" -- occur verbatim somewhere in
 * the JSON-stringified tool results for the turn. Deliberately stricter
 * than "some number close to this exists in the results somewhere": an
 * LLM that says "about 1250 hectares" when the tool returned 1251.4
 * should fail this check, not pass on a fuzzy-match technicality --
 * "about" is exactly the kind of imprecise paraphrase spec section 6.1
 * exists to catch, not wave through.
 */
export function isGrounded(claim: NumericClaim, toolResultsText: string): boolean {
  const bare = String(claim.value);
  const withCommas = claim.value.toLocaleString("en-US");
  return (
    toolResultsText.includes(bare) ||
    toolResultsText.includes(withCommas) ||
    toolResultsText.includes(claim.text)
  );
}

export function checkGroundedness(responseText: string, toolResults: unknown[]): GroundednessCheck {
  const toolResultsText = JSON.stringify(toolResults);
  const claims = extractNumericClaims(responseText);
  const grounded = claims.map((c) => isGrounded(c, toolResultsText));

  // Identifier claims (scene/chip ids) fold into the same overall rate --
  // a fabricated id is exactly as disqualifying as a fabricated number,
  // even though it's checked by literal substring match rather than the
  // numeric-value comparison isGrounded uses.
  const identifierClaims = extractIdentifierClaims(responseText);
  const identifierGrounded = identifierClaims.map((c) => toolResultsText.includes(c.text));

  const totalClaims = claims.length + identifierClaims.length;
  const totalGrounded = grounded.filter(Boolean).length + identifierGrounded.filter(Boolean).length;
  const groundednessRate = totalClaims === 0 ? 1 : totalGrounded / totalClaims;

  return { claims, grounded, groundednessRate };
}
