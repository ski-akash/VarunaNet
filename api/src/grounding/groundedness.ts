/**
 * Production copy of ai-eval/src/groundedness.ts's checker -- this is
 * where spec section 6.1's hard rule ("the LLM may never state a number
 * that did not come from a tool result") is actually *enforced* at
 * request time, not just tested in isolation. `ai-eval/` is the canonical
 * source and owns the test suite (11 tests, including the adversarial
 * cases); this file is a deliberate, documented duplication rather than a
 * cross-package import.
 *
 * Why duplicated instead of imported: api/ and ai-eval/ are separate,
 * independently-installable npm packages (no npm workspace links them),
 * matching how this repo's CI already runs each with its own `npm ci` in
 * its own directory. Restructuring into a workspace to de-duplicate one
 * ~90-line, zero-dependency module was judged not worth destabilizing
 * every already-working, already-tested package's install/CI setup for.
 * If this drifts from ai-eval's copy, keep them in sync by hand; a real
 * cross-package workspace is the honest fix if this file grows.
 */

export interface NumericClaim {
  text: string;
  value: number;
  index: number;
}

export interface GroundednessCheck {
  claims: NumericClaim[];
  grounded: boolean[];
  groundednessRate: number;
}

const ISO_TIMESTAMP_PATTERN = /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?/g;
const CHIP_ID_PATTERN = /\b[A-Za-z]+_\d+\b/g;

function maskNonClaimDigits(text: string): string {
  return text
    .replace(ISO_TIMESTAMP_PATTERN, (m) => " ".repeat(m.length))
    .replace(CHIP_ID_PATTERN, (m) => " ".repeat(m.length));
}

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
  const groundednessRate = claims.length === 0 ? 1 : grounded.filter(Boolean).length / claims.length;
  return { claims, grounded, groundednessRate };
}
