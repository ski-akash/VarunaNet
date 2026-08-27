// Flood-severity color scale (spec section 7: "sequential color scale,"
// "never encode severity by hue alone" -- colorblind safety in
// particular). Green (safest) -> yellow (mid) -> dark red (most
// dangerous) -- ColorBrewer's standard "RdYlGn" diverging palette, 8
// stops, reversed so green anchors the safe end.
//
// This project's first version deliberately used blue instead of green at
// the safe end, specifically because pure red-green scales collapse to
// nearly the same color under red-green color blindness (deuteranopia/
// protanopia, ~1 in 12 men). Reverted to green here at the user's
// explicit request, once blue became the map's river color and a
// blue-anchored severity scale started reading as "this district is a
// river" rather than "this district is safe". The yellow midpoint is kept
// (not a flat two-color green-to-red) specifically to preserve some of
// that original accessibility intent -- a three-hue progression with a
// real luminosity change gives more to go on than a pure two-color jump
// would, even though it's not as robust as the blue-anchored version was.
// The legend's own "Safe"/"Danger" text labels are the actual guarantee
// against hue-alone encoding, per spec section 7, regardless of palette.
const SEVERITY_COLOR_STOPS: readonly string[] = [
  '#006837', // safest
  '#1a9850',
  '#66bd63',
  '#a6d96a',
  '#ffffbf', // midpoint
  '#fdae61',
  '#f46d43',
  '#a50026', // most dangerous
]

function hexToRgb(hex: string): [number, number, number] {
  const value = Number.parseInt(hex.slice(1), 16)
  return [(value >> 16) & 0xff, (value >> 8) & 0xff, value & 0xff]
}

function rgbToHex(rgb: [number, number, number]): string {
  return '#' + rgb.map((channel) => Math.round(channel).toString(16).padStart(2, '0')).join('')
}

/**
 * Maps a severity value in [0, 1] (0 = safest, 1 = most dangerous) to a
 * hex color, linearly interpolated across SEVERITY_COLOR_STOPS. Values
 * outside [0, 1] are clamped rather than extrapolated -- a severity
 * score is expected to already be normalized before reaching this
 * function.
 */
export function severityColor(value: number): string {
  const clamped = Math.min(1, Math.max(0, value))
  const segments = SEVERITY_COLOR_STOPS.length - 1
  const position = clamped * segments
  const lowerIndex = Math.min(Math.floor(position), segments - 1)
  const t = position - lowerIndex

  const lower = hexToRgb(SEVERITY_COLOR_STOPS[lowerIndex])
  const upper = hexToRgb(SEVERITY_COLOR_STOPS[lowerIndex + 1])
  const interpolated: [number, number, number] = [
    lower[0] + (upper[0] - lower[0]) * t,
    lower[1] + (upper[1] - lower[1]) * t,
    lower[2] + (upper[2] - lower[2]) * t,
  ]
  return rgbToHex(interpolated)
}

export { SEVERITY_COLOR_STOPS }
