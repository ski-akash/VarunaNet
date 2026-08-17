// Flood-severity color scale (spec section 7: "sequential color scale,"
// "never encode severity by hue alone" -- colorblind safety in
// particular). Blue (safest) -> pale yellow (mid) -> dark red (most
// dangerous), not the red-to-green scale that was the initial instinct:
// red and green collapse to nearly the same color for red-green color
// blindness (deuteranopia/protanopia, ~1 in 12 men), which would make
// the two most important ends of the scale indistinguishable for a real
// share of users. Blue-to-red keeps both ends clearly separated for
// every common type of color vision, including full grayscale -- this
// exact 8-stop family (a ColorBrewer-style "YlOrRd" scale with a blue
// anchor added at the safe end) is a standard, well-tested choice for
// exactly this kind of severity map.
//
// Deliberately not wired to any real state/district yet: there is no
// trained model output to color by (spec section 6.1's grounding
// principle applies here as much as it does to the AI layer -- don't
// visually imply severity data that isn't real). This is ready
// infrastructure, not a currently-active choropleth.
const SEVERITY_COLOR_STOPS: readonly string[] = [
  '#2c7fb8', // safest
  '#41b6c4',
  '#a1dab4',
  '#ffffb2', // midpoint
  '#fecc5c',
  '#fd8d3c',
  '#e31a1c',
  '#7f0000', // most dangerous
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
