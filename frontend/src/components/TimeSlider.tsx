// Scrubs across satellite passes to animate flood progression (spec
// section 7). Disabled rather than wired to sample dates on purpose:
// any dates shown here would be fabricated, since no scene has actually
// been run through the inference pipeline yet (Phase 5) -- there are no
// real satellite pass timestamps to scrub through. Showing invented
// dates on a *time* control specifically risks looking like real
// acquisition history at a glance, which is exactly the kind of
// visually-implied-but-not-real data spec section 6.1's grounding rule
// refuses to allow in the AI layer; the same standard applies here.
export default function TimeSlider() {
  return (
    <div className="time-slider">
      <button type="button" className="time-slider-play" disabled aria-label="Play">
        ▶
      </button>
      <input
        type="range"
        className="time-slider-track"
        min={0}
        max={0}
        value={0}
        disabled
        aria-label="Satellite pass (not yet available)"
      />
      <span className="time-slider-label">No satellite passes processed yet</span>
    </div>
  )
}
