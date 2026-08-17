export interface LayerToggleState {
  countryBorders: boolean
  stateBorders: boolean
  districtBorders: boolean
}

interface LayerTogglesProps {
  value: LayerToggleState
  onChange: (next: LayerToggleState) => void
}

// Spec section 7's full layer list: flood extent, permanent water,
// confidence heatmap, SAR backscatter, admin boundaries. Only "admin
// boundaries" (split here into state/district borders, since they're
// independently useful) is backed by real data right now -- the other
// four need actual model output that doesn't exist yet (Phase 5). They're
// listed and visibly disabled rather than omitted, so the panel shows the
// real planned structure, but a disabled control that can't be toggled is
// the honest way to represent "this exists as a concept, not as data" --
// building working-looking toggles for data that isn't real would imply
// something is there when it isn't, the same principle spec section 6.1
// applies to the AI layer's grounding rule.
const UPCOMING_LAYERS = [
  'Flood extent',
  'Permanent water',
  'Confidence heatmap',
  'SAR backscatter',
] as const

export default function LayerToggles({ value, onChange }: LayerTogglesProps) {
  return (
    <div className="layer-toggles">
      <span className="layer-toggles-heading">Layers</span>
      <label className="layer-toggle">
        <input
          type="checkbox"
          checked={value.countryBorders}
          onChange={(event) => onChange({ ...value, countryBorders: event.target.checked })}
        />
        Country boundaries
      </label>
      <label className="layer-toggle">
        <input
          type="checkbox"
          checked={value.stateBorders}
          onChange={(event) => onChange({ ...value, stateBorders: event.target.checked })}
        />
        State boundaries
      </label>
      <label className="layer-toggle">
        <input
          type="checkbox"
          checked={value.districtBorders}
          onChange={(event) => onChange({ ...value, districtBorders: event.target.checked })}
        />
        District boundaries
      </label>
      {UPCOMING_LAYERS.map((name) => (
        <label key={name} className="layer-toggle layer-toggle-disabled" title="No data yet">
          <input type="checkbox" checked={false} disabled />
          {name}
        </label>
      ))}
    </div>
  )
}
