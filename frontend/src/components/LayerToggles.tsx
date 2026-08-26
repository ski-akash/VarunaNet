// Spec section 7's layer list: flood extent, permanent water, confidence
// heatmap, SAR backscatter. Each one needs real model output that doesn't
// exist yet (Phase 5), so all four are listed and visibly disabled rather
// than omitted -- the panel shows the real planned structure, and a
// disabled control that can't be toggled is the honest way to represent
// "this exists as a concept, not as data". Building working-looking
// toggles for data that isn't real would imply something is there when it
// isn't, the same principle spec section 6.1 applies to the AI layer.
//
// The state outline and district boundaries used to be toggleable here.
// They aren't any more: they are the entire map, and a control whose only
// purpose is to hide the subject of the page isn't a real choice.
const UPCOMING_LAYERS = [
  'Flood extent',
  'Permanent water',
  'Confidence heatmap',
  'SAR backscatter',
] as const

export default function LayerToggles() {
  return (
    <div className="layer-toggles">
      <span className="layer-toggles-heading">Layers</span>
      {UPCOMING_LAYERS.map((name) => (
        <label key={name} className="layer-toggle layer-toggle-disabled" title="No data yet">
          <input type="checkbox" checked={false} disabled />
          {name}
        </label>
      ))}
    </div>
  )
}
