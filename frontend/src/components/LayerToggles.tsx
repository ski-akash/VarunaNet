// Spec section 7's layer list: flood extent, permanent water, confidence
// heatmap, SAR backscatter. Each one needs real model/pipeline output to
// be a genuine toggle rather than a decoration -- "Flood extent" now has
// one (the vectorized Otsu+HAND polygons data/build_assam_statewide.py
// produces), so it's wired to a real map layer below. The other three
// still have no real output (no separate permanent-water export, no
// per-pixel confidence from a classical baseline, no exported backscatter
// raster), so they stay visibly disabled rather than built to look
// functional over nothing -- the same principle spec section 6.1 applies
// to the AI layer applies here: a control that can't be toggled is the
// honest way to represent "this exists as a concept, not as data".
//
// The state outline and district boundaries used to be toggleable here.
// They aren't any more: they are the entire map, and a control whose only
// purpose is to hide the subject of the page isn't a real choice.
interface LayerToggle {
  id: string
  label: string
}

const REAL_LAYERS: LayerToggle[] = [{ id: 'flood-extent', label: 'Flood extent' }]

const UPCOMING_LAYERS = ['Permanent water', 'Confidence heatmap', 'SAR backscatter'] as const

interface LayerTogglesProps {
  checked: Record<string, boolean>
  onToggle: (id: string) => void
}

export default function LayerToggles({ checked, onToggle }: LayerTogglesProps) {
  return (
    <div className="layer-toggles">
      <span className="layer-toggles-heading">Layers</span>
      {REAL_LAYERS.map((layer) => (
        <label key={layer.id} className="layer-toggle">
          <input
            type="checkbox"
            checked={checked[layer.id] ?? false}
            onChange={() => onToggle(layer.id)}
          />
          {layer.label}
        </label>
      ))}
      {UPCOMING_LAYERS.map((name) => (
        <label key={name} className="layer-toggle layer-toggle-disabled" title="No data yet">
          <input type="checkbox" checked={false} disabled />
          {name}
        </label>
      ))}
    </div>
  )
}
