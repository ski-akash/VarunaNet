import { useEffect, useState } from 'react'
import { fetchAssamFloodDemo, type AssamFloodDistrictFull } from '../lib/assamFloodDemo'

interface ReportViewerProps {
  stateName: string
  onClose: () => void
}

// The exact fields spec section 6.4 asks a real situation report to
// contain. "Flooded area" now has a real source for districts a
// statewide build actually covers (data/build_assam_statewide.py +
// data/add_district_breakdown.py) -- see the effect below. The rest still
// need something that doesn't exist yet: population/infrastructure need
// intersecting flood polygons against WorldPop/GHSL and OSM data;
// confidence needs a model's own uncertainty estimate, which the
// classical Otsu+HAND baseline doesn't produce; change-over-time needs a
// second dated pass to compare against, which no district has yet. Shown
// as visible, labeled gaps rather than invented plausible-looking numbers
// -- a fabricated "2,340 people affected" would be exactly the kind of
// ungrounded claim spec section 6.1's hard rule refuses to allow the AI
// layer to state, and a report is not a different kind of claim just
// because a human reads it instead of an LLM saying it out loud.
const PLACEHOLDER_FIELDS = [
  { label: 'Change since previous pass', note: 'requires a second dated pass to compare against' },
  { label: 'Population affected', note: 'requires flood polygons intersected with WorldPop/GHSL' },
  { label: 'Infrastructure affected', note: 'requires flood polygons intersected with OSM roads/settlements' },
  { label: 'Model confidence', note: "requires the model's own uncertainty estimate" },
] as const

export default function ReportViewer({ stateName, onClose }: ReportViewerProps) {
  const [district, setDistrict] = useState<AssamFloodDistrictFull | null | undefined>(undefined)

  useEffect(() => {
    const controller = new AbortController()
    fetchAssamFloodDemo(controller.signal).then((demo) => {
      const match = demo?.districts?.find((d) => d.name === stateName)
      setDistrict(match && match.tiles_covering > 0 ? match : null)
    })
    return () => controller.abort()
  }, [stateName])

  return (
    <div className="report-viewer-backdrop" onClick={onClose}>
      <div className="report-viewer" onClick={(event) => event.stopPropagation()}>
        <div className="report-viewer-header">
          <div>
            <h2>Situation report — {stateName}</h2>
            <span className="report-viewer-subtitle">
              {district === undefined ? 'Loading…' : district ? 'Partially generated' : 'Not yet generated'}
            </span>
          </div>
          <button type="button" className="report-viewer-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="report-viewer-body">
          <div className="report-field">
            <span className="report-field-label">Flooded area</span>
            <span className="report-field-value">
              {district === undefined
                ? 'Loading…'
                : district
                  ? `${district.flooded_hectares.toLocaleString()} ha (${district.flooded_percent}% of district area)`
                  : 'No data yet — requires Sentinel-1 coverage over this district'}
            </span>
          </div>
          {PLACEHOLDER_FIELDS.map((field) => (
            <div key={field.label} className="report-field">
              <span className="report-field-label">{field.label}</span>
              <span className="report-field-value">No data yet — {field.note}</span>
            </div>
          ))}
        </div>

        <div className="report-viewer-footer">
          <button type="button" disabled>
            Export PDF
          </button>
        </div>
      </div>
    </div>
  )
}
