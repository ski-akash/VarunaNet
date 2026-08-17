interface ReportViewerProps {
  stateName: string
  onClose: () => void
}

// The exact fields spec section 6.4 asks a real situation report to
// contain, each requiring something that doesn't exist yet: flooded area
// needs a trained model's output; population/infrastructure need
// intersecting flood polygons against WorldPop/GHSL and OSM data;
// confidence needs the model's own uncertainty estimate. Shown as
// visible, labeled gaps rather than invented plausible-looking numbers --
// a fabricated "2,340 people affected" would be exactly the kind of
// ungrounded claim spec section 6.1's hard rule refuses to allow the AI
// layer to state, and a report is not a different kind of claim just
// because a human reads it instead of an LLM saying it out loud.
const REPORT_FIELDS = [
  { label: 'Flooded area', note: 'requires trained model output' },
  { label: 'Change since previous pass', note: 'requires a second dated pass to compare against' },
  { label: 'Population affected', note: 'requires flood polygons intersected with WorldPop/GHSL' },
  { label: 'Infrastructure affected', note: 'requires flood polygons intersected with OSM roads/settlements' },
  { label: 'Model confidence', note: "requires the model's own uncertainty estimate" },
] as const

export default function ReportViewer({ stateName, onClose }: ReportViewerProps) {
  return (
    <div className="report-viewer-backdrop" onClick={onClose}>
      <div className="report-viewer" onClick={(event) => event.stopPropagation()}>
        <div className="report-viewer-header">
          <div>
            <h2>Situation report — {stateName}</h2>
            <span className="report-viewer-subtitle">Not yet generated</span>
          </div>
          <button type="button" className="report-viewer-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="report-viewer-body">
          {REPORT_FIELDS.map((field) => (
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
