import { SEVERITY_COLOR_STOPS } from '../lib/severityColor'

// Explains what the color scale means once it's actually driving the
// choropleth -- safe to show now even without real severity data, the
// same way the layer toggle panel shows its four not-yet-real layers:
// this labels a concept, it doesn't claim any specific state has any
// specific severity.
export default function SeverityLegend() {
  const gradient = `linear-gradient(to right, ${SEVERITY_COLOR_STOPS.join(', ')})`

  return (
    <div className="severity-legend">
      <span className="severity-legend-label">Flood severity</span>
      <div className="severity-legend-bar" style={{ background: gradient }} />
      <div className="severity-legend-ticks">
        <span>Safe</span>
        <span>Danger</span>
      </div>
    </div>
  )
}
