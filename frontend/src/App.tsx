import { useState } from 'react'
import ChatPanel from './components/ChatPanel'
import MapView from './components/MapView'
import ReportViewer from './components/ReportViewer'
import TimeSlider from './components/TimeSlider'
import './App.css'

// Top-level dashboard shell (spec section 7): a header bar, the map view
// with a time slider beneath it, the chat panel docked beside both, and
// a report viewer triggered from whichever state is currently selected.
function App() {
  const [selectedState, setSelectedState] = useState<string | null>(null)
  const [isReportOpen, setReportOpen] = useState(false)

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>VarunaNet</h1>
        <span className="app-subtitle">SAR flood detection &amp; situational awareness</span>
        <button
          type="button"
          className="generate-report-button"
          disabled={selectedState === null}
          onClick={() => setReportOpen(true)}
        >
          {selectedState === null ? 'Select a state for a report' : `Report: ${selectedState}`}
        </button>
      </header>
      <main className="app-main">
        <div className="map-column">
          <MapView onSelectionChange={setSelectedState} />
          <TimeSlider />
        </div>
        <ChatPanel />
      </main>
      {isReportOpen && selectedState !== null && (
        <ReportViewer stateName={selectedState} onClose={() => setReportOpen(false)} />
      )}
    </div>
  )
}

export default App
