import { useState } from 'react'
import BackendStatus from './components/BackendStatus'
import ChatPanel from './components/ChatPanel'
import MapView from './components/MapView'
import ReportViewer from './components/ReportViewer'
import TimeSlider from './components/TimeSlider'
import './App.css'

// Top-level dashboard shell (spec section 7): a header bar, the map view
// with a time slider beneath it, the chat panel docked beside both, and
// a report viewer triggered from whichever district is currently selected.
// The selection is a district, not a state: the map covers Assam only, so
// the state is fixed and picking one would be a choice of exactly one.
function App() {
  const [selectedDistrict, setSelectedDistrict] = useState<string | null>(null)
  const [isReportOpen, setReportOpen] = useState(false)

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>VarunaNet</h1>
        <span className="app-subtitle">SAR flood detection &amp; situational awareness</span>
        <BackendStatus />
        <button
          type="button"
          className="generate-report-button"
          disabled={selectedDistrict === null}
          onClick={() => setReportOpen(true)}
        >
          {selectedDistrict === null ? 'Select a district for a report' : `Report: ${selectedDistrict}`}
        </button>
      </header>
      <main className="app-main">
        <div className="map-column">
          <MapView onSelectionChange={setSelectedDistrict} />
          <TimeSlider />
        </div>
        <ChatPanel />
      </main>
      {isReportOpen && selectedDistrict !== null && (
        <ReportViewer stateName={selectedDistrict} onClose={() => setReportOpen(false)} />
      )}
    </div>
  )
}

export default App
