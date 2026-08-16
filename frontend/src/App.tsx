import ChatPanel from './components/ChatPanel'
import MapView from './components/MapView'
import TimeSlider from './components/TimeSlider'
import './App.css'

// Top-level dashboard shell (spec section 7): a header bar, the map
// view with a time slider beneath it, and the chat panel docked beside
// both. The report viewer is a separate follow-up task.
function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>VarunaNet</h1>
        <span className="app-subtitle">SAR flood detection &amp; situational awareness</span>
      </header>
      <main className="app-main">
        <div className="map-column">
          <MapView />
          <TimeSlider />
        </div>
        <ChatPanel />
      </main>
    </div>
  )
}

export default App
