import MapView from './components/MapView'
import './App.css'

// Top-level dashboard shell (spec section 7): a header bar plus the map
// view for now. The chat panel, time slider, layer toggles, and report
// viewer are separate follow-up tasks -- this piece is specifically
// "does a real MapLibre map render inside a React app shell", verified
// in a browser before anything else gets built on top of it.
function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>VarunaNet</h1>
        <span className="app-subtitle">SAR flood detection &amp; situational awareness</span>
      </header>
      <main className="app-main">
        <MapView />
      </main>
    </div>
  )
}

export default App
