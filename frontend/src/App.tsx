import ChatPanel from './components/ChatPanel'
import MapView from './components/MapView'
import './App.css'

// Top-level dashboard shell (spec section 7): a header bar, the map
// view, and the chat panel docked beside it. The time slider and report
// viewer are separate follow-up tasks.
function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>VarunaNet</h1>
        <span className="app-subtitle">SAR flood detection &amp; situational awareness</span>
      </header>
      <main className="app-main">
        <MapView />
        <ChatPanel />
      </main>
    </div>
  )
}

export default App
