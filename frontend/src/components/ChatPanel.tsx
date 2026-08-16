// Docked beside the map (spec section 7), talking to the API gateway's
// tool-calling agent loop once that exists (spec section 6) -- neither
// the gateway (api/) nor the AI layer (Phase 6) is built yet, so this is
// real layout, not real functionality. A disabled input and an empty
// message area show the actual intended structure honestly, rather than
// faking a working chat or leaving the space unaccounted for until
// Phase 6 -- so that work slots into this dock later instead of needing
// a layout change to make room for it.
export default function ChatPanel() {
  return (
    <aside className="chat-panel">
      <div className="chat-panel-header">
        <h2>Ask VarunaNet</h2>
      </div>
      <div className="chat-panel-messages">
        <p className="chat-panel-placeholder-text">
          Chat isn&apos;t connected yet — it needs the API gateway and tool-calling agent
          (spec Phase 6), neither of which exist yet. This dock is where it'll live once they do.
        </p>
      </div>
      <div className="chat-panel-input-row">
        <input
          type="text"
          placeholder="Which districts are worst hit this week?"
          disabled
          aria-label="Chat input (not yet connected)"
        />
        <button type="button" disabled>
          Send
        </button>
      </div>
    </aside>
  )
}
