import { useState } from 'react'
import { sendChatMessage, ChatUnavailableError, type ToolCall } from '../lib/apiClient'

interface ChatMessage {
  role: 'user' | 'assistant' | 'error'
  text: string
  toolCalls?: ToolCall[]
  grounded?: boolean
}

// Docked beside the map (spec section 7), talking to the API gateway's
// tool-calling agent loop (spec section 6) -- real now, not a placeholder:
// the gateway, tool registry, and agent loop all exist and this calls the
// real POST /chat endpoint. Tool-call traces are shown inline per spec
// section 7 ("showing the tools firing is a strong demo moment") -- that's
// also the visible half of spec section 6.1's grounding rule: a response
// flagged ungrounded here is not hidden, it's shown with a warning.
export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)

  async function handleSend() {
    const message = input.trim()
    if (!message || isSending) return

    setMessages((prev) => [...prev, { role: 'user', text: message }])
    setInput('')
    setIsSending(true)

    try {
      const result = await sendChatMessage(message)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: result.response, toolCalls: result.tool_calls, grounded: result.grounded },
      ])
    } catch (err) {
      const text =
        err instanceof ChatUnavailableError
          ? err.message
          : `Something went wrong: ${err instanceof Error ? err.message : String(err)}`
      setMessages((prev) => [...prev, { role: 'error', text }])
    } finally {
      setIsSending(false)
    }
  }

  return (
    <aside className="chat-panel">
      <div className="chat-panel-header">
        <h2>Ask VarunaNet</h2>
      </div>
      <div className="chat-panel-messages">
        {messages.length === 0 && (
          <p className="chat-panel-placeholder-text">
            Ask about flood data for a processed scene -- e.g. &quot;which district is worst
            affected?&quot; or &quot;zoom to Golaghat and show the flood extent layer.&quot;
          </p>
        )}
        {messages.map((message, index) => (
          <div key={index} className={`chat-message chat-message--${message.role}`}>
            <p className="chat-message-text">{message.text}</p>
            {message.toolCalls && message.toolCalls.length > 0 && (
              <details className="chat-tool-trace">
                <summary>
                  {message.toolCalls.length} tool call{message.toolCalls.length > 1 ? 's' : ''}
                  {message.grounded === false && (
                    <span className="chat-grounding-warning"> -- unverified figures</span>
                  )}
                </summary>
                {message.toolCalls.map((call, callIndex) => (
                  <div key={callIndex} className="chat-tool-call">
                    <code>{call.name}({JSON.stringify(call.args)})</code>
                  </div>
                ))}
              </details>
            )}
          </div>
        ))}
        {isSending && <p className="chat-panel-placeholder-text">Thinking…</p>}
      </div>
      <div className="chat-panel-input-row">
        <input
          type="text"
          placeholder="Which districts are worst hit this week?"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') handleSend()
          }}
          disabled={isSending}
          aria-label="Chat input"
        />
        <button type="button" onClick={handleSend} disabled={isSending || input.trim().length === 0}>
          Send
        </button>
      </div>
    </aside>
  )
}
