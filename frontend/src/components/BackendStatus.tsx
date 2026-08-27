import { useEffect, useState } from 'react'
import { getGatewayHealth } from '../lib/apiClient'

type Status = 'checking' | 'ok' | 'degraded' | 'unreachable'

// The first real (non-placeholder) connection between this frontend and
// api/ -- a small honest status dot rather than anything data-bearing,
// because that's the only thing currently true to show: no scene has ever
// been processed for Assam, so any flooded-area figure here would be
// fabricated (same reasoning ReportViewer's placeholders already use).
// "Is the backend actually up" is real and checkable right now, and this
// is the piece everything else (report data, the chat panel) will build on.
export default function BackendStatus() {
  const [status, setStatus] = useState<Status>('checking')
  const [modelName, setModelName] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    async function check() {
      try {
        const health = await getGatewayHealth(controller.signal)
        if (cancelled) return
        setStatus(health.status)
        setModelName(health.inference?.model ?? null)
      } catch {
        if (!cancelled) setStatus('unreachable')
      }
    }

    check()
    // Polled, not a one-off check on mount: the gateway/inference service
    // can go up or down independently of a page load, and a stale "ok"
    // shown forever would be worse than no status at all.
    const interval = setInterval(check, 15_000)
    return () => {
      cancelled = true
      controller.abort()
      clearInterval(interval)
    }
  }, [])

  const label =
    status === 'checking'
      ? 'Checking backend…'
      : status === 'ok'
        ? modelName
          ? `Backend online (${modelName})`
          : 'Backend online'
        : status === 'degraded'
          ? 'Backend online, inference service unreachable'
          : 'Backend unreachable'

  return (
    <div className="backend-status" title={label}>
      <span className={`backend-status-dot backend-status-dot--${status}`} aria-hidden="true" />
      <span className="backend-status-label">{label}</span>
    </div>
  )
}
