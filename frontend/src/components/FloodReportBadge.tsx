import { useEffect, useRef, useState } from 'react'
import { CURRENT_FLOOD_REPORT } from '../lib/currentFloodReports'

// Source/attribution for the reported-flood-affected districts highlighted
// on the map. Was previously an always-open red box sitting on top of the
// map; that's now a small circled "?" that only reveals the source/date
// (and the unmatched-districts caveat) on click, so the map itself isn't
// permanently obstructed by an attribution note.
export default function FloodReportBadge() {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) {
      return
    }
    function handlePointerDown(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [open])

  return (
    <div className="flood-source-info" ref={containerRef}>
      <button
        type="button"
        className="flood-source-info-button"
        aria-label="Flood data source and last modified date"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
      >
        i
      </button>
      {open && (
        <div className="flood-source-popover" role="dialog">
          <span>
            Reported flood-affected districts — from news reports, not SAR-measured extent.
          </span>
          <span className="flood-source-popover-meta">
            Last modified: {CURRENT_FLOOD_REPORT.asOf} ·{' '}
            <a href={CURRENT_FLOOD_REPORT.sourceUrl} target="_blank" rel="noreferrer">
              Source
            </a>
          </span>
          {CURRENT_FLOOD_REPORT.unmatchedDistricts.length > 0 && (
            <span className="flood-source-popover-note">
              Also reported affected but not shown (created after the 2011 census boundaries this
              map uses): {CURRENT_FLOOD_REPORT.unmatchedDistricts.join(', ')}.
            </span>
          )}
        </div>
      )}
    </div>
  )
}
