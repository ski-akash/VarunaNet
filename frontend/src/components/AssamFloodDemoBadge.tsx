import { useEffect, useRef, useState } from 'react'
import type { AssamFloodDemo } from '../lib/assamFloodDemo'

interface AssamFloodDemoBadgeProps {
  demo: AssamFloodDemo
}

// Provenance for the real SAR-derived district coloring on the map --
// same collapsed-by-default "?" pattern as FloodReportBadge, for the same
// reason: don't permanently obstruct the map with an attribution note,
// but make the source one click away, since spec section 6.1's grounding
// principle applies to what a map visually implies just as much as to
// what an LLM says out loud.
export default function AssamFloodDemoBadge({ demo }: AssamFloodDemoBadgeProps) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!open) return
    function handlePointerDown(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [open])

  const affectedNames = demo.worst_affected.map((d) => d.name).join(', ')

  return (
    <div className="flood-source-info assam-flood-demo-badge" ref={containerRef}>
      <button
        type="button"
        className="flood-source-info-button"
        aria-label="SAR flood coloring source and coverage"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
      >
        SAR
      </button>
      {open && (
        <div className="flood-source-popover" role="dialog">
          <span>
            District coloring for {affectedNames} is a real measurement: a Sentinel-1 pass over
            the 2020 Assam monsoon flood, scored with this project&apos;s own classical baseline
            (Otsu + terrain masking + permanent-water removal) -- not the trained CNN yet, and
            not full-state coverage.
          </span>
          <span className="flood-source-popover-meta">
            Scene {demo.scene_id} · processed {demo.processed_at}
          </span>
          <span className="flood-source-popover-note">
            Every other district has no data yet, not zero flooding -- this AOI is one small
            area (~16km), not the whole state.
          </span>
        </div>
      )}
    </div>
  )
}
