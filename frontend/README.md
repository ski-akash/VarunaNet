# frontend/

The React + Vite + TypeScript dashboard — the only part of this project a non-technical person would actually look at.

This folder holds:
- The map view (MapLibre GL), showing an India state/district choropleth colored by flood severity, with layer toggles (flood extent, permanent water, confidence, raw SAR backscatter).
- A time slider to scrub across satellite passes and watch flood extent change.
- The chat panel that talks to the API gateway's agent loop, including showing which tools fired for a given answer (that trace is a deliberate feature, not debug output — it's what makes the grounding claim visible).
- The report viewer, with PDF export of generated situation reports.

Colors are chosen to be colorblind-safe on purpose — severity is never encoded by hue alone.

## Backend wiring

`src/lib/apiClient.ts` is the one place this app talks to the Node gateway (`api/`) from
— everything else should go through it rather than calling `fetch()` directly. Reads the
gateway's base URL from `VITE_API_BASE_URL` (see `.env.example`), defaulting to
`http://localhost:3000`.

`src/components/BackendStatus.tsx` is the first real (non-placeholder) connection to that
gateway: a small status dot in the header, polling `GET /health` every 15s. Deliberately
shows only reachability, not any flood figures — no scene has ever been processed for
Assam, so a flooded-area number here would be fabricated, the same reasoning
`ReportViewer`'s placeholders already use. Verified live end to end (gateway → Redis →
Postgres → the Python inference service serving a real, if untrained, exported model on
a real staged Sen1Floods11 chip — see `inference/README.md`'s `stage_demo_scene.py`), not
just against a mock: this is the piece everything else (real report data, the chat panel)
builds on next.
