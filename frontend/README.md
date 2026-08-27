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
just against a mock: this was the piece everything else built on next.

## Chat (spec section 6, live)

`ChatPanel.tsx` is a real, working chat now, not a placeholder — it calls the gateway's
`POST /chat` (`sendChatMessage` in `apiClient.ts`), which runs the real tool-calling
agent loop (`api/`'s README covers the agent/tool/grounding side in full). Both natural-
language query (feature 1) and conversational map control (feature 3, via the
`set_map_view` tool) work through this one input.

- **Tool-call traces are shown inline**, collapsed by default (`<details>`) — spec
  section 7 calls this out specifically ("showing the tools firing is a strong demo
  moment"), and it's also the visible half of spec section 6.1's grounding rule: when
  the gateway flags a response as not fully grounded (a numeric claim it couldn't trace
  to a tool result this turn), that's shown as a warning next to the trace, not hidden.
- **`ChatUnavailableError`** (`apiClient.ts`) distinguishes "the gateway has no LLM key
  configured" (a 404 on `/chat`, shown as a plain informational message) from an actual
  request failure (shown as an error message) — the same distinguish-reachability-from-
  content-failure reasoning `BackendStatus` already uses for `/health`.
- Verified live in a real browser against the real stack: asking "which district is
  worst affected?" correctly and honestly reports the demo scene's real (untrained-model)
  numbers rather than inventing a flood; asking to "zoom to Golaghat and show the flood
  extent layer" correctly triggers `set_map_view` and shows it in the trace.
- **Not yet wired**: the `set_map_view` tool call result isn't applied to the actual map
  yet (no `onSelectionChange`-style callback from `ChatPanel` into `MapView`) — the
  trace shows the model proposed a real view change, but the map itself doesn't move.
  That plumbing is the natural next step once this piece is reviewed.
