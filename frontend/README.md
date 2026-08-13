# frontend/

The React + Vite + TypeScript dashboard — the only part of this project a non-technical person would actually look at.

This folder holds:
- The map view (MapLibre GL), showing an India state/district choropleth colored by flood severity, with layer toggles (flood extent, permanent water, confidence, raw SAR backscatter).
- A time slider to scrub across satellite passes and watch flood extent change.
- The chat panel that talks to the API gateway's agent loop, including showing which tools fired for a given answer (that trace is a deliberate feature, not debug output — it's what makes the grounding claim visible).
- The report viewer, with PDF export of generated situation reports.

Colors are chosen to be colorblind-safe on purpose — severity is never encoded by hue alone.
