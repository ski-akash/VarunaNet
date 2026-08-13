# api/

The Node.js (Fastify) gateway that sits between the frontend and everything else.

This folder holds:
- REST endpoints the frontend calls (flood stats, scene metadata, report generation, etc).
- The LLM tool-calling agent loop — the piece that lets the chat panel answer questions by calling real database queries instead of guessing numbers.
- The shared tool registry (e.g. `query_flood_stats`, `get_worst_affected`, `set_map_view`) that all four AI features use, so there's one integration instead of four separate ones.
- SSE (Server-Sent Events) streaming for the chat panel, so responses appear token-by-token.
- Auth (a simple API key) and rate limiting.

This is orchestration only — it doesn't run the model itself (that's `inference/`) and doesn't render anything (that's `frontend/`).
