# AGENTS — ov-llm-server

## Role & Scope
You are the maintainer for **this service only**. Do not modify other services or
system-wide settings unless explicitly requested.

## Non‑Negotiables (Hard Constraints)
- **Do not touch** the existing `ollama` service (port 11434).
- **Dependency management:** use `uv` only; no global `pip`.
- **Drivers:** do not install system drivers.
- **Inference library:** use `openvino-genai` for generation.
- **Ports:** keep the service on port `9000`.

## Start Here (Required Reading)
- `SERVICE_SPEC.md` — ports, env vars, endpoints
- `RUNBOOK.md` — systemd, health checks, logs
- `ARCHITECTURE.md` — internal flow + registry
- `docs/REFERENCE.md` — API and conversion details

## Expected Behavior
- Maintain OpenAI‑compatible endpoints (`/v1/chat/completions`, `/v1/models`, `/health`).
- Keep registry‑backed model loading intact.
- Preserve lazy loading and single‑worker behavior unless explicitly changed.

## Working Rules
- Update **SERVICE_SPEC.md** and **RUNBOOK.md** when runtime behavior changes.
- Update **docs/REFERENCE.md** when API or conversion details change.
- Keep changes small and reversible.

## Tests / Verification
- `curl -fsS http://127.0.0.1:9000/health | jq .`
- `curl -fsS http://127.0.0.1:9000/v1/models | jq .`
- `curl http://localhost:9000/v1/chat/completions -H "Content-Type: application/json" -d @docs/examples/request.json`

## Definition of Done
- Server starts with `uv run uvicorn main:app --host 0.0.0.0 --port 9000`.
- `/v1/chat/completions` returns a valid OpenAI‑compatible response (tools supported).
- Docs updated for any behavior‑level change.
