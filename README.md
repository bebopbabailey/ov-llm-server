# ov-llm-server

Bare-metal OpenVINO GenAI server targeting an OpenAI-compatible chat API.

## Quick Start
```bash
uv add openvino-genai fastapi uvicorn huggingface-hub numpy
uv run uvicorn main:app --host 0.0.0.0 --port 9000
```

If you need conversion, registry, or API details, see `docs/REFERENCE.md`.
Operational steps (systemd, logs, health) are in `RUNBOOK.md`.

Note: runtime defaults to GPU and uses int8 for `benny-clean-*` via LiteLLM routing.
fp16 variants remain in the registry; int4 is GPU-unstable on this iGPU stack and
only viable on CPU with reduced fidelity.

## Project Structure
- `main.py` contains the FastAPI server implementation.
- `pyproject.toml` defines runtime dependencies (managed by `uv`).
- `AGENTS.md`, `docs/REFERENCE.md`, `TASKS.md` document constraints and tasks.
- `docs/` contains reference material and examples.
- `RUNBOOK.md` covers systemd operations and health checks.

## Operational Constraints
- Do not touch or restart the existing `ollama` service.
- Do not install system drivers or use global `pip`.
- Use `openvino-genai` (not `optimum-intel` or `openvino.runtime`) for generation.
