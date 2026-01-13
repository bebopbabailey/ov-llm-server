# Repository Guidelines

## Project Structure & Module Organization
This repository is a small FastAPI server for OpenVINO GenAI.
- `main.py` holds the server implementation and API handlers.
- `pyproject.toml` defines Python 3.12 settings and dependencies (managed by `uv`).
- `DEV_CONTRACT.md`, `REFERENCE.md`, and `TASKS.md` document constraints, API shapes, and planned work.
- `scripts/` holds local automation like `ov-convert-model`.
- No dedicated `tests/` directory exists yet.

## Build, Test, and Development Commands
Use `uv` for all Python workflow tasks.
- `uv run uvicorn main:app --host 0.0.0.0 --port 9000` starts the API server on port `9000`.
- `uv add openvino-genai fastapi uvicorn huggingface-hub numpy` installs required libs.
- `hf download Qwen/Qwen2.5-3B-Instruct --local-dir ~/models/Qwen2.5-3B-Instruct` fetches the model outside the repo.
- `hf auth login` authenticates if the model requires access.
- `uv venv .venv-convert` creates a conversion-only environment.
- `uv pip install --python .venv-convert "optimum[openvino]" transformers==4.49.0 sentencepiece tiktoken` installs conversion tools.
- `./scripts/ov-convert-model` converts all local models under `~/models`, writes OpenVINO IR to `~/models/converted_models`, and moves originals to `~/models/og_models`.
  It prompts once per model for a custom name; blank uses a slugged folder name and auto-suffixes if needed.
- `make install` installs `ov-convert-model` to `~/.local/bin`.
- `ov-warm-models` warms one or more models via the API to keep them loaded.
There is no build step beyond installing dependencies.

## Coding Style & Naming Conventions
- Indentation: 4 spaces (Python).
- Prefer type hints and Pydantic models for request/response payloads.
- Naming: `snake_case` for variables and functions, `PascalCase` for classes.
- Keep API paths OpenAI-compatible (e.g., `/v1/chat/completions`).

## Testing Guidelines
No test framework is configured yet. If you add tests, propose `pytest` and place files under `tests/` with names like `test_api.py`. Include at least one request/response shape validation for `tools` handling.

## Commit & Pull Request Guidelines
There is no commit history yet. If you begin committing, use clear, imperative summaries (e.g., `Add OpenAI-compatible chat endpoint`). For pull requests, include:
- A short problem/solution description.
- Any new commands or config changes.
- Example request/response payloads when API behavior changes.

## Operational Constraints
- Do not touch or restart the existing `ollama` service.
- Do not install system drivers or use global `pip`.
- Use `openvino-genai` (not `optimum-intel` or `openvino.runtime`) for generation.

## Delivery Process (Agile)
- Update `TASKS.md` and `AGENTS.md` before implementing new features to keep planning current.
- Implement in small, testable increments: plan → update docs → build → verify → update docs.
- Do not update `TASKS.md` during debugging sessions unless explicitly asked; track fixes in chat and update after.
- Before proposing changes, confirm they align with the latest project documentation as of today (current session date).

## Planned Work
- Model registry with per-model metadata and atomic updates.
- Reload-on-miss behavior in the server for new models.
- Install target to put `ov-convert-model` on PATH.
- Basic UX improvements: `/health`, `/v1/models`, and runtime env var docs.
- Systemd ergonomics: env file, single worker, lazy loading defaults.
- Logging QOL: startup config, registry count, model load timing, request latency.
- Converter robustness: skip converted or incomplete model dirs with clear warnings.

## Service Ergonomics
- Prefer a systemd env file, single worker, and lazy model loading to keep resource use predictable.

## Deployment
- Use a systemd service with env file at `/etc/homelab-llm/ov-server.env`.

## Conversion Metadata (Planned)
- Each converted model will include `conversion.json` with the chosen name, paths, task, and weight format.

## Model Registry Format (Planned)
- Location: `~/models/converted_models/registry.json`
- Shape: `{ "version": 1, "models": { "<slug>": { "path": "...", "task": "...", "weight_format": "..." }}}`
