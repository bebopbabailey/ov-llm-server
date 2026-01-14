yea# Service Spec: ov-llm-server

## Purpose
OpenAI-compatible chat backend for lightweight OpenVINO models on the Mac Mini.
Designed to act as a specialist node behind a proxy (e.g., LiteLLM).

## Host & Runtime
- **Host**: Mac mini (Intel i7, 64 GB RAM), Ubuntu 24.04
- **GPU**: Intel iGPU (OpenVINO `GPU` device)
- **Language/Framework**: Python 3.12, FastAPI + Uvicorn
- **Inference**: `openvino_genai.LLMPipeline`

## Endpoints
- `POST /v1/chat/completions` (OpenAI-compatible; tools supported; stream supported)
- `GET /health` (status, device, registry count, loaded models)
- `GET /v1/models` (registry-backed model list)

## Configuration (Env Vars)
- `OV_MODEL_PATH` fallback model path
- `OV_REGISTRY_PATH` registry file path (default `~/models/converted_models/registry.json`)
- `OV_DEVICE` (default `GPU`)
- `OV_LOG_LEVEL` (default `INFO`)
Note: runtime defaults to GPU and uses int8 for `benny-clean-*` via LiteLLM routing.
fp16 variants remain in the registry. int4 is unstable on GPU on this iGPU stack;
CPU-only int4 is possible but lower fidelity.

## Model Registry
Registry file: `~/models/converted_models/registry.json`
Format: `{ "version": 1, "models": { "<name>": { "path": "...", "task": "...", "weight_format": "..." }}}`

## Conversion Tooling
- CLI: `ov-convert-model`
- Source models: `~/models`
- Converted models: `~/models/converted_models/<name>/task-...__wf-...`
- Originals moved to: `~/models/og_models`
- Writes `conversion.json` per model and updates `registry.json` atomically

## Logging
- Startup config logged (device, registry path, fallback model)
- Model load timing logged
- Per-request completion log includes request id, model, client IP, latency

## Service Management (Systemd)
- Unit: `/etc/systemd/system/ov-server.service`
- Env file: `/etc/homelab-llm/ov-server.env` (runtime; `services/ov-llm-server/ov-server.env` is a template)

## Interop Notes (for LiteLLM)
- This service is a backend only; it does not provide a frontend UI.
- Use LiteLLM as the single “front door” and route model names to this backend.
