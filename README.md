# ov-llm-server

Bare-metal OpenVINO GenAI server targeting an OpenAI-compatible chat API.

## Requirements
- Python 3.12
- `uv` for dependency management
- OpenVINO GenAI runtime (Python package, no system drivers to install)

## Quick Start
1) Install dependencies:
```
uv add openvino-genai fastapi uvicorn huggingface-hub numpy
```
2) Download a model outside the repo (recommended):
```
hf auth login
hf download Qwen/Qwen2.5-3B-Instruct --local-dir ~/model/Qwen2.5-3B-Instruct
```
3) Convert models to OpenVINO IR (use a separate conversion env):
```
uv venv .venv-convert
uv pip install --python .venv-convert "optimum[openvino]" sentencepiece tiktoken
./scripts/ov-convert-model
```
You will be prompted once per model for a custom name; leaving it blank uses a slugged version
of the source folder name.
Optional: install the converter on your PATH:
```
make install
```
4) Run the server:
```
uv run uvicorn main:app --host 0.0.0.0 --port 9000
```
If your model path differs, set it explicitly:
```
OV_MODEL_PATH=~/models/converted_models/qwen2-5-3b-instruct/task-text-generation-with-past__wf-fp32 \
  uv run uvicorn main:app --host 0.0.0.0 --port 9000
```

## Conversion Metadata
Each converted model gets a `conversion.json` in its output folder:
```json
{
  "name": "qwen2-5-3b-instruct",
  "source_path": "/home/christopherbailey/model/Qwen2.5-3B-Instruct",
  "original_path": "/home/christopherbailey/model/og_models/Qwen2.5-3B-Instruct",
  "converted_path": "/home/christopherbailey/models/converted_models/qwen2-5-3b-instruct/task-text-generation-with-past__wf-fp32",
  "task": "text-generation-with-past",
  "weight_format": "fp32",
  "converted_at": "2025-12-30T10:05:00+00:00"
}
```

The registry file lives at `~/models/converted_models/registry.json`:
```json
{
  "version": 1,
  "models": {
    "qwen2-5-3b-instruct": {
      "path": "/home/christopherbailey/models/converted_models/qwen2-5-3b-instruct/task-text-generation-with-past__wf-fp32",
      "task": "text-generation-with-past",
      "weight_format": "fp32"
    }
  }
}
```

## Remote Conversion via SSH
From another machine on your network, you can trigger conversions remotely:
```
ssh christopherbailey@192.168.1.71 "cd ~/ov-llm-server && ./scripts/ov-convert-model"
```
If you want to override the model root or weight format:
```
ssh christopherbailey@192.168.1.71 "OV_MODEL_SRC=~/model OV_MODEL_OUT=~/models/converted_models OV_WEIGHT_FORMAT=int8 cd ~/ov-llm-server && ./scripts/ov-convert-model"
```

## API
The server exposes an OpenAI-compatible endpoint:
- `POST /v1/chat/completions`

Example request:
```json
{
  "model": "qwen",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain what OpenVINO is in one sentence."}
  ],
  "tools": [],
  "stream": false,
  "temperature": 0.7
}
```
Note: the current server uses `OV_MODEL_PATH` for selection and echoes the `model` field back.
If a registry entry exists for the requested `model`, the server will load it instead.

Example curl:
```
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @request.json
```

## Project Structure
- `main.py` contains the FastAPI server implementation.
- `pyproject.toml` defines runtime dependencies (managed by `uv`).
- `DEV_CONTRACT.md`, `REFERENCE.md`, `TASKS.md` document constraints and tasks.

## Operational Constraints
- Do not touch or restart the existing `ollama` service.
- Do not install system drivers or use global `pip`.
- Use `openvino-genai` (not `optimum-intel` or `openvino.runtime`) for generation.
