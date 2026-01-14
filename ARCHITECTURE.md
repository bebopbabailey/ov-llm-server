# ov-llm-server Architecture

## Purpose
Local OpenVINO GenAI backend that serves OpenAI-compatible chat completions.
This service is a lightweight specialist behind the system gateway.

## Core Components
- **FastAPI server** (`main.py`)
  - Exposes `/v1/chat/completions`, `/v1/models`, `/health`
- **Registry** (`~/models/converted_models/registry.json`)
  - Maps model name → OpenVINO IR path + metadata
  - Enables lazy loading and stable model identifiers
- **OpenVINO GenAI pipeline**
  - Loads models on demand
  - Handles tokenization + generation

## Request Flow (local)s
1) Request hits `/v1/chat/completions`.
2) Model name resolves via registry (fallback to `OV_MODEL_PATH`).
3) Pipeline loads (first request) or reuses the cached model.
4) Response returned in OpenAI-compatible format.

## Model Conversion Flow
1) `ov-convert-model` converts source models to OpenVINO IR.
2) Converter writes `conversion.json` and updates registry atomically.
3) Service loads by registry key at request time.

## Constraints
- Single worker to avoid duplicate loads.
- Do not use `optimum-intel` or `openvino.runtime` for generation.
