# TASKS.md
## Phase 1: Environment Setup (COMPLETED)
- [x] **1.1 Install Drivers:** (DONE - System sees Intel UHD Graphics 630).
- [x] **1.2 Verify Driver:** (DONE).

## Phase 2: Python Dependencies (COMPLETED)
- [x] **2.1 Install Libraries:**
    Use 'uv add' to install:
    - 'openvino-genai'
    - 'fastapi'
    - 'uvicorn'
    - 'huggingface-hub'
    - 'numpy'

## Phase 3: Model Management
- [x] **3.1 Download Model:**
    Use 'hf download Qwen/Qwen2.5-3B-Instruct --local-dir ~/models/Qwen2.5-3B-Instruct'.
- [x] **3.2 Convert Model (OpenVINO IR):**
    Use './scripts/ov-convert-model' to convert all local models under '~/models'.
- [x] **3.3 Naming Preflight:**
    Prompt once per model and slugify defaults when no custom name is provided.
- [x] **3.4 Converter Robustness:**
    Skip converted or incomplete model dirs (missing `config.json`) with a warning.
- [x] **3.5 Standardize Model Root:**
    Use `~/models` for downloads, originals (`~/models/og_models`), and converted outputs (`~/models/converted_models`).

## Phase 4: Server Implementation
- [x] **4.1 Create Server:**
    Write 'main.py' implementing an OpenAI-compatible /v1/chat/completions endpoint.
    - Must support 'stream=True'.
    - Must support 'tools' parameter.
    - Must map 'GPU' device in 'LLMPipeline'.

## Phase 5: Model Registry & Auto-Mapping
- [x] **5.1 Define Registry Format:**
    Define `registry.json` in `~/models/converted_models` with `version` and `models` map.
- [x] **5.2 Update Converter to Write Registry:**
    Extend 'ov-convert-model' to write per-model `conversion.json` and update `registry.json` atomically.
- [x] **5.3 Add Reload-on-Miss in Server:**
    Load the registry at startup and reload it when a requested model is missing.

Ye## Phase 6: CLI Install Convenience
- [x] **6.1 Add Install Target:**
    Provide a simple install target to place 'ov-convert-model' on PATH.

## Phase 7: UX & Ergonomics (Planned)
- [x] **7.1 Add Health Endpoint:**
    Provide `/health` for quick status checks (status and registry count).
- [x] **7.2 Add Models Endpoint:**
    Provide `/v1/models` backed by the registry.
- [x] **7.3 Document Runtime Env Vars:**
    Explain `OV_MODEL_PATH`, `OV_REGISTRY_PATH`, and `OV_DEVICE` defaults.
- [x] **7.4 Provide Sample Request File:**
    Add `docs/examples/request.json` example for curl usage.
- [x] **7.5 Document Service Ergonomics:**
    Recommend systemd env file, single worker, and lazy loading defaults.
- [x] **7.6 Refresh Registry Updates:**
    Add a registry timestamp check to pick up updated models.
- [x] **7.7 Clear Cached Pipelines on Update:**
    Ensure updated registry entries replace cached pipelines for matching names.
- [x] **7.8 Add Logging QOL:**
    Log startup config, registry load count, model load timing, and per-request latency.
- [x] **7.9 Add Warm-Up Command:**
    Provide `ov-warm-models` to preload models via the API.
- [x] **7.10 Add Loaded Models Status:**
    Expose a quick status endpoint to list models currently loaded in memory.

## Phase 8: Deployment
- [x] **8.1 Create Service:** Generate `ov-server.service` and env file, then enable with `systemctl`.

## Future Projects
- [ ] **F1 Choose Embedding Model:** Decide on a quality English embedding model (e.g., `intfloat/e5-large-v2` vs `BAAI/bge-base-en-v1.5`) and document the selection.
- [ ] **F2 Embedding Pipeline:** Implement a batch embedding + indexing workflow for daily runs.
- [ ] **F3 Nightly Automation:** Add a scheduled job (systemd timer or cron) to stop the chat service, run embeddings, then restart it.
- [ ] **F4 Vector Database Deployment:** Decide where to host the vector DB (local vs separate machine) and document the setup.

## Nice-to-Have
- [ ] **N1 Improve `/health` Registry Count:**
    Report on-disk registry count instead of in-memory count.
