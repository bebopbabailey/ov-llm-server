# DEV_CONTRACT.md
## Project: Bare-Metal OpenVINO LLM Server (Agentic)

### 1. System Context
* **Hardware:** Mac mini (Intel Core i7 6-Core, 64GB RAM).
* **Accelerator:** Integrated Intel Graphics (iGPU - Coffee Lake).
* **OS:** Ubuntu 24.04 Server (Headless/SSH).
* **Existing Services:** OLLAMA (running on port 11434, must be preserved).

### 2. Critical Constraints
* **OLLAMA Preservation:** Do NOT touch/restart/remove the existing 'ollama' service.
* **Dependency Management:** Use 'uv' for all Python management. No global pip.
* **Driver Layer:** Drivers are ALREADY INSTALLED. Do not try to apt install anything.
* **Library Strategy:** Use 'openvino-genai' (Native Python). Do NOT use C++ binaries.
* **Doc Consistency:** Before proposing changes, confirm they align with the latest project documentation as of today (current session date).

### 3. Technical Stack
* **Language:** Python 3.12
* **Inference Engine:** 'openvino_genai.LLMPipeline'
* **Server Framework:** 'FastAPI' + 'Uvicorn'
* **API Standard:** OpenAI-Compatible (v1/chat/completions with 'tools' support).

### 4. Definition of Done
The system is operational when:
1. 'uv run uvicorn main:app --host 0.0.0.0 --port 9000' starts a server on port 9000.
2. The endpoint accepts a JSON request with 'tools' and returns a valid OpenAI 'tool_calls' response.
