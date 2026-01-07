# Local Specialist Mesh with Single-Endpoint Front Door

## Goal
Provide a single OpenAI-compatible endpoint while enabling multiple models to collaborate
as a “board of specialists.” This Mac Mini runs a lightweight OpenVINO backend and serves
as one specialist node.

## Layer 1 — Front Door (Single Endpoint)
- **Service**: LiteLLM proxy
- **Role**: Unified OpenAI-compatible API for all clients (UI, agents, scripts)
- **Why**: Central routing point that hides backend complexity and supports multiple providers
- **Where**: Runs on this Mac Mini for simplicity (can move later)

## Layer 2 — Orchestration & Routing
- **Service**: tinyagents (planned), optional OptiLLM loops
- **Role**: Decide which specialist models to call, aggregate responses, refine prompts
- **Pattern**: Router-Coordinator
  - Router model picks the best specialist(s) for the request
  - Coordinator merges or refines results into one response
- **Why**: Mix fast/cheap models with slow/accurate ones without changing clients

## Layer 3 — Specialist Backends
- **Local specialist (this repo)**:
  - OpenVINO GenAI server on the Mac Mini
  - Lightweight models for classification, tool calls, quick instruction following
  - Optimized for low-resource, always-on use
- **Remote specialist (big box)**:
  - Heavy-inference models for deep reasoning and generation
  - Accessed through LiteLLM as another backend

## Layer 4 — Model Registry & Conversion
- **Registry**: `~/models/converted_models/registry.json`
  - Keeps local model names and paths stable
  - Supports lazy loading and reload-on-miss
- **Conversion**: `ov-convert-model`
  - Converts raw models to OpenVINO IR
  - Prompts for friendly names and auto-suffixes duplicates
  - Writes conversion metadata for traceability

## Data Flow
1. Client sends an OpenAI-compatible request to LiteLLM.
2. Router decides which backend model(s) to call.
3. LiteLLM forwards to the chosen backend:
   - Local OpenVINO server (this repo), or
   - Remote heavy-inference machine.
4. Coordinator returns a single response to the client.
5. Optional: OptiLLM or teacher/student refinement loops run between steps 2–4.

## Why This Is Durable
- Clear separation of routing vs inference.
- Single endpoint for clients; direct backends still available for one-offs.
- Easy to add/remove specialists without changing clients.
- Keeps the Mac Mini lightweight, reliable, and power-efficient.
