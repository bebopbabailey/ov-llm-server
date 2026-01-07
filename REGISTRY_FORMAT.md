# Registry Format

The model registry is stored outside the repo at:
`~/models/converted_models/registry.json`

## Schema
The file is JSON with a version and a models map:
```json
{
  "version": 1,
  "models": {
    "qwen2-5-3b-instruct": {
      "path": "/home/christopherbailey/models/converted_models/qwen2-5-3b-instruct/task-text-generation-with-past__wf-fp16",
      "task": "text-generation-with-past",
      "weight_format": "fp16"
    }
  }
}
```

## Notes
- Keys under `models` are slugged names chosen during conversion.
- `path` points to the converted OpenVINO model directory.
- Extra metadata can be added later without breaking the format.
