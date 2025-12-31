import json
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import openvino_genai as ov_genai
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


REGISTRY_PATH = os.path.expanduser(
    os.environ.get("OV_REGISTRY_PATH", "~/models/converted_models/registry.json")
)
MODEL_PATH = os.path.expanduser(
    os.environ.get(
        "OV_MODEL_PATH",
        "~/models/converted_models/qwen2-5-3b-instruct/task-text-generation-with-past__wf-fp32",
    )
)
DEVICE = os.environ.get("OV_DEVICE", "GPU")

app = FastAPI()
_PIPELINES: Dict[str, ov_genai.LLMPipeline] = {}
_REGISTRY: Dict[str, Dict[str, Any]] = {}


def load_registry() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("models", {}) if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


_REGISTRY = load_registry()


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 1024


def get_pipeline(model_name: str) -> ov_genai.LLMPipeline:
    if model_name in _PIPELINES:
        return _PIPELINES[model_name]

    model_entry = _REGISTRY.get(model_name)
    model_path = model_entry.get("path") if model_entry else None
    if not model_path:
        model_path = MODEL_PATH

    pipeline = ov_genai.LLMPipeline(model_path, DEVICE)
    _PIPELINES[model_name] = pipeline
    return pipeline


def build_prompt(
    model_name: str, messages: List[ChatMessage], tools: Optional[List[Dict[str, Any]]]
) -> str:
    pipe = get_pipeline(model_name)
    tokenizer = pipe.get_tokenizer()
    payload = [{"role": m.role, "content": m.content or ""} for m in messages]
    if tools:
        payload.insert(
            0,
            {
                "role": "system",
                "content": (
                    "You can call tools. When you want to call a tool, respond with a "
                    'JSON object like {"name":"tool_name","arguments":"{...}"} and no '
                    "extra text."
                ),
            },
        )
    return tokenizer.apply_chat_template(
        payload, add_generation_prompt=True, chat_template=""
    )


def build_config(req: ChatCompletionRequest) -> ov_genai.GenerationConfig:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = req.max_tokens
    config.do_sample = True
    config.temperature = req.temperature
    return config


def extract_tool_call(
    text: str, tools: Optional[List[Dict[str, Any]]]
) -> Optional[ToolCall]:
    if not tools:
        return None
    try:
        obj = json.loads(text.strip())
        name = obj.get("name")
        arguments = obj.get("arguments")
        if isinstance(name, str) and isinstance(arguments, str):
            return ToolCall(
                id="call_0", function=FunctionCall(name=name, arguments=arguments)
            )
    except json.JSONDecodeError:
        pass
    tool = tools[0]
    function = tool.get("function", {})
    name = function.get("name") or tool.get("name") or "tool"
    return ToolCall(
        id="call_0",
        function=FunctionCall(name=name, arguments="{}"),
    )


def generate_text(prompt: str, config: ov_genai.GenerationConfig, model_name: str) -> str:
    pipe = get_pipeline(model_name)
    parts: List[str] = []
    done = threading.Event()
    token_queue: "queue.Queue[Optional[str]]" = queue.Queue()

    def callback(subword: str) -> bool:
        token_queue.put(subword)
        return False

    def run_generate() -> None:
        try:
            pipe.generate(prompt, config, callback)
        finally:
            done.set()
            token_queue.put(None)

    thread = threading.Thread(target=run_generate, daemon=True)
    thread.start()

    while not done.is_set():
        token = token_queue.get()
        if token is None:
            break
        parts.append(token)
    return "".join(parts)


def stream_tokens(prompt: str, config: ov_genai.GenerationConfig, model_name: str):
    pipe = get_pipeline(model_name)
    token_queue: "queue.Queue[Optional[str]]" = queue.Queue()

    def callback(subword: str) -> bool:
        token_queue.put(subword)
        return False

    def run_generate() -> None:
        try:
            pipe.generate(prompt, config, callback)
        finally:
            token_queue.put(None)

    thread = threading.Thread(target=run_generate, daemon=True)
    thread.start()
    while True:
        token = token_queue.get()
        if token is None:
            break
        yield token


def stream_chat_response(
    text_stream, model_name: str, tool_call: Optional[ToolCall]
):
    created = int(time.time())
    yield (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}}],
            }
        )
        + "\n\n"
    )

    if tool_call:
        yield (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments,
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            )
            + "\n\n"
        )
    else:
        for token in text_stream:
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": {"content": token}}
                        ],
                    }
                )
                + "\n\n"
            )
        yield (
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            + "\n\n"
        )
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    model_name = request.model
    if model_name not in _REGISTRY and REGISTRY_PATH:
        _REGISTRY.update(load_registry())
        print(f"Reloaded registry for missing model: {model_name}")
    prompt = build_prompt(model_name, request.messages, request.tools)
    config = build_config(request)
    created = int(time.time())

    if request.stream:
        if request.tools:
            text = generate_text(prompt, config, model_name)
            tool_call = extract_tool_call(text, request.tools)
            return StreamingResponse(
                stream_chat_response([], model_name, tool_call),
                media_type="text/event-stream",
            )
        return StreamingResponse(
            stream_chat_response(stream_tokens(prompt, config, model_name), model_name, None),
            media_type="text/event-stream",
        )

    text = generate_text(prompt, config, model_name)
    tool_call = extract_tool_call(text, request.tools)
    message: Dict[str, Any]
    finish_reason = "stop"
    if tool_call:
        message = {"role": "assistant", "content": None, "tool_calls": [tool_call.dict()]}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}

    return JSONResponse(
        {
            "id": "chatcmpl-0",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def main():
    print("Run with: uv run uvicorn main:app --host 0.0.0.0 --port 9000")


if __name__ == "__main__":
    main()
