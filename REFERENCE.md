# REFERENCE.md
## 1. OpenVINO GenAI Syntax (Python)
*Critical: Do not use 'optimum-intel' or 'openvino.runtime' for generation.*
*Use the native 'openvino_genai' bindings as shown below.*

\`\`\`python
import openvino_genai as ov_genai

# Initialization (GPU with CPU Fallback)
# Note: "GPU" targets the Intel iGPU (Neo drivers)
pipe = ov_genai.LLMPipeline("./model", "GPU")

# Generation Config
config = ov_genai.GenerationConfig()
config.max_new_tokens = 1024
config.do_sample = True
config.temperature = 0.7

# Tokenization & Chat Templates
# The pipeline handles tokenization, but for an API server, we usually
# apply the template manually to get the prompt string first.
tokenizer = pipe.get_tokenizer()
prompt = tokenizer.apply_chat_template(
messages,
add_generation_prompt=True,
chat_template="" # Use default from tokenizer_config.json
)

# Streaming Inference
def stream_generator():
streamer = pipe.get_streamer()
# Generation happens in a separate thread usually, or via callback
# For simple implementation, use the callback-based generation:
def streamer_callback(subword: str) -> bool:
print(subword, end="", flush=True)
return False # Continue generation

    pipe.generate(prompt, config, streamer_callback)
\`\`\`

## 2. Pydantic Schemas for OpenASho
*Use this structure to handle the 'tools' parameter in FastAPI.*

\`\`\`python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

class FunctionCall(BaseModel):
name: str
arguments: str  # JSON string

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
\`\`\`