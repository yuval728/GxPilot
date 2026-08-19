"""vLLM engine wrapper."""
import os
import asyncio
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


@dataclass
class GenerationResponse:
    request_id: str
    text: str
    finish_reason: str
    usage: dict
    created: int


@dataclass
class StreamChunk:
    text: str
    finish_reason: Optional[str]


class VLLMEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.engine = None

    async def start(self):
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            dtype="auto",
            trust_remote_code=True,
            enforce_eager=False,  # Use CUDA graphs for speed
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print(f"vLLM engine started with model: {self.model_path}")

    async def shutdown(self):
        if self.engine:
            # vLLM handles cleanup on process exit
            pass

    async def generate(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> GenerationResponse:
        # Apply chat template
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )

        request_id = f"req-{os.urandom(8).hex()}"
        results = await self.engine.generate(prompt, sampling_params, request_id)

        final_output = results[-1]
        return GenerationResponse(
            request_id=request_id,
            text=final_output.outputs[0].text,
            finish_reason=final_output.outputs[0].finish_reason,
            usage={
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
                "total_tokens": len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids),
            },
            created=int(asyncio.get_event_loop().time()),
        )

    async def stream_generate(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )

        request_id = f"req-{os.urandom(8).hex()}"
        async for output in self.engine.generate_stream(prompt, sampling_params, request_id):
            text = output.outputs[0].text
            # Only yield new text (incremental)
            if hasattr(self, '_last_text'):
                new_text = text[len(self._last_text):]
            else:
                new_text = text
                self._last_text = ""
            self._last_text = text

            if new_text:
                yield StreamChunk(text=new_text, finish_reason=None)

        yield StreamChunk(text="", finish_reason="stop")
        self._last_text = ""