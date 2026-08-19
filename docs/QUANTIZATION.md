# Quantization Documentation

## Overview
The `kaggle/quantize.ipynb` and `quantize/fp8_quantize.py` implement **post-training quantization** of the merged 16-bit fine-tuned model. Three methods:

| Method | Tool | Hardware | Use Case |
|--------|------|----------|----------|
| **GPTQ** | auto-gptq | Kaggle T4/P100 | General-purpose, mature |
| **AWQ** | autoawq | Kaggle T4/P100 | Better accuracy at 4-bit |
| **FP8** | llm-compressor | Modal H100 | Native FP8 on Hopper+ |

## GPTQ (Generalized Post-Training Quantization)

### Theory
- One-shot quantization using calibration data
- Per-channel scaling with group-wise quantization (group_size=128)
- `desc_act=False` (faster, slightly less accurate) or `True` (slower, better)

### Code (`kaggle/quantize.ipynb`)
```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=True,
)

model = AutoGPTQForCausalLM.from_pretrained(
    MODEL_PATH,
    quantize_config=quantize_config,
    device_map="auto",
)

# Calibration data (128 samples from train)
calib_texts = [format_prompt(ex) for ex in train_data[:128]]
model.quantize(calib_texts)
model.save_quantized("./quantized/gptq-4bit")
```

### Key Parameters
| Param | Value | Effect |
|-------|-------|--------|
| `bits` | 4 | 4-bit quantization |
| `group_size` | 128 | Weight groups per scale (smaller = more accurate, larger model) |
| `desc_act` | False | Activation order (True = better perplexity, slower) |
| `sym` | True | Symmetric quantization |

## AWQ (Activation-aware Weight Quantization)

### Theory
- Protects salient weights (important for activation) during quantization
- Uses calibration data to identify important weight channels
- Generally better accuracy than GPTQ at same bit-width

### Code (`kaggle/quantize.ipynb`)
```python
from awq import AutoAWQForCausalLM

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

model = AutoAWQForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)
model.save_quantized("./quantized/awq-4bit")
```

### Key Parameters
| Param | Value | Effect |
|-------|-------|--------|
| `w_bit` | 4 | 4-bit weights |
| `q_group_size` | 128 | Quantization group size |
| `zero_point` | True | Asymmetric quantization (better for activations) |
| `version` | "GEMM" | Kernel optimization |

## FP8 (Float8) — Modal H100 Only

### Theory
- Native 8-bit floating point (E4M3 or E5M2 format)
- Requires Hopper (H100) or newer GPU
- No calibration needed (dynamic per-tensor scaling)
- Near-FP16 accuracy with 2x memory bandwidth

### Code (`quantize/fp8_quantize.py`)
```python
# Modal deployment
@app.function(gpu="H100", timeout=3600)
def quantize_fp8():
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )

    oneshot(
        model=MODEL_ID,
        dataset="open_platypus",
        num_calibration_samples=512,
        recipe=recipe,
        output_dir="/results/fp8",
    )
```

### Deployment
```bash
# Deploy to Modal
modal deploy quantize/fp8_quantize.py

# Run quantization
modal run quantize/fp8_quantize.py

# Upload to HF Hub (optional)
HF_TOKEN=xxx modal run quantize/fp8_quantize.py::upload_model
```

## Evaluation Protocol

**Critical**: Re-run the **exact same eval harness** on every quantized variant.

```bash
# GPTQ
python -m eval.run --model ./quantized/gptq-4bit --data-dir data --output-dir eval_gptq

# AWQ
python -m eval.run --model ./quantized/awq-4bit --data-dir data --output-dir eval_awq

# FP8 (after Modal download)
python -m eval.run --model ./fp8_model --data-dir data --output-dir eval_fp8
```

## Comparison Metrics

| Metric | How Measured |
|--------|--------------|
| **Perplexity** | `eval_loss` on eval.jsonl (lower = better) |
| **Task Quality** | Exact match, ROUGE-L, LLM judge scores (same as fine-tuned) |
| **Adversarial** | Refusal/false compliance rates (must not degrade) |
| **VRAM** | `torch.cuda.max_memory_allocated()` during inference |
| **Latency** | Batch=1, 128 tokens generated (TTFT + inter-token) |
| **Throughput** | Tokens/sec at batch=1, 8, 32 |

## Expected Results Table

| Model | Perplexity | Exact Match | ROUGE-L | Judge Acc | VRAM (GB) | Latency (s) |
|-------|------------|-------------|---------|-----------|-----------|-------------|
| Base (4-bit) | — | — | — | — | — | — |
| Fine-tuned (FP16) | — | — | — | — | ~14 | — |
| Fine-tuned (4-bit NF4) | — | — | — | — | ~5 | — |
| **GPTQ 4-bit** | — | — | — | — | ~4 | — |
| **AWQ 4-bit** | — | — | — | — | ~4 | — |
| **FP8** | — | — | — | — | ~8 | — |

*Run eval harness to populate.*

## Artifact Storage

### W&B Artifacts
```python
artifact = wandb.Artifact("qwen2.5-7b-gxp-gptq-4bit", type="model")
artifact.add_dir("./quantized/gptq-4bit")
wandb.log_artifact(artifact)
```

### Modal Volume
```python
volumes={"/results": modal.Volume.from_name("gxp-results")}
# Model saved to /results/fp8/
```

### Hugging Face Hub (Optional)
```python
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path="./quantized/gptq-4bit",
    repo_id="your-username/qwen2.5-7b-gxp-gptq",
)
```

## Serving Quantized Models

### vLLM (GPTQ/AWQ/FP8)
```python
# GPTQ
engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="./quantized/gptq-4bit",
    quantization="gptq",
    ...
))

# AWQ
engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="./quantized/awq-4bit",
    quantization="awq",
    ...
))

# FP8
engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model="./fp8_model",
    quantization="fp8",
    ...
))
```

### SGLang
```python
runtime = sgl.Runtime(
    model_path="./quantized/awq-4bit",
    quantization="awq",
    ...
)
```

## Decision Guide

| Priority | Choose |
|----------|--------|
| Best accuracy at 4-bit | **AWQ** |
| Fastest quantization | **GPTQ** (no activation-aware search) |
| H100 available, want native FP8 | **FP8** |
| No H100, need serving now | **AWQ** or **GPTQ** |
| Maximum compatibility | **GPTQ** (widest tool support) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| GPTQ: "No GPU found" | Ensure `device_map="auto"` and CUDA visible |
| AWQ: "Kernel not found" | Install `autoawq` with `--no-build-isolation` or use pre-built wheel |
| FP8: "Unsupported dtype" | Requires H100 (compute capability 9.0+) |
| Perplexity spike | Increase calibration samples to 512, try `desc_act=True` for GPTQ |
| Serving fails | Verify `quantization` arg matches model format in vLLM/SGLang |