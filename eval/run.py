"""Full evaluation harness orchestrator."""
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .metrics import compute_all_metrics, DEVIATION_HEADERS, CAPA_HEADERS
from .llm_judge import evaluate_with_judge, DEFAULT_RUBRIC
from .adversarial import evaluate_adversarial, print_adversarial_report


@dataclass
class EvalResult:
    model_name: str
    split: str
    n_examples: int
    metrics: Dict
    judge_scores: Dict
    adversarial: Dict


def load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
    device_map: str = "auto",
):
    """Load model with optional LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages: List[Dict], max_new_tokens: int = 512) -> str:
    """Generate a single response using chat template."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def load_split(path: str) -> List[Dict]:
    """Load JSONL split with metadata."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def run_split(
    model,
    tokenizer,
    split_data: List[Dict],
    split_name: str,
    judge_model: str = "gpt-4o-mini",
) -> EvalResult:
    """Run evaluation on a single split."""
    predictions = []
    references = []
    categories = []

    for ex in split_data:
        messages = ex["messages"][:2]  # system + user
        ref = ex["messages"][2]["content"]
        pred = generate_response(model, tokenizer, messages)
        predictions.append(pred)
        references.append(ref)
        categories.append(ex.get("category", "unknown"))

    # Compute metrics per example
    all_metrics = []
    for pred, ref, cat in zip(predictions, references, categories):
        headers = DEVIATION_HEADERS if cat == "deviation" else \
                  CAPA_HEADERS if cat == "capa" else []
        m = compute_all_metrics(pred, ref, headers if headers else None)
        all_metrics.append(m)

    # Aggregate metrics
    agg_metrics = {}
    for key in all_metrics[0].keys():
        if isinstance(all_metrics[0][key], dict):
            agg_metrics[key] = {k: sum(m[key][k] for m in all_metrics) / len(all_metrics)
                                for k in all_metrics[0][key]}
        else:
            agg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

    # LLM judge evaluation
    judge_data = [{"user": ex["messages"][1]["content"],
                   "assistant": pred,
                   "reference": ref,
                   "category": cat} for ex, pred, ref, cat in
                  zip(split_data, predictions, references, categories)]
    judge_scores = evaluate_with_judge(judge_model, judge_data, DEFAULT_RUBRIC)

    # Adversarial evaluation (if split contains adversarial)
    adv_results = []
    for ex, pred in zip(split_data, predictions):
        if ex.get("is_adversarial"):
            adv_results.append({"response": pred, "is_adversarial": True})
    adversarial = evaluate_adversarial(adv_results).__dict__ if adv_results else {}

    return EvalResult(
        model_name=model.config._name_or_path if hasattr(model, 'config') else "unknown",
        split=split_name,
        n_examples=len(split_data),
        metrics=agg_metrics,
        judge_scores=judge_scores,
        adversarial=adversarial,
    )


def run_full_eval(
    model_path: str,
    adapter_path: Optional[str] = None,
    data_dir: str = "data",
    judge_model: str = "gpt-4o-mini",
    output_dir: str = "eval_results",
) -> Dict[str, EvalResult]:
    """Run full evaluation on all splits."""
    Path(output_dir).mkdir(exist_ok=True)

    model, tokenizer = load_model(model_path, adapter_path)

    splits = {
        "eval": f"{data_dir}/eval.jsonl",
        "adversarial_holdout": f"{data_dir}/adversarial_holdout.jsonl",
    }

    results = {}
    for name, path in splits.items():
        print(f"\nEvaluating {name}...")
        split_data = load_split(path)
        result = run_split(model, tokenizer, split_data, name, judge_model)
        results[name] = result

        # Save individual result
        with open(f"{output_dir}/{name}_results.json", "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        print(f"  Metrics: {result.metrics}")
        if result.adversarial:
            print_adversarial_report(eval_adversarial([]).__class__(**result.adversarial))

    # Save combined results
    with open(f"{output_dir}/full_results.json", "w") as f:
        json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--output-dir", default="eval_results")
    args = parser.parse_args()

    run_full_eval(args.model, args.adapter, args.data_dir, args.judge_model, args.output_dir)