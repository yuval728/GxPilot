"""
Reusable synthetic data generation pipeline for the cEMS GxP fine-tuning
project. Calls the Anthropic API to generate additional examples beyond the
seed set, following the same schema and category mix.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python generate_synthetic_data.py --category deviation --count 20 --adversarial-ratio 0.15

Requires: pip install anthropic --break-system-packages
"""
import argparse
import json
import os
import time
import random

import anthropic
import numpy as np

CLIENT = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-5"  # check docs.claude.com for the current recommended model string

SYSTEM_PROMPT = (
    "You are a GxP-compliance assistant embedded in a continuous environmental "
    "monitoring system for a pharmaceutical manufacturing site. You draft, review, "
    "and answer questions about SOPs, deviation reports, CAPA plans, and audit trail "
    "records. You follow ALCOA+ data integrity principles, never fabricate data, "
    "never alter or delete audit trail entries, and flag out-of-scope requests."
)

CATEGORY_PROMPTS = {
    "sop": (
        "Generate one realistic training example for a GxP environmental monitoring "
        "system. Write a user request asking to draft, review, or answer a question "
        "about a Standard Operating Procedure (SOP) section related to sensor "
        "calibration, alarm response, excursion handling, or monitoring gaps. Then "
        "write the ideal assistant response: precise, compliant, procedural."
    ),
    "deviation": (
        "Generate one realistic deviation report training example. Write a user "
        "message describing a specific environmental monitoring incident (sensor "
        "excursion, equipment failure, data gap, etc.) with concrete numbers "
        "(timestamps, values, durations). Then write the ideal assistant response: "
        "a structured deviation report with Description, Immediate Action, Product "
        "Impact, Preliminary Classification, and Next Step."
    ),
    "capa": (
        "Generate one realistic CAPA (Corrective and Preventive Action) training "
        "example. Write a user message asking for a CAPA summary for a specific "
        "deviation, including a stated or implied root cause. Then write the ideal "
        "assistant response: a structured CAPA with Root Cause, Corrective Action, "
        "Preventive Action, Effectiveness Check, and Target Date. The corrective and "
        "preventive actions must be genuinely distinct, not restated versions of "
        "each other."
    ),
    "audit_qa": (
        "Generate one realistic audit trail Q&A training example for a GxP "
        "environmental monitoring system. Write a user question about audit trail "
        "policy, data retention, access control, or record integrity. Then write "
        "the ideal assistant response: accurate, grounded in ALCOA+ principles."
    ),
}

ADVERSARIAL_SUFFIX = (
    "\n\nIMPORTANT: Make this example ADVERSARIAL. The user message should contain "
    "one of: (a) a request to falsify, backdate, delete, or omit a record, (b) an "
    "ambiguous or underspecified instruction, (c) conflicting constraints, or (d) an "
    "out-of-scope request unrelated to GxP monitoring. The assistant response must "
    "correctly decline or redirect the problematic part while remaining helpful "
    "where possible — never comply with a data integrity violation."
)

OUTPUT_SCHEMA_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object in this exact format, no other text:\n"
    '{"user": "<user message text>", "assistant": "<assistant response text>"}'
)


def generate_one(category: str, adversarial: bool) -> dict:
    prompt = CATEGORY_PROMPTS[category]
    if adversarial:
        prompt += ADVERSARIAL_SUFFIX
    prompt += OUTPUT_SCHEMA_INSTRUCTION

    response = CLIENT.messages.create(
        model=MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # Strip markdown code fences if the model wraps the JSON despite instructions
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text.strip())
    return {
        "category": category,
        "is_adversarial": adversarial,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": parsed["user"]},
            {"role": "assistant", "content": parsed["assistant"]},
        ],
    }


def _embed_similarity_matrix(texts):
    """Sentence-transformers cosine similarity, falling back to TF-IDF if the
    model can't be downloaded (no Hugging Face access). Same logic as
    process_dataset.py — kept in sync so both stages use one dedup standard."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=False)
        norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return norm @ norm.T
    except Exception as e:
        print(f"WARNING: sentence-transformers unavailable ({type(e).__name__}: {e}). "
              "Falling back to TF-IDF cosine similarity for dedup.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf)


def dedup_against_existing(new_examples, existing_path="all_examples_with_metadata.jsonl",
                            threshold=0.85):
    """Drop any new example whose assistant response is near-duplicate of an
    existing one."""
    try:
        with open(existing_path) as f:
            existing = [json.loads(line) for line in f]
    except FileNotFoundError:
        existing = []

    existing_texts = [ex["messages"][-1]["content"] for ex in existing]
    new_texts = [ex["messages"][-1]["content"] for ex in new_examples]
    all_texts = existing_texts + new_texts
    if len(all_texts) < 2:
        return new_examples

    sim = _embed_similarity_matrix(all_texts)

    n_existing = len(existing_texts)
    keep = []
    for i, ex in enumerate(new_examples):
        idx = n_existing + i
        is_dup = any(sim[idx][j] > threshold for j in range(n_existing))
        is_dup = is_dup or any(
            sim[idx][n_existing + k] > threshold
            for k in range(i)
            if new_examples[k] in keep
        )
        if not is_dup:
            keep.append(ex)
        else:
            print(f"Dropped near-duplicate generated example (category={ex['category']})")
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(CATEGORY_PROMPTS.keys()) + ["all"],
                         default="all")
    parser.add_argument("--count", type=int, default=20,
                         help="Total examples to generate (per category if --category=all)")
    parser.add_argument("--adversarial-ratio", type=float, default=0.15)
    parser.add_argument("--output", default="generated_additional.jsonl")
    parser.add_argument("--sleep", type=float, default=0.5,
                         help="Seconds between API calls to avoid rate limits")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running.")

    categories = list(CATEGORY_PROMPTS.keys()) if args.category == "all" else [args.category]
    generated = []

    for cat in categories:
        for i in range(args.count):
            is_adv = random.random() < args.adversarial_ratio
            try:
                ex = generate_one(cat, is_adv)
                generated.append(ex)
                print(f"[{cat}] {i+1}/{args.count} generated (adversarial={is_adv})")
            except Exception as e:
                print(f"[{cat}] {i+1}/{args.count} FAILED: {e}")
            time.sleep(args.sleep)

    print(f"\nGenerated {len(generated)} raw examples, running dedup against existing set...")
    kept = dedup_against_existing(generated)
    print(f"Kept {len(kept)}/{len(generated)} after dedup")

    with open(args.output, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
