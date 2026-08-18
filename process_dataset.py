"""
Phase 1 pipeline step 2: quality filter (incl. ChatML schema validation),
dedup via sentence-embedding cosine similarity, stratified split.

Dedup: uses sentence-transformers (all-MiniLM-L6-v2) for true semantic
similarity, not lexical TF-IDF. This requires downloading model weights from
Hugging Face on first run — if that's unreachable (e.g. a network-restricted
sandbox), this script automatically falls back to TF-IDF and prints a clear
warning so you always know which method actually ran. On Kaggle/Colab/a
normal dev machine with internet access, the sentence-transformers path runs.
"""
import json
import random

import numpy as np

random.seed(42)

with open("raw_examples_all150.jsonl") as f:
    examples = [json.loads(line) for line in f]

# ---------------------------------------------------------------------------
# Quality filter — includes explicit ChatML schema validation
# ---------------------------------------------------------------------------
REQUIRED_ROLES = ["system", "user", "assistant"]
LEAKED_PROMPT_MARKERS = [
    "as an ai language model", "here's the json", "here is the json",
    "i'll generate", "sure, here", "```json", "{{", "[placeholder",
]

def validate_chatml_schema(ex):
    """Structural validity: correct keys, correct role order, non-empty content."""
    if "messages" not in ex:
        return False, "missing 'messages' key"
    msgs = ex["messages"]
    if len(msgs) != 3:
        return False, f"expected 3 messages, got {len(msgs)}"
    roles = [m.get("role") for m in msgs]
    if roles != REQUIRED_ROLES:
        return False, f"role order invalid: {roles}"
    for m in msgs:
        if "content" not in m or not isinstance(m["content"], str):
            return False, "missing/non-string content field"
        if len(m["content"].strip()) == 0:
            return False, "empty content field"
    return True, None

def passes_quality_filter(ex):
    valid, reason = validate_chatml_schema(ex)
    if not valid:
        return False, f"schema: {reason}"

    assistant_text = ex["messages"][-1]["content"]
    user_text = ex["messages"][-2]["content"]

    if len(assistant_text.strip()) < 20:
        return False, "assistant response too short"
    if len(user_text.strip()) < 5:
        return False, "user prompt too short"
    if assistant_text.strip().lower() == user_text.strip().lower():
        return False, "assistant echoed user prompt verbatim"

    lowered = assistant_text.lower()
    for marker in LEAKED_PROMPT_MARKERS:
        if marker in lowered:
            return False, f"leaked prompt artifact: '{marker}'"

    return True, None

filtered, rejected = [], []
for ex in examples:
    ok, reason = passes_quality_filter(ex)
    (filtered if ok else rejected).append(ex if ok else (ex["id"], reason))

print(f"Quality filter: {len(filtered)}/{len(examples)} passed")
if rejected:
    print("Rejected:", rejected)

# ---------------------------------------------------------------------------
# Dedup — sentence-transformers embeddings, with TF-IDF fallback if the
# model can't be downloaded (e.g. no Hugging Face access in this environment)
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD = 0.85
texts = [ex["messages"][-1]["content"] for ex in filtered]

def get_similarity_matrix(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=False)
        norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sim = norm @ norm.T
        print("Dedup method: sentence-transformers (all-MiniLM-L6-v2) embeddings")
        return sim
    except Exception as e:
        print(f"WARNING: sentence-transformers unavailable ({type(e).__name__}: {e}).")
        print("WARNING: Falling back to TF-IDF cosine similarity — this is NOT "
              "true semantic dedup. Run this on a machine with Hugging Face "
              "access (e.g. Kaggle/Colab) to get the real embedding-based dedup.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf)

sim_matrix = get_similarity_matrix(texts)

to_drop = set()
for i in range(len(filtered)):
    if i in to_drop:
        continue
    for j in range(i + 1, len(filtered)):
        if j in to_drop:
            continue
        if sim_matrix[i][j] > SIMILARITY_THRESHOLD:
            print(f"Near-duplicate: {filtered[i]['id']} ~ {filtered[j]['id']} "
                  f"(sim={sim_matrix[i][j]:.3f}) -> dropping {filtered[j]['id']}")
            to_drop.add(j)

deduped = [ex for i, ex in enumerate(filtered) if i not in to_drop]
print(f"Dedup: {len(deduped)}/{len(filtered)} kept (threshold={SIMILARITY_THRESHOLD})")

# ---------------------------------------------------------------------------
# Stratified split — adversarial examples reserved entirely for holdout,
# so eval and train never see adversarial patterns, and eval never leaks
# into train (split happens AFTER dedup, on disjoint index sets)
# ---------------------------------------------------------------------------
adversarial = [ex for ex in deduped if ex["is_adversarial"]]
non_adversarial = [ex for ex in deduped if not ex["is_adversarial"]]

by_category = {}
for ex in non_adversarial:
    by_category.setdefault(ex["category"], []).append(ex)

train, eval_set = [], []
for cat, items in by_category.items():
    random.shuffle(items)
    n_eval = max(1, round(len(items) * 0.15))
    eval_set.extend(items[:n_eval])
    train.extend(items[n_eval:])

# sanity check: zero overlap between splits
train_ids = {ex["id"] for ex in train}
eval_ids = {ex["id"] for ex in eval_set}
adv_ids = {ex["id"] for ex in adversarial}
assert not (train_ids & eval_ids), "LEAKAGE: train/eval overlap"
assert not (train_ids & adv_ids), "LEAKAGE: train/adversarial overlap"
assert not (eval_ids & adv_ids), "LEAKAGE: eval/adversarial overlap"
print("Leakage check passed: train/eval/adversarial are fully disjoint")

random.shuffle(train)
random.shuffle(eval_set)

print(f"\nFinal split: train={len(train)}, eval={len(eval_set)}, "
      f"adversarial_holdout={len(adversarial)}")

for name, split in [("train", train), ("eval", eval_set),
                     ("adversarial_holdout", adversarial)]:
    with open(f"{name}.jsonl", "w") as f:
        for ex in split:
            clean = {"messages": ex["messages"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
    print(f"Wrote {name}.jsonl ({len(split)} examples)")

with open("all_examples_with_metadata.jsonl", "w") as f:
    for ex in deduped:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
