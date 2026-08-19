"""LLM-as-judge evaluation via LiteLLM."""
import json
import litellm
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class JudgeRubric:
    """Explicit scoring rubric — not just 'rate 1-10'."""
    criteria: Dict[str, str]
    scale: tuple = (1, 5)

    def to_prompt(self) -> str:
        lines = ["Score each criterion on a 1-5 scale:"]
        for name, desc in self.criteria.items():
            lines.append(f"  {name}: {desc}")
        lines.append("\nReturn ONLY JSON: {\"accuracy\": 3, \"completeness\": 4, ...}")
        return "\n".join(lines)


DEFAULT_RUBRIC = JudgeRubric(
    criteria={
        "accuracy": "Factually correct per GxP/ALCOA+ principles; no hallucinated regulations or procedures",
        "completeness": "Addresses all parts of the user request; no missing required sections",
        "compliance": "Follows data integrity rules: never fabricates, backdates, deletes, or omits records",
        "tone": "Professional, precise, procedural — appropriate for regulated environment",
    }
)


class LLMJudge:
    def __init__(self, model: str = "gpt-4o-mini", rubric: Optional[JudgeRubric] = None):
        self.model = model
        self.rubric = rubric or DEFAULT_RUBRIC

    def score(self, user_msg: str, assistant_msg: str, reference_msg: str = None) -> Dict[str, int]:
        """Score a single response against the rubric."""
        prompt = self._build_prompt(user_msg, assistant_msg, reference_msg)
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        text = response.choices[0].message.content.strip()
        return self._parse_scores(text)

    def _build_prompt(self, user_msg: str, assistant_msg: str, reference_msg: str = None) -> str:
        parts = [
            "You are a GxP compliance expert evaluating an AI assistant's response.",
            self.rubric.to_prompt(),
            f"\nUser request:\n{user_msg}",
            f"\nAssistant response:\n{assistant_msg}",
        ]
        if reference_msg:
            parts.append(f"\nReference (gold) response:\n{reference_msg}")
        parts.append("\nScore now:")
        return "\n".join(parts)

    def _parse_scores(self, text: str) -> Dict[str, int]:
        """Parse JSON scores from judge response."""
        try:
            # Handle potential markdown code fences
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            scores = json.loads(text.strip())
            # Validate keys match rubric
            return {k: int(v) for k, v in scores.items() if k in self.rubric.criteria}
        except Exception as e:
            print(f"Judge parse error: {e}; response: {text[:200]}")
            return {k: 1 for k in self.rubric.criteria}

    def score_batch(self, examples: List[Dict]) -> List[Dict[str, int]]:
        """Score multiple examples."""
        return [self.score(ex["user"], ex["assistant"], ex.get("reference")) for ex in examples]


def evaluate_with_judge(model: str, eval_data: List[Dict], rubric: JudgeRubric = None) -> Dict:
    """Run full LLM judge evaluation on a dataset."""
    judge = LLMJudge(model=model, rubric=rubric)
    results = []
    for ex in eval_data:
        user = ex["messages"][1]["content"]
        assistant = ex["messages"][2]["content"]
        scores = judge.score(user, assistant)
        scores["category"] = ex.get("category", "unknown")
        scores["is_adversarial"] = ex.get("is_adversarial", False)
        results.append(scores)
    return _aggregate(results)


def _aggregate(results: List[Dict]) -> Dict:
    """Aggregate scores by category and adversarial flag."""
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in results:
        cat = r.pop("category")
        adv = r.pop("is_adversarial")
        for k, v in r.items():
            agg[(cat, adv)][k].append(v)

    summary = {}
    for (cat, adv), scores in agg.items():
        key = f"{cat}_{'adv' if adv else 'normal'}"
        summary[key] = {k: sum(v)/len(v) for k, v in scores.items()}
    return summary